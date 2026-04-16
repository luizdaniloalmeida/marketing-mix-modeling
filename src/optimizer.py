"""
Otimizador de budget de marketing baseado no MarketingMixModel.

Dado um modelo MMM treinado, este módulo encontra a alocação de investimento
entre canais que maximiza a receita prevista, respeitando:
    - Um budget total (mensal) fixo.
    - Limites mínimo e máximo por canal (pisos contratuais, tetos de canal etc).

Como a resposta de cada canal é não-linear (adstock geométrico + saturação de
Hill), o otimizador assume **regime estacionário**: se o anunciante mantiver
um investimento semanal constante `w`, o adstock converge para
`w / (1 - decay)`, e a receita semanal do canal é
`coef * hill(w / (1 - decay))`.

O budget é tratado em R$ mensais (aprox. 4,333 semanas por mês); internamente
o otimizador converte para valores semanais antes de aplicar as transformações.

Uso típico:
    from src.model import MarketingMixModel
    from src.optimizer import BudgetOptimizer

    mmm = MarketingMixModel().fit(df_historico)
    opt = BudgetOptimizer(mmm)
    alocacao = opt.optimize(total_budget=280_000)
    comparativo = opt.compare(alocacao_atual, alocacao)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.model import CHANNELS, CONTEXT_FEATURES, MarketingMixModel


# ---------------------------------------------------------------------------
# Constantes e defaults de negócio
# ---------------------------------------------------------------------------

# Conversão de budget mensal para semanal (base: 52 semanas / 12 meses)
WEEKS_PER_MONTH: float = 52.0 / 12.0  # ≈ 4,333

# Budget total default (R$ por mês)
DEFAULT_TOTAL_BUDGET: float = 280_000.0

# Pisos mínimos por canal (R$ por mês)
DEFAULT_MIN_PER_CHANNEL: dict[str, float] = {
    "meta_ads": 20_000.0,
    "google_ads": 15_000.0,
    "linkedin_ads": 5_000.0,
    "email_marketing": 3_000.0,
    "content_organic": 2_000.0,
}

# Tetos máximos por canal (R$ por mês)
DEFAULT_MAX_PER_CHANNEL: dict[str, float] = {
    "meta_ads": 150_000.0,
    "google_ads": 120_000.0,
    "linkedin_ads": 60_000.0,
    "email_marketing": 40_000.0,
    "content_organic": 20_000.0,
}

# Numero de semanas do historico usado como "mes atual" de referencia
DEFAULT_REFERENCE_WEEKS: int = 4

# Budget do cenario de "budget expandido"
DEFAULT_EXPANDED_BUDGET: float = 280_000.0


class BudgetOptimizer:
    """
    Otimizador de alocação de budget entre canais de marketing.

    Encapsula a lógica de previsão de receita a partir de um vetor de
    investimento por canal e chama `scipy.optimize.minimize` (método SLSQP)
    para maximizar a receita prevista sob restrição de budget total.

    Atributos
    ---------
    model : MarketingMixModel
        Modelo MMM já treinado. Deve ter coeficientes e parâmetros de canal.
    channels : list[str]
        Lista de canais otimizados (mesma ordem usada pelo modelo).
    """

    def __init__(self, model: MarketingMixModel) -> None:
        """
        Parâmetros
        ----------
        model : MarketingMixModel
            Instância treinada. Se não foi ajustada, levanta RuntimeError na
            primeira chamada de optimize/predict.
        """
        if model.params is None:
            raise RuntimeError(
                "MarketingMixModel precisa estar treinado antes de otimizar."
            )
        self.model = model
        self.channels: list[str] = list(CHANNELS)

    # ------------------------------------------------------------------
    # Núcleo: previsão de receita a partir de alocação
    # ------------------------------------------------------------------

    def _predict_monthly_revenue(
        self, monthly_spends: np.ndarray
    ) -> float:
        """
        Calcula a receita mensal prevista pelo modelo dado um vetor de
        investimento mensal por canal, assumindo regime estacionário.

        Para cada canal:
            weekly = monthly / 4.333
            adstock_ss = weekly / (1 - decay)   (soma geométrica infinita)
            saturated = hill(adstock_ss; half_saturation, slope)
            contrib_semanal = coef_canal * saturated

        Baseline semanal = const + soma(coef_ctx * média_histórica_ctx).
        Receita mensal = (contrib_canais + baseline) * WEEKS_PER_MONTH.

        Parâmetros
        ----------
        monthly_spends : np.ndarray
            Vetor com o investimento mensal de cada canal, na mesma ordem
            de `self.channels`.

        Retorna
        -------
        float
            Receita mensal estimada (R$).
        """
        params = self.model.params
        channel_params = self.model.channel_params

        contrib_semanal = 0.0
        for i, canal in enumerate(self.channels):
            p = channel_params[canal]
            weekly = monthly_spends[i] / WEEKS_PER_MONTH
            decay = p["decay"]

            if decay >= 1.0:
                adstock_ss = weekly * 1e6  # proteção contra decay=1
            else:
                adstock_ss = weekly / (1.0 - decay)

            if adstock_ss > 0:
                saturated = 1.0 / (
                    1.0 + (p["half_saturation"] / adstock_ss) ** p["slope"]
                )
            else:
                saturated = 0.0

            coef = float(params[f"spend_{canal}_transformed"])
            contrib_semanal += coef * saturated

        # Baseline semanal: intercepto + média histórica das variáveis de contexto
        baseline_semanal = float(params["const"])
        for ctx in CONTEXT_FEATURES:
            baseline_semanal += float(params[ctx]) * float(
                self.model._X[ctx].mean()
            )

        return (contrib_semanal + baseline_semanal) * WEEKS_PER_MONTH

    # ------------------------------------------------------------------
    # Otimização
    # ------------------------------------------------------------------

    def optimize(
        self,
        total_budget: float = DEFAULT_TOTAL_BUDGET,
        min_per_channel: dict[str, float] | None = None,
        max_per_channel: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Encontra a alocação de budget que maximiza a receita prevista.

        Usa SLSQP com restrição de igualdade `sum(spends) == total_budget`
        e bounds por canal. Se nenhum dicionário de min/max for fornecido,
        aplica os defaults de negócio.

        Parâmetros
        ----------
        total_budget : float
            Budget total mensal disponível (R$).
        min_per_channel : dict[str, float] | None
            Piso de investimento por canal (R$/mês). Default: DEFAULT_MIN_PER_CHANNEL.
        max_per_channel : dict[str, float] | None
            Teto de investimento por canal (R$/mês). Default: DEFAULT_MAX_PER_CHANNEL.

        Retorna
        -------
        dict[str, float]
            Alocação ótima por canal: {canal: investimento_mensal_em_reais}.

        Raises
        ------
        ValueError
            Se o budget total for inviável (menor que a soma dos mínimos
            ou maior que a soma dos máximos).
        RuntimeError
            Se o otimizador SLSQP não convergir.
        """
        mins = {**DEFAULT_MIN_PER_CHANNEL, **(min_per_channel or {})}
        maxes = {**DEFAULT_MAX_PER_CHANNEL, **(max_per_channel or {})}

        soma_min = sum(mins[c] for c in self.channels)
        soma_max = sum(maxes[c] for c in self.channels)
        if total_budget < soma_min - 1e-6:
            raise ValueError(
                f"Budget ({total_budget:,.2f}) abaixo da soma dos minimos "
                f"({soma_min:,.2f})."
            )
        if total_budget > soma_max + 1e-6:
            raise ValueError(
                f"Budget ({total_budget:,.2f}) acima da soma dos maximos "
                f"({soma_max:,.2f})."
            )

        bounds = [(mins[c], maxes[c]) for c in self.channels]

        # Chute inicial: divide o excedente (budget − mínimos) proporcionalmente
        # à "folga" (max - min) de cada canal. Mantém viabilidade.
        folgas = np.array(
            [maxes[c] - mins[c] for c in self.channels], dtype=float
        )
        excedente = total_budget - soma_min
        x0 = np.array([mins[c] for c in self.channels], dtype=float)
        if folgas.sum() > 0 and excedente > 0:
            x0 += excedente * folgas / folgas.sum()

        def objetivo(x: np.ndarray) -> float:
            # Minimizamos a receita negativa
            return -self._predict_monthly_revenue(x)

        restricao = {
            "type": "eq",
            "fun": lambda x: float(np.sum(x) - total_budget),
        }

        resultado = minimize(
            objetivo,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[restricao],
            options={"ftol": 1e-8, "maxiter": 500, "disp": False},
        )

        if not resultado.success:
            raise RuntimeError(
                f"SLSQP nao convergiu: {resultado.message}"
            )

        return {canal: float(resultado.x[i]) for i, canal in enumerate(self.channels)}

    # ------------------------------------------------------------------
    # Comparação atual vs otimizado
    # ------------------------------------------------------------------

    def compare(
        self,
        current_allocation: dict[str, float],
        optimized_allocation: dict[str, float],
    ) -> pd.DataFrame:
        """
        Compara uma alocação atual com uma otimizada.

        Estima a receita mensal para cada cenário e detalha, por canal,
        o investimento, a receita estimada total e o ganho esperado.

        Parâmetros
        ----------
        current_allocation : dict[str, float]
            Alocação atual {canal: investimento_mensal_em_reais}.
        optimized_allocation : dict[str, float]
            Alocação otimizada {canal: investimento_mensal_em_reais}.

        Retorna
        -------
        pd.DataFrame
            Uma linha por canal + uma linha agregadora "TOTAL" com as colunas:
              - canal
              - spend_atual
              - spend_otimizado
              - delta_spend        (R$ otim - atual)
              - delta_spend_pct    (% sobre o atual)
              - receita_atual      (apenas na linha TOTAL)
              - receita_otimizada  (apenas na linha TOTAL)
              - ganho_receita      (apenas na linha TOTAL)
              - ganho_receita_pct  (apenas na linha TOTAL)
        """
        x_atual = np.array(
            [current_allocation.get(c, 0.0) for c in self.channels], dtype=float
        )
        x_otim = np.array(
            [optimized_allocation.get(c, 0.0) for c in self.channels], dtype=float
        )

        receita_atual = self._predict_monthly_revenue(x_atual)
        receita_otim = self._predict_monthly_revenue(x_otim)

        linhas: list[dict[str, Any]] = []
        for i, canal in enumerate(self.channels):
            delta = x_otim[i] - x_atual[i]
            delta_pct = 100.0 * delta / x_atual[i] if x_atual[i] > 0 else np.nan
            linhas.append(
                {
                    "canal": canal,
                    "spend_atual": x_atual[i],
                    "spend_otimizado": x_otim[i],
                    "delta_spend": delta,
                    "delta_spend_pct": delta_pct,
                    "receita_atual": np.nan,
                    "receita_otimizada": np.nan,
                    "ganho_receita": np.nan,
                    "ganho_receita_pct": np.nan,
                }
            )

        soma_atual = float(x_atual.sum())
        soma_otim = float(x_otim.sum())
        linhas.append(
            {
                "canal": "TOTAL",
                "spend_atual": soma_atual,
                "spend_otimizado": soma_otim,
                "delta_spend": soma_otim - soma_atual,
                "delta_spend_pct": (
                    100.0 * (soma_otim - soma_atual) / soma_atual
                    if soma_atual > 0
                    else np.nan
                ),
                "receita_atual": receita_atual,
                "receita_otimizada": receita_otim,
                "ganho_receita": receita_otim - receita_atual,
                "ganho_receita_pct": (
                    100.0 * (receita_otim - receita_atual) / receita_atual
                    if receita_atual > 0
                    else np.nan
                ),
            }
        )

        return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Cenários padrão para dashboard e notebooks
# ---------------------------------------------------------------------------

def current_allocation_from_history(
    df: pd.DataFrame,
    reference_weeks: int = DEFAULT_REFERENCE_WEEKS,
) -> dict[str, float]:
    """
    Extrai a alocação mensal atual somando os spends das últimas N semanas.

    Essa é a "foto" do comportamento recente de investimento usada como
    baseline de comparação nos cenários de otimização.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame histórico contendo colunas `spend_<canal>`.
    reference_weeks : int
        Quantas semanas finais do histórico agregar (default: 4 ~= 1 mês).

    Retorna
    -------
    dict[str, float]
        Alocação mensal atual {canal: R$}.
    """
    ultimas = df.tail(reference_weeks)
    return {canal: float(ultimas[f"spend_{canal}"].sum()) for canal in CHANNELS}


def run_standard_scenarios(
    model: MarketingMixModel,
    df: pd.DataFrame,
    expanded_budget: float = DEFAULT_EXPANDED_BUDGET,
    reference_weeks: int = DEFAULT_REFERENCE_WEEKS,
    min_per_channel: dict[str, float] | None = None,
    max_per_channel: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Executa os dois cenários de otimização usados no projeto.

    - Cenário 1 ("realocacao_puro"): mesmo budget total da alocação atual,
      redistribuído de forma ótima. Isola o ganho *de alocação*, removendo
      o efeito de aumentar o investimento.
    - Cenário 2 ("budget_expandido"): budget total `expanded_budget` com
      alocação ótima. Mostra o potencial combinado de realocar + investir mais.

    Parâmetros
    ----------
    model : MarketingMixModel
        Modelo treinado.
    df : pd.DataFrame
        Histórico usado para inferir a alocação atual.
    expanded_budget : float
        Budget mensal do cenário expandido.
    reference_weeks : int
        Janela histórica usada para a alocação atual (default: 4).
    min_per_channel, max_per_channel : dict | None
        Bounds por canal; se None, usa os defaults do módulo.

    Retorna
    -------
    dict[str, Any]
        Dicionário com:
          - 'atual' : alocação atual (dict)
          - 'budget_atual' : soma mensal da alocação atual
          - 'realocacao_puro' : dict com 'alocacao' e 'comparativo' (DataFrame)
          - 'budget_expandido' : dict com 'alocacao' e 'comparativo' (DataFrame)
    """
    opt = BudgetOptimizer(model)
    atual = current_allocation_from_history(df, reference_weeks=reference_weeks)
    budget_atual = sum(atual.values())

    alocacao_cenario1 = opt.optimize(
        total_budget=budget_atual,
        min_per_channel=min_per_channel,
        max_per_channel=max_per_channel,
    )
    alocacao_cenario2 = opt.optimize(
        total_budget=expanded_budget,
        min_per_channel=min_per_channel,
        max_per_channel=max_per_channel,
    )

    return {
        "atual": atual,
        "budget_atual": budget_atual,
        "realocacao_puro": {
            "alocacao": alocacao_cenario1,
            "comparativo": opt.compare(atual, alocacao_cenario1),
        },
        "budget_expandido": {
            "alocacao": alocacao_cenario2,
            "comparativo": opt.compare(atual, alocacao_cenario2),
        },
    }
