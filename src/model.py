"""
Modelo de Marketing Mix Modeling (MMM) baseado em regressão linear.

Este módulo encapsula o pipeline completo de modelagem:
    1. Preparação de features (adstock + saturação + variáveis de contexto).
    2. Ajuste da regressão via OLS (Statsmodels).
    3. Fallback para otimização restrita (SciPy) quando algum coeficiente de
       canal sai negativo — isso não faz sentido em marketing, logo forçamos
       não-negatividade apenas nos canais.
    4. Diagnósticos (R², MAPE, MAE, Durbin-Watson, VIF).
    5. Decomposição de contribuição por canal e cálculo de ROI.

Uso típico:
    from src.model import MarketingMixModel
    mmm = MarketingMixModel()
    mmm.fit(df)
    mmm.summary()
    contribuicoes = mmm.get_channel_contributions()
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from src.transformations import (
    DEFAULT_CHANNEL_PARAMS,
    apply_all_transformations,
)


# ---------------------------------------------------------------------------
# Constantes do modelo
# ---------------------------------------------------------------------------

# Canais de marketing considerados na modelagem (coincidem com DEFAULT_CHANNEL_PARAMS)
CHANNELS: list[str] = list(DEFAULT_CHANNEL_PARAMS.keys())

# Variáveis de contexto que entram como controle na regressão
CONTEXT_FEATURES: list[str] = [
    "seasonality_index",
    "is_holiday",
    "is_black_friday",
    "competitor_spend_index",
    "trend",
]


class MarketingMixModel:
    """
    Modelo de Marketing Mix Modeling (MMM).

    Ajusta uma regressão linear que explica a receita semanal em função do
    investimento transformado em cada canal (adstock + saturação) somado a
    variáveis de controle (sazonalidade, feriado, tendência).

    Atributos principais
    --------------------
    params : pd.Series
        Coeficientes estimados (incluindo o intercepto).
    fitted_values : np.ndarray
        Valores previstos no conjunto de treino.
    residuals : np.ndarray
        Resíduos do ajuste (y - ŷ).
    results : statsmodels.regression.linear_model.RegressionResultsWrapper | None
        Objeto de resultado do OLS; None quando o fallback restrito foi usado.
    is_constrained : bool
        True se foi necessário recorrer ao otimizador com bounds.
    """

    def __init__(self) -> None:
        """Inicializa o modelo em estado não ajustado."""
        self.channel_params: dict[str, dict[str, float]] = DEFAULT_CHANNEL_PARAMS
        self.channel_features: list[str] = [f"spend_{c}_transformed" for c in CHANNELS]
        self.feature_names: list[str] = self.channel_features + CONTEXT_FEATURES

        self.params: pd.Series | None = None
        self.fitted_values: np.ndarray | None = None
        self.residuals: np.ndarray | None = None
        self.results: Any | None = None
        self.is_constrained: bool = False

        self._X: pd.DataFrame | None = None  # design matrix sem constante
        self._X_const: pd.DataFrame | None = None  # design matrix com constante
        self._y: pd.Series | None = None
        self._df_prepared: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Pipeline de features
    # ------------------------------------------------------------------

    def prepare_features(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Aplica transformações de mídia e monta a matriz de features X e o alvo y.

        Passos:
            1. Aplica adstock + saturação em cada canal (coluna `_transformed`).
            2. Adiciona a coluna `trend` (índice temporal 0..N-1).
            3. Seleciona X = [canais transformados, seasonality_index, is_holiday, trend]
               e y = revenue.

        Parâmetros
        ----------
        df : pd.DataFrame
            DataFrame bruto contendo colunas `spend_<canal>`, `revenue`,
            `seasonality_index` e `is_holiday`.

        Retorna
        -------
        tuple[pd.DataFrame, pd.Series]
            (X, y) prontos para serem usados em fit/predict.
        """
        df_t = apply_all_transformations(df, self.channel_params)

        if "trend" not in df_t.columns:
            df_t = df_t.reset_index(drop=True)
            df_t["trend"] = np.arange(len(df_t))

        X = df_t[self.feature_names].copy()
        y = df_t["revenue"].copy() if "revenue" in df_t.columns else None

        self._df_prepared = df_t
        return X, y

    # ------------------------------------------------------------------
    # Ajuste
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "MarketingMixModel":
        """
        Treina o modelo via OLS e aplica fallback restrito se necessário.

        Se algum coeficiente de canal sair negativo, refaz o ajuste com
        `scipy.optimize.minimize` impondo bounds `(0, None)` apenas nos canais
        (intercepto e variáveis de contexto permanecem livres).

        Parâmetros
        ----------
        df : pd.DataFrame
            DataFrame de treino contendo `revenue` e as colunas de spend.

        Retorna
        -------
        MarketingMixModel
            Retorna self para permitir encadeamento.
        """
        X, y = self.prepare_features(df)
        if y is None:
            raise ValueError("Coluna 'revenue' não encontrada no DataFrame.")

        X_const = sm.add_constant(X, has_constant="add")
        ols = sm.OLS(y.to_numpy(), X_const.to_numpy()).fit()

        params_ols = pd.Series(ols.params, index=X_const.columns)
        canais_negativos = [
            c for c in self.channel_features if params_ols.get(c, 0.0) < 0
        ]

        if canais_negativos:
            params_final = self._fit_constrained(X_const, y, params_ols)
            self.is_constrained = True
            self.results = None
        else:
            params_final = params_ols
            self.is_constrained = False
            self.results = ols

        self.params = params_final
        self._X = X
        self._X_const = X_const
        self._y = y
        self.fitted_values = (X_const.to_numpy() @ params_final.to_numpy())
        self.residuals = y.to_numpy() - self.fitted_values

        return self

    def _fit_constrained(
        self,
        X_const: pd.DataFrame,
        y: pd.Series,
        x0: pd.Series,
    ) -> pd.Series:
        """
        Ajuste restrito via L-BFGS-B com bounds não-negativos nos canais.

        Minimiza a soma dos quadrados dos resíduos. Intercepto e variáveis de
        contexto ficam livres; canais ficam em `[0, +∞)`.
        """
        X_mat = X_const.to_numpy()
        y_vec = y.to_numpy()
        colunas = list(X_const.columns)

        bounds = []
        for nome in colunas:
            if nome in self.channel_features:
                bounds.append((0.0, None))
            else:
                bounds.append((None, None))

        chute_inicial = x0.to_numpy().copy()
        for i, nome in enumerate(colunas):
            if nome in self.channel_features and chute_inicial[i] < 0:
                chute_inicial[i] = 0.0

        def sse(beta: np.ndarray) -> float:
            residuos = y_vec - X_mat @ beta
            return float(np.dot(residuos, residuos))

        def grad(beta: np.ndarray) -> np.ndarray:
            return -2.0 * X_mat.T @ (y_vec - X_mat @ beta)

        resultado = minimize(
            sse,
            x0=chute_inicial,
            jac=grad,
            method="L-BFGS-B",
            bounds=bounds,
        )

        return pd.Series(resultado.x, index=colunas)

    # ------------------------------------------------------------------
    # Predição
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Gera predições de receita para um DataFrame novo.

        Parâmetros
        ----------
        df : pd.DataFrame
            DataFrame com as mesmas colunas usadas no treino.

        Retorna
        -------
        np.ndarray
            Vetor com as receitas previstas.
        """
        self._check_fitted()
        X, _ = self.prepare_features(df)
        X_const = sm.add_constant(X, has_constant="add")
        X_const = X_const[self._X_const.columns]  # garante ordem consistente
        return X_const.to_numpy() @ self.params.to_numpy()

    # ------------------------------------------------------------------
    # Contribuição e ROI
    # ------------------------------------------------------------------

    def get_channel_contributions(self) -> pd.DataFrame:
        """
        Decompõe a receita prevista em contribuição por canal.

        A contribuição de cada canal na semana t é `coef_canal * X_transformado_t`.
        O baseline (intercepto + variáveis de contexto) também é reportado para
        conferência de totais.

        Retorna
        -------
        pd.DataFrame
            DataFrame com uma linha por canal e as colunas:
              - canal
              - contribuicao_total : soma das contribuições (R$ no período)
              - contribuicao_semanal_media : média por semana (R$)
              - contribuicao_pct : % da receita total
              - investimento_total : soma do spend bruto (R$)
              - roi : contribuicao_total / investimento_total

            Inclui também uma linha agregada `baseline_e_contexto` para
            sazonalidade, feriado, tendência e intercepto.
        """
        self._check_fitted()

        df = self._df_prepared
        receita_total = float(self._y.sum())
        linhas: list[dict[str, Any]] = []

        for canal, coluna_transf in zip(CHANNELS, self.channel_features):
            coef = float(self.params[coluna_transf])
            contrib_semanal = coef * self._X[coluna_transf].to_numpy()
            contrib_total = float(contrib_semanal.sum())
            investimento = float(df[f"spend_{canal}"].sum())
            roi = contrib_total / investimento if investimento > 0 else np.nan

            linhas.append(
                {
                    "canal": canal,
                    "contribuicao_total": contrib_total,
                    "contribuicao_semanal_media": contrib_total / len(df),
                    "contribuicao_pct": 100.0 * contrib_total / receita_total,
                    "investimento_total": investimento,
                    "roi": roi,
                }
            )

        # Baseline agregado (intercepto + contexto)
        baseline = float(self.params["const"]) * len(df)
        for ctx in CONTEXT_FEATURES:
            baseline += float(self.params[ctx]) * float(self._X[ctx].sum())

        linhas.append(
            {
                "canal": "baseline_e_contexto",
                "contribuicao_total": baseline,
                "contribuicao_semanal_media": baseline / len(df),
                "contribuicao_pct": 100.0 * baseline / receita_total,
                "investimento_total": np.nan,
                "roi": np.nan,
            }
        )

        return pd.DataFrame(linhas)

    # ------------------------------------------------------------------
    # Diagnósticos
    # ------------------------------------------------------------------

    def get_model_diagnostics(self) -> dict[str, Any]:
        """
        Calcula métricas de qualidade do ajuste.

        Retorna
        -------
        dict[str, Any]
            Dicionário com:
              - r2, adj_r2 : coeficiente de determinação e ajustado
              - mape : erro percentual absoluto médio (%)
              - mae : erro absoluto médio (R$)
              - rmse : raiz do erro quadrático médio (R$)
              - durbin_watson : estatística de autocorrelação dos resíduos
              - vif : dict {feature: valor} com o VIF de cada feature
              - n_obs, n_features : tamanhos do problema
              - is_constrained : indica se houve fallback com bounds
        """
        self._check_fitted()

        y = self._y.to_numpy()
        y_hat = self.fitted_values
        residuos = self.residuals
        n = len(y)
        k = len(self.feature_names)  # sem contar a constante

        ss_res = float(np.sum(residuos ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else np.nan

        mae = float(np.mean(np.abs(residuos)))
        rmse = float(np.sqrt(ss_res / n))
        mask_nonzero = y != 0
        mape = float(
            np.mean(np.abs(residuos[mask_nonzero] / y[mask_nonzero])) * 100.0
        ) if mask_nonzero.any() else np.nan

        dw = float(durbin_watson(residuos))

        # VIF — calculado sobre a matriz com constante, mas reportado apenas
        # para as features de interesse (VIF da constante não é informativo).
        X_mat = self._X_const.to_numpy()
        vif = {}
        colunas = list(self._X_const.columns)
        for i, nome in enumerate(colunas):
            if nome == "const":
                continue
            try:
                vif[nome] = float(variance_inflation_factor(X_mat, i))
            except Exception:
                vif[nome] = np.nan

        return {
            "r2": r2,
            "adj_r2": adj_r2,
            "mape": mape,
            "mae": mae,
            "rmse": rmse,
            "durbin_watson": dw,
            "vif": vif,
            "n_obs": n,
            "n_features": k,
            "is_constrained": self.is_constrained,
        }

    # ------------------------------------------------------------------
    # Sumário formatado
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """
        Imprime um relatório formatado com coeficientes, diagnósticos e
        contribuição por canal.
        """
        self._check_fitted()
        diag = self.get_model_diagnostics()
        contrib = self.get_channel_contributions()

        largura = 74
        linha = "=" * largura
        sublinha = "-" * largura

        print(linha)
        print("MARKETING MIX MODEL — RESUMO".center(largura))
        print(linha)

        modo = (
            "OLS com bounds (L-BFGS-B)"
            if self.is_constrained
            else "OLS (Statsmodels)"
        )
        print(f"Método de ajuste      : {modo}")
        print(f"Observações           : {diag['n_obs']}")
        print(f"Features              : {diag['n_features']}")

        print()
        print("QUALIDADE DO AJUSTE".center(largura))
        print(sublinha)
        print(f"R²                    : {diag['r2']:.4f}")
        print(f"R² ajustado           : {diag['adj_r2']:.4f}")
        print(f"MAPE                  : {diag['mape']:.2f}%")
        print(f"MAE                   : R$ {diag['mae']:,.2f}")
        print(f"RMSE                  : R$ {diag['rmse']:,.2f}")
        print(f"Durbin-Watson         : {diag['durbin_watson']:.3f}  "
              f"(~2 indica ausencia de autocorrelacao)")

        print()
        print("COEFICIENTES".center(largura))
        print(sublinha)
        for nome, valor in self.params.items():
            print(f"  {nome:<38s} {valor:>18,.4f}")

        print()
        print("MULTICOLINEARIDADE (VIF)".center(largura))
        print(sublinha)
        for nome, valor in diag["vif"].items():
            alerta = "  <-- ALTO" if valor > 10 else ""
            print(f"  {nome:<38s} {valor:>10.2f}{alerta}")

        print()
        print("CONTRIBUICAO POR CANAL".center(largura))
        print(sublinha)
        header = (
            f"  {'canal':<22s}{'contrib.(R$)':>16s}"
            f"{'% receita':>12s}{'ROI':>10s}"
        )
        print(header)
        for _, row in contrib.iterrows():
            roi_txt = f"{row['roi']:.2f}x" if pd.notna(row["roi"]) else "  —"
            print(
                f"  {row['canal']:<22s}"
                f"{row['contribuicao_total']:>16,.0f}"
                f"{row['contribuicao_pct']:>11.1f}%"
                f"{roi_txt:>10s}"
            )
        print(linha)

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Garante que o modelo foi treinado antes de consultas."""
        if self.params is None:
            raise RuntimeError(
                "Modelo ainda não foi ajustado. Chame .fit(df) antes."
            )
