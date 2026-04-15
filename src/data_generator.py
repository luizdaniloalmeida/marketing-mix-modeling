"""
Gerador de dados sintéticos para Marketing Mix Modeling (MMM).

Criei este módulo para não divulgar dados sensíveis da empresa inicial para qual produzi o projeto.
Aqui, o módulo gera um dataset semanal realista de 2 anos (104 semanas) com investimento em mídia,
métricas intermediárias (impressões, cliques, leads) e receita resultante.

A receita é construída a partir de uma linha de base mais a contribuição de cada canal,
aplicando efeitos de Carryover (adstock) e saturação (curva de Hill),
além de sazonalidade geográfica típica da região brasileira e ruído.

Uso:
    from src.data_generator import generate_marketing_data
    df = generate_marketing_data()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configurações globais do gerador
# ---------------------------------------------------------------------------

# Semente para garantir reprodutibilidade dos dados sintéticos
SEED = 42

# Parâmetros do horizonte temporal
N_WEEKS = 104
START_DATE = "2024-01-01"

# Distribuição do budget mensal entre os canais
CHANNEL_SHARES = {
    "meta_ads": 0.40,
    "google_ads": 0.30,
    "linkedin_ads": 0.15,
    "email_marketing": 0.10,
    "content_organic": 0.05,
}

# Parâmetros de adstock (carryover): fração do efeito do investimento
# que permanece na próxima semana. Canais de branding retêm mais.
ADSTOCK_DECAY = {
    "meta_ads": 0.55,
    "google_ads": 0.30,
    "linkedin_ads": 0.60,
    "email_marketing": 0.20,
    "content_organic": 0.70,
}

# Parâmetros da curva de saturação de Hill (S-shape)
# contribuicao = beta * x^alpha / (gamma^alpha + x^alpha)
# - beta: contribuição máxima (teto) em R$ por semana
# - alpha: inclinação da curva (quanto maior, mais "S")
# - gamma: ponto de meia-saturação (em R$ de investimento)
HILL_PARAMS = {
    "meta_ads":        {"beta": 35000, "alpha": 1.5, "gamma": 18000},
    "google_ads":      {"beta": 42000, "alpha": 1.8, "gamma": 12000},
    "linkedin_ads":    {"beta": 15000, "alpha": 1.3, "gamma": 8000},
    "email_marketing": {"beta": 12000, "alpha": 2.0, "gamma": 3000},
    "content_organic": {"beta": 10000, "alpha": 2.5, "gamma": 2500},
}


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _aplicar_adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    """
    Aplica efeito de carryover (adstock geométrico) sobre uma série de gastos.

    A cada semana, o investimento efetivo é a soma do gasto atual com uma
    fração do investimento efetivo da semana anterior, simulando o efeito
    residual de campanhas passadas sobre a semana corrente.

    Parâmetros
    ----------
    spend : np.ndarray
        Série semanal de investimento em R$.
    decay : float
        Taxa de retenção entre 0 e 1 (ex.: 0.5 retém 50% da semana anterior).

    Retorna
    -------
    np.ndarray
        Série de investimento ajustada pelo adstock.
    """
    adstocked = np.zeros_like(spend, dtype=float)
    adstocked[0] = spend[0]
    for t in range(1, len(spend)):
        adstocked[t] = spend[t] + decay * adstocked[t - 1]
    return adstocked


def _aplicar_saturacao_hill(x: np.ndarray, beta: float, alpha: float, gamma: float) -> np.ndarray:
    """
    Aplica a curva de saturação de Hill sobre o investimento adstockado.

    Modela a ideia de que, acima de um certo nível de investimento, o
    retorno marginal decresce — isto é, cada real adicional contribui
    cada vez menos para a receita.

    Parâmetros
    ----------
    x : np.ndarray
        Investimento efetivo (já ajustado por adstock).
    beta : float
        Contribuição máxima possível (teto assintótico), em R$.
    alpha : float
        Inclinação da curva; valores maiores geram transição mais abrupta.
    gamma : float
        Ponto de meia-saturação: valor de x onde a contribuição é beta/2.

    Retorna
    -------
    np.ndarray
        Contribuição semanal em R$ do canal, após saturação.
    """
    x_safe = np.maximum(x, 0.0)
    return beta * (x_safe ** alpha) / (gamma ** alpha + x_safe ** alpha)


def _construir_sazonalidade(datas: pd.DatetimeIndex) -> np.ndarray:
    """
    Constrói um índice de sazonalidade mensal para o mercado brasileiro.

    Picos sazonais são aplicados em janeiro (volta às aulas/liquidações),
    junho (Dia dos Namorados/meio do ano), novembro (Black Friday) e
    dezembro (Natal). Outros meses ficam em torno de 1.0.
    """
    # Fator multiplicativo base por mês (1 = neutro)
    fator_mes = {
        1: 1.15,   # Janeiro: liquidações de início de ano
        2: 0.95,
        3: 1.00,
        4: 1.00,
        5: 1.05,
        6: 1.15,   # Junho: Dia dos Namorados
        7: 0.95,
        8: 1.00,
        9: 1.05,
        10: 1.05,
        11: 1.30,  # Novembro: Black Friday
        12: 1.25,  # Dezembro: Natal
    }
    return np.array([fator_mes[d.month] for d in datas])


def _flag_black_friday(datas: pd.DatetimeIndex) -> np.ndarray:
    """
    Marca a semana de Black Friday (última sexta-feira de novembro).
    """
    flag = np.zeros(len(datas), dtype=int)
    for i, d in enumerate(datas):
        if d.month == 11 and d.day >= 20 and d.day <= 30:
            flag[i] = 1
    return flag


def _flag_feriados(datas: pd.DatetimeIndex) -> np.ndarray:
    """
    Marca semanas com feriados nacionais brasileiros relevantes para varejo.

    Considera: Carnaval (fev/mar), Páscoa (mar/abr), Tiradentes (21/abr),
    Dia do Trabalho (01/mai), Independência (07/set), N. Sra. Aparecida
    (12/out), Finados (02/nov), Proclamação (15/nov) e Natal (25/dez).
    """
    flag = np.zeros(len(datas), dtype=int)
    feriados_fixos = [(4, 21), (5, 1), (9, 7), (10, 12), (11, 2), (11, 15), (12, 25)]
    for i, d in enumerate(datas):
        # Semana contém algum feriado fixo?
        inicio = d
        fim = d + pd.Timedelta(days=6)
        for mes, dia in feriados_fixos:
            for ano in (inicio.year, fim.year):
                feriado = pd.Timestamp(year=ano, month=mes, day=dia)
                if inicio <= feriado <= fim:
                    flag[i] = 1
    return flag


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def generate_marketing_data(
    n_weeks: int = N_WEEKS,
    start_date: str = START_DATE,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Gera um DataFrame sintético semanal para Marketing Mix Modeling.

    A receita é construída da seguinte forma:
        receita = baseline * sazonalidade
                  + soma(contribuicao_canal_i)
                  + efeitos_promocionais (Black Friday)
                  - pressao_competitiva
                  + ruido_gaussiano

    Cada contribuição de canal passa por adstock (carryover) e saturação
    (curva de Hill), reproduzindo o comportamento observado em campanhas
    reais de marketing.

    Parâmetros
    ----------
    n_weeks : int
        Número de semanas a gerar (padrão: 104, equivalente a 2 anos).
    start_date : str
        Data inicial no formato 'YYYY-MM-DD'.
    seed : int
        Semente para reprodutibilidade.

    Retorna
    -------
    pd.DataFrame
        Dataset com colunas de data, investimento por canal, métricas
        intermediárias, receita e variáveis de controle.
    """
    rng = np.random.default_rng(seed)

    # Eixo temporal semanal (cada linha representa uma semana que inicia na data)
    datas = pd.date_range(start=start_date, periods=n_weeks, freq="W-MON")

    # -------------------------------------------------------------------
    # 1) Budget mensal e alocação semanal por canal
    # -------------------------------------------------------------------
    # Budget mensal total amostrado entre R$ 50k e R$ 80k, com leve
    # tendência de crescimento ao longo dos 2 anos (maturação da operação)
    tendencia = np.linspace(1.0, 1.15, n_weeks)
    budget_mensal_base = rng.uniform(50_000, 80_000, size=n_weeks) * tendencia
    # Convertemos orçamento mensal em semanal (aprox. 4.33 semanas por mês)
    budget_semanal = budget_mensal_base / 4.33

    # Sazonalidade aplicada ao budget (empresa investe mais em picos)
    sazonalidade = _construir_sazonalidade(datas)
    budget_semanal = budget_semanal * (0.8 + 0.3 * (sazonalidade - 1.0))

    # Aloca o budget semanal entre os canais, com pequena variação aleatória
    spends: dict[str, np.ndarray] = {}
    for canal, share in CHANNEL_SHARES.items():
        ruido_share = rng.normal(1.0, 0.08, size=n_weeks)  # +/- 8% de variação
        spends[canal] = np.maximum(budget_semanal * share * ruido_share, 0.0)

    # -------------------------------------------------------------------
    # 2) Métricas intermediárias (impressões, cliques, leads, conversões)
    # -------------------------------------------------------------------
    # CPMs e CTRs aproximados por plataforma (mercado brasileiro)
    # Meta Ads: CPM ~R$ 15, CTR ~1.2%
    impressions_meta = (spends["meta_ads"] / 15.0) * 1000.0 * rng.normal(1.0, 0.08, n_weeks)
    clicks_meta = impressions_meta * 0.012 * rng.normal(1.0, 0.10, n_weeks)

    # Google Ads: CPM ~R$ 25, CTR ~3.5% (search tem CTR mais alto)
    impressions_google = (spends["google_ads"] / 25.0) * 1000.0 * rng.normal(1.0, 0.08, n_weeks)
    clicks_google = impressions_google * 0.035 * rng.normal(1.0, 0.10, n_weeks)

    impressions_meta = np.maximum(impressions_meta, 0).astype(int)
    impressions_google = np.maximum(impressions_google, 0).astype(int)
    clicks_meta = np.maximum(clicks_meta, 0).astype(int)
    clicks_google = np.maximum(clicks_google, 0).astype(int)

    # -------------------------------------------------------------------
    # 3) Variáveis de controle
    # -------------------------------------------------------------------
    is_black_friday = _flag_black_friday(datas)
    is_holiday = _flag_feriados(datas)

    # Índice competitivo: concorrentes investem mais em picos sazonais
    competitor_spend_index = sazonalidade * rng.normal(1.0, 0.10, n_weeks)
    competitor_spend_index = np.clip(competitor_spend_index, 0.7, 1.6)

    # -------------------------------------------------------------------
    # 4) Contribuição de cada canal para a receita (adstock + saturação)
    # -------------------------------------------------------------------
    contribuicoes: dict[str, np.ndarray] = {}
    for canal in CHANNEL_SHARES.keys():
        spend_adstocked = _aplicar_adstock(spends[canal], ADSTOCK_DECAY[canal])
        params = HILL_PARAMS[canal]
        contribuicoes[canal] = _aplicar_saturacao_hill(
            spend_adstocked, params["beta"], params["alpha"], params["gamma"]
        )

    # -------------------------------------------------------------------
    # 5) Receita final: baseline + canais + efeitos + ruído
    # -------------------------------------------------------------------
    # Baseline entre R$ 80k e R$ 120k (vendas orgânicas não atribuíveis)
    baseline = rng.uniform(80_000, 120_000, size=n_weeks)

    # Aplica sazonalidade sobre o baseline (mercado como um todo se move)
    baseline_sazonal = baseline * sazonalidade

    # Soma as contribuições dos canais de mídia
    contribuicao_canais = sum(contribuicoes.values())

    # Efeito adicional de Black Friday sobre a receita (pico promocional)
    efeito_black_friday = is_black_friday * rng.uniform(35_000, 55_000, size=n_weeks)

    # Pressão competitiva reduz ligeiramente a receita (quanto maior, mais caro)
    pressao_competitiva = (competitor_spend_index - 1.0) * 15_000

    # Ruído gaussiano multiplicativo (~12% de variação)
    ruido = rng.normal(1.0, 0.12, size=n_weeks)

    revenue = (
        baseline_sazonal
        + contribuicao_canais
        + efeito_black_friday
        - pressao_competitiva
    ) * ruido
    revenue = np.maximum(revenue, 0)

    # -------------------------------------------------------------------
    # 6) Leads e conversões derivados da receita e dos cliques
    # -------------------------------------------------------------------
    # Leads dependem de cliques pagos + um componente orgânico
    leads_total = (
        0.08 * clicks_meta
        + 0.12 * clicks_google
        + 0.0005 * spends["email_marketing"]
        + 0.0003 * spends["content_organic"]
    ) * rng.normal(1.0, 0.10, n_weeks)
    leads_total = np.maximum(leads_total, 0).astype(int)

    # Conversões ~ taxa média de 8% sobre os leads, com variação
    conversions = leads_total * rng.uniform(0.06, 0.10, size=n_weeks)
    conversions = np.maximum(conversions, 0).astype(int)

    # -------------------------------------------------------------------
    # 7) Montagem do DataFrame final
    # -------------------------------------------------------------------
    df = pd.DataFrame({
        "date": datas,
        "revenue": np.round(revenue, 2),
        "spend_meta_ads": np.round(spends["meta_ads"], 2),
        "spend_google_ads": np.round(spends["google_ads"], 2),
        "spend_linkedin_ads": np.round(spends["linkedin_ads"], 2),
        "spend_email_marketing": np.round(spends["email_marketing"], 2),
        "spend_content_organic": np.round(spends["content_organic"], 2),
        "impressions_meta": impressions_meta,
        "impressions_google": impressions_google,
        "clicks_meta": clicks_meta,
        "clicks_google": clicks_google,
        "leads_total": leads_total,
        "conversions": conversions,
        "is_holiday": is_holiday,
        "is_black_friday": is_black_friday,
        "seasonality_index": np.round(sazonalidade, 3),
        "competitor_spend_index": np.round(competitor_spend_index, 3),
    })

    return df


def _salvar_csv(df: pd.DataFrame, caminho: Path) -> None:
    """Salva o DataFrame em CSV, criando a pasta de destino se necessário."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False, encoding="utf-8")


if __name__ == "__main__":
    # Gera os dados sintéticos e salva no caminho padrão do projeto
    df_marketing = generate_marketing_data()

    caminho_saida = Path(__file__).resolve().parent.parent / "data" / "raw" / "marketing_data.csv"
    _salvar_csv(df_marketing, caminho_saida)

    # Resumo rápido para conferência no terminal
    print(f"Dados gerados: {len(df_marketing)} semanas")
    print(f"Período: {df_marketing['date'].min().date()} a {df_marketing['date'].max().date()}")
    print(f"Receita média semanal: R$ {df_marketing['revenue'].mean():,.2f}")
    print(f"Investimento total: R$ {df_marketing.filter(like='spend_').sum().sum():,.2f}")
    print(f"Arquivo salvo em: {caminho_saida}")
