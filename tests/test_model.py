"""
Testes do pipeline de Marketing Mix Modeling.

Cobre geração de dados, transformações (adstock e saturação),
treinamento do modelo e decomposição de contribuição.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_generator import generate_marketing_data
from src.model import MarketingMixModel
from src.transformations import geometric_adstock, hill_saturation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def df_marketing() -> pd.DataFrame:
    """Gera o dataset sintético uma única vez para todo o módulo."""
    return generate_marketing_data()


@pytest.fixture(scope="module")
def modelo_treinado(df_marketing: pd.DataFrame) -> MarketingMixModel:
    """Treina o modelo MMM uma única vez para todo o módulo."""
    return MarketingMixModel().fit(df_marketing)


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

COLUNAS_ESPERADAS = [
    "date", "revenue",
    "spend_meta_ads", "spend_google_ads", "spend_linkedin_ads",
    "spend_email_marketing", "spend_content_organic",
    "impressions_meta", "impressions_google",
    "clicks_meta", "clicks_google",
    "leads_total", "conversions",
    "is_holiday", "is_black_friday",
    "seasonality_index", "competitor_spend_index",
]


def test_data_generator_shape(df_marketing: pd.DataFrame) -> None:
    """Verifica se o dataset gerado tem 104 linhas e todas as colunas esperadas."""
    assert df_marketing.shape[0] == 104, (
        f"Esperado 104 semanas, obteve {df_marketing.shape[0]}"
    )
    for col in COLUNAS_ESPERADAS:
        assert col in df_marketing.columns, f"Coluna '{col}' ausente no dataset"


def test_data_no_nulls(df_marketing: pd.DataFrame) -> None:
    """Verifica se não há valores nulos no dataset gerado."""
    nulos = df_marketing.isnull().sum()
    colunas_com_nulo = nulos[nulos > 0]
    assert colunas_com_nulo.empty, (
        f"Colunas com valores nulos: {dict(colunas_com_nulo)}"
    )


def test_adstock_decay_zero() -> None:
    """Adstock com decay=0 deve retornar a série original inalterada."""
    serie = pd.Series([100.0, 200.0, 150.0, 300.0, 50.0])
    resultado = geometric_adstock(serie, decay_rate=0.0)
    pd.testing.assert_series_equal(resultado, serie, check_names=False)


def test_saturation_bounds() -> None:
    """Saturação de Hill deve sempre retornar valores entre 0 e 1."""
    serie = pd.Series([0.0, 100.0, 1_000.0, 10_000.0, 100_000.0])
    resultado = hill_saturation(serie, half_saturation=5_000, slope=2.0)
    assert (resultado >= 0.0).all(), "Saturação retornou valor negativo"
    assert (resultado <= 1.0).all(), "Saturação retornou valor acima de 1"
    assert resultado.iloc[0] == 0.0, "Saturação de investimento zero deve ser zero"


def test_model_r2(modelo_treinado: MarketingMixModel) -> None:
    """Modelo treinado deve atingir R² >= 0.80."""
    diag = modelo_treinado.get_model_diagnostics()
    assert diag["r2"] >= 0.80, (
        f"R² abaixo do esperado: {diag['r2']:.4f}"
    )


def test_contributions_sum(
    df_marketing: pd.DataFrame,
    modelo_treinado: MarketingMixModel,
) -> None:
    """Soma das contribuições (canais + baseline) deve ser próxima da receita prevista total."""
    contributions = modelo_treinado.get_channel_contributions()
    soma_contribuicoes = contributions["contribuicao_total"].sum()
    receita_prevista_total = float(modelo_treinado.fitted_values.sum())

    tolerancia = receita_prevista_total * 0.05  # 5% de margem
    diferenca = abs(soma_contribuicoes - receita_prevista_total)
    assert diferenca < tolerancia, (
        f"Diferença entre soma das contribuições (R$ {soma_contribuicoes:,.0f}) "
        f"e receita prevista (R$ {receita_prevista_total:,.0f}) "
        f"excede 5%: R$ {diferenca:,.0f}"
    )
