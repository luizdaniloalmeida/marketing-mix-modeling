"""
Transformações de mídia para Marketing Mix Modeling (MMM).

Este módulo implementa as duas transformações clássicas aplicadas ao investimento
em mídia antes da modelagem estatística:

1. Adstock geométrico: captura o efeito de carryover, ou seja, a parcela do
   investimento de uma semana que continua gerando impacto nas semanas seguintes.

2. Saturação de Hill: captura o efeito de retornos decrescentes — a partir de
   certo ponto, dobrar o investimento não dobra o resultado.

Uso típico:
    from src.transformations import apply_all_transformations, DEFAULT_CHANNEL_PARAMS
    df_transformado = apply_all_transformations(df, DEFAULT_CHANNEL_PARAMS)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parâmetros default por canal
# ---------------------------------------------------------------------------
# Cada canal possui três hiperparâmetros:
#   - decay: taxa de carryover do adstock (quanto do efeito permanece na semana seguinte)
#   - half_saturation: nível de investimento no qual a resposta atinge metade do máximo
#   - slope: inclinação da curva Hill (quão rápido satura)

DEFAULT_CHANNEL_PARAMS: dict[str, dict[str, float]] = {
    "meta_ads": {"decay": 0.4, "half_saturation": 20_000, "slope": 2.0},
    "google_ads": {"decay": 0.2, "half_saturation": 15_000, "slope": 2.5},
    "linkedin_ads": {"decay": 0.5, "half_saturation": 8_000, "slope": 1.8},
    "email_marketing": {"decay": 0.3, "half_saturation": 5_000, "slope": 2.0},
    "content_organic": {"decay": 0.7, "half_saturation": 3_000, "slope": 1.5},
}


# ---------------------------------------------------------------------------
# Transformações individuais
# ---------------------------------------------------------------------------

def geometric_adstock(series: pd.Series, decay_rate: float) -> pd.Series:
    """
    Aplica adstock geométrico a uma série temporal de investimento.

    O adstock modela o efeito de carryover: parte do investimento de uma semana
    continua gerando impacto nas semanas seguintes, com decaimento geométrico.

    Fórmula recursiva:
        adstock_t = spend_t + decay_rate * adstock_{t-1}

    Parâmetros
    ----------
    series : pd.Series
        Série temporal com o investimento semanal no canal.
    decay_rate : float
        Taxa de decaimento entre 0 e 1. Valores mais altos indicam maior
        retenção do efeito ao longo do tempo (típico de canais de branding).

    Retorna
    -------
    pd.Series
        Série com o efeito de carryover acumulado, preservando índice e nome.

    Raises
    ------
    ValueError
        Se decay_rate estiver fora do intervalo [0, 1].
    """
    if not 0.0 <= decay_rate <= 1.0:
        raise ValueError(
            f"decay_rate deve estar entre 0 e 1, recebido: {decay_rate}"
        )

    valores = series.to_numpy(dtype=float)
    adstocked = np.zeros_like(valores)
    adstocked[0] = valores[0]
    for t in range(1, len(valores)):
        adstocked[t] = valores[t] + decay_rate * adstocked[t - 1]

    return pd.Series(adstocked, index=series.index, name=series.name)


def hill_saturation(
    series: pd.Series,
    half_saturation: float,
    slope: float,
) -> pd.Series:
    """
    Aplica a curva de saturação de Hill a uma série.

    A função Hill modela retornos decrescentes: quanto maior o investimento,
    menor o ganho marginal. O resultado é normalizado entre 0 e 1, onde 0,5
    corresponde exatamente ao nível de `half_saturation`.

    Fórmula:
        saturated = 1 / (1 + (half_saturation / x) ** slope)

    Parâmetros
    ----------
    series : pd.Series
        Série temporal (tipicamente já com adstock aplicado).
    half_saturation : float
        Nível de investimento no qual a resposta atinge 50% do máximo.
        Deve ser estritamente positivo.
    slope : float
        Inclinação da curva. Valores maiores geram transição mais abrupta
        entre a região linear e o platô de saturação.

    Retorna
    -------
    pd.Series
        Série saturada com valores no intervalo [0, 1]. Posições onde o
        investimento original é 0 retornam 0.
    """
    if half_saturation <= 0:
        raise ValueError(
            f"half_saturation deve ser positivo, recebido: {half_saturation}"
        )

    x = series.to_numpy(dtype=float)
    saturated = np.zeros_like(x)
    mascara_positiva = x > 0
    saturated[mascara_positiva] = 1.0 / (
        1.0 + (half_saturation / x[mascara_positiva]) ** slope
    )

    return pd.Series(saturated, index=series.index, name=series.name)


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def apply_all_transformations(
    df: pd.DataFrame,
    params_dict: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """
    Aplica adstock e saturação em todos os canais de investimento do DataFrame.

    Para cada canal presente em `params_dict`, busca a coluna `spend_<canal>`,
    aplica o adstock geométrico com a `decay` configurada e, em seguida, a
    saturação de Hill com `half_saturation` e `slope`. O resultado é gravado
    em uma nova coluna chamada `spend_<canal>_transformed`.

    O DataFrame original não é modificado — uma cópia é retornada.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com colunas de investimento no formato `spend_<canal>`.
    params_dict : dict[str, dict[str, float]] | None
        Dicionário de parâmetros por canal. Cada entrada deve conter as
        chaves `decay`, `half_saturation` e `slope`. Se None, utiliza
        `DEFAULT_CHANNEL_PARAMS`.

    Retorna
    -------
    pd.DataFrame
        Cópia do DataFrame com as colunas transformadas adicionadas.

    Raises
    ------
    KeyError
        Se alguma coluna `spend_<canal>` esperada não existir no DataFrame.
    """
    if params_dict is None:
        params_dict = DEFAULT_CHANNEL_PARAMS

    df_out = df.copy()

    for canal, params in params_dict.items():
        coluna_spend = f"spend_{canal}"
        if coluna_spend not in df_out.columns:
            raise KeyError(
                f"Coluna esperada '{coluna_spend}' não encontrada no DataFrame."
            )

        adstocked = geometric_adstock(df_out[coluna_spend], params["decay"])
        saturated = hill_saturation(
            adstocked,
            half_saturation=params["half_saturation"],
            slope=params["slope"],
        )
        df_out[f"{coluna_spend}_transformed"] = saturated

    return df_out
