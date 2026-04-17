"""
Módulo de visualizações para Marketing Mix Modeling (MMM).

Gera gráficos interativos (Plotly) e versões estáticas em PNG (Matplotlib)
para análise de contribuição de canais, curvas de resposta, diagnósticos
do modelo e comparação de cenários de otimização de budget.

Cada função pública retorna uma go.Figure do Plotly e salva automaticamente
o PNG correspondente em docs/images/.

Uso típico:
    from src.visualizations import plot_roi_by_channel
    fig = plot_roi_by_channel(contributions)
    fig.show()  # abre no navegador
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from src.model import CHANNELS, CONTEXT_FEATURES, MarketingMixModel
from src.transformations import DEFAULT_CHANNEL_PARAMS, geometric_adstock


# ---------------------------------------------------------------------------
# Constantes visuais
# ---------------------------------------------------------------------------

CHANNEL_LABELS: dict[str, str] = {
    "meta_ads": "Meta Ads",
    "google_ads": "Google Ads",
    "linkedin_ads": "LinkedIn Ads",
    "email_marketing": "Email Marketing",
    "content_organic": "Conteúdo Orgânico",
    "baseline_e_contexto": "Baseline",
}

CHANNEL_COLORS: dict[str, str] = {
    "meta_ads": "#1877F2",
    "google_ads": "#34A853",
    "linkedin_ads": "#0A66C2",
    "email_marketing": "#FF6B35",
    "content_organic": "#7C3AED",
    "baseline_e_contexto": "#94A3B8",
}

IMAGES_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans"],
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Utilitários internos
# ---------------------------------------------------------------------------

def _garantir_diretorio() -> None:
    """Cria o diretório docs/images se ainda não existir."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _label(canal: str) -> str:
    """Retorna o nome de exibição de um canal."""
    return CHANNEL_LABELS.get(canal, canal)


def _cor(canal: str) -> str:
    """Retorna a cor da paleta oficial do canal."""
    return CHANNEL_COLORS.get(canal, "#333333")


def _hex_to_rgb(hex_cor: str) -> tuple[int, int, int]:
    """Converte hex para tupla RGB."""
    h = hex_cor.lstrip("#")
    return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _cor_clara(canal: str, alpha: float = 0.25) -> str:
    """Retorna a cor do canal em formato rgba (compatível com Plotly)."""
    r, g, b = _hex_to_rgb(_cor(canal))
    return f"rgba({r},{g},{b},{alpha})"


def _cor_clara_mpl(canal: str, alpha: float = 0.25) -> tuple[float, ...]:
    """Retorna a cor do canal como tupla RGBA (compatível com Matplotlib)."""
    r, g, b = _hex_to_rgb(_cor(canal))
    return (r / 255, g / 255, b / 255, alpha)


def _formatar_reais_mpl(x: float, _pos: int | None = None) -> str:
    """Formatter do Matplotlib para valores em reais."""
    return f"R$ {x:,.0f}"


_fmt_reais = mticker.FuncFormatter(_formatar_reais_mpl)


def _estilo_eixos(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Aplica estilo padrão aos eixos: remove bordas e adiciona grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.3)


def _ticks_datas(ax: plt.Axes, datas: np.ndarray, n: int) -> None:
    """Configura ticks trimestrais no eixo X a partir de datas."""
    posicoes = list(range(0, n, 13))
    rotulos = [pd.Timestamp(datas[i]).strftime("%b/%Y") for i in posicoes]
    ax.set_xticks(posicoes)
    ax.set_xticklabels(rotulos, rotation=45, ha="right")


# ---------------------------------------------------------------------------
# 1. Waterfall — Decomposição de Receita
# ---------------------------------------------------------------------------

def plot_revenue_decomposition_waterfall(
    contributions: pd.DataFrame,
) -> go.Figure:
    """
    Gráfico waterfall de decomposição da receita por canal.

    Mostra como cada canal de marketing contribui para a receita total,
    partindo do baseline e acumulando a contribuição de cada canal até
    chegar à receita total prevista pelo modelo.

    Parâmetros
    ----------
    contributions : pd.DataFrame
        DataFrame retornado por ``MarketingMixModel.get_channel_contributions()``.

    Retorna
    -------
    go.Figure
        Figura Plotly interativa.
        PNG salva em ``docs/images/waterfall_decomposicao_receita.png``.
    """
    _garantir_diretorio()

    # Separar baseline e canais; ordenar canais por contribuição
    mask_base = contributions["canal"] == "baseline_e_contexto"
    baseline_val = float(contributions.loc[mask_base, "contribuicao_total"].iloc[0])
    canais_df = (
        contributions[~mask_base]
        .sort_values("contribuicao_total", ascending=False)
    )

    nomes = ["Baseline"]
    valores = [baseline_val]
    cores = [_cor("baseline_e_contexto")]

    for _, row in canais_df.iterrows():
        nomes.append(_label(row["canal"]))
        valores.append(float(row["contribuicao_total"]))
        cores.append(_cor(row["canal"]))

    receita_total = sum(valores)

    # ---- Plotly (barras flutuantes manuais para manter cores por canal) ----
    fig = go.Figure()
    acumulado = 0.0

    for i, (nome, val, cor) in enumerate(zip(nomes, valores, cores)):
        fig.add_trace(go.Bar(
            x=[nome], y=[val], base=[acumulado],
            marker_color=cor, name=nome,
            text=f"R$ {val:,.0f}", textposition="outside",
            showlegend=False, width=0.55,
        ))
        acumulado += val

    # Barra de total
    fig.add_trace(go.Bar(
        x=["Receita Total"], y=[receita_total], base=[0],
        marker_color="#1E293B", name="Receita Total",
        text=f"R$ {receita_total:,.0f}", textposition="outside",
        showlegend=False, width=0.55,
    ))

    # Linhas conectoras
    acum = 0.0
    for i in range(len(valores) - 1):
        acum += valores[i]
        fig.add_shape(
            type="line", x0=i + 0.3, x1=i + 0.7, y0=acum, y1=acum,
            line=dict(color="#94A3B8", width=1, dash="dot"),
        )

    fig.update_layout(
        title="Decomposição da Receita — Contribuição por Canal",
        yaxis_title="Receita (R$)",
        font=dict(family="Segoe UI, Arial, sans-serif"),
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#E5E7EB", tickformat=",.0f"),
    )

    # ---- Matplotlib PNG ----
    fig_mpl, ax = plt.subplots(figsize=(12, 6))

    acumulado_mpl = 0.0
    x_pos = range(len(nomes) + 1)
    bottoms = []
    heights = []
    bar_colors = []
    tick_labels = list(nomes) + ["Receita Total"]

    for val, cor in zip(valores, cores):
        bottoms.append(acumulado_mpl)
        heights.append(val)
        bar_colors.append(cor)
        acumulado_mpl += val

    bottoms.append(0)
    heights.append(receita_total)
    bar_colors.append("#1E293B")

    ax.bar(x_pos, heights, bottom=bottoms, color=bar_colors,
           edgecolor="white", linewidth=0.5, width=0.6)

    for i, (h, b) in enumerate(zip(heights, bottoms)):
        ax.text(i, b + h + receita_total * 0.008, f"R$ {h:,.0f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Conectores entre barras de canal
    acum = 0.0
    for i in range(len(valores) - 1):
        acum += valores[i]
        ax.plot([i + 0.3, i + 0.7], [acum, acum],
                color="#94A3B8", linewidth=0.8, linestyle="--")

    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_ylabel("Receita (R$)")
    ax.set_title("Decomposição da Receita — Contribuição por Canal",
                 fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(_fmt_reais)
    _estilo_eixos(ax)

    fig_mpl.tight_layout()
    fig_mpl.savefig(IMAGES_DIR / "waterfall_decomposicao_receita.png")
    plt.close(fig_mpl)

    return fig


# ---------------------------------------------------------------------------
# 2. Curvas de Resposta
# ---------------------------------------------------------------------------

def plot_response_curves(model: MarketingMixModel) -> go.Figure:
    """
    Curvas de resposta de cada canal de marketing.

    Para cada canal, plota a relação entre investimento semanal (eixo X)
    e receita incremental prevista pelo modelo (eixo Y), considerando
    o regime estacionário de adstock + saturação de Hill. O ponto de
    investimento médio atual é destacado na curva.

    Parâmetros
    ----------
    model : MarketingMixModel
        Modelo treinado.

    Retorna
    -------
    go.Figure
        Figura Plotly interativa.
        PNG salva em ``docs/images/curvas_de_resposta.png``.
    """
    _garantir_diretorio()
    model._check_fitted()

    n_pts = 200

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[_label(c) for c in CHANNELS] + [""],
        horizontal_spacing=0.08, vertical_spacing=0.15,
    )
    fig_mpl, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    for idx, canal in enumerate(CHANNELS):
        row, col = idx // 3 + 1, idx % 3 + 1
        ax = axes_flat[idx]

        params = model.channel_params[canal]
        coef = float(model.params[f"spend_{canal}_transformed"])
        decay = params["decay"]
        half_sat = params["half_saturation"]
        slope = params["slope"]

        spend_atual = float(model._df_prepared[f"spend_{canal}"].mean())
        spend_max = float(model._df_prepared[f"spend_{canal}"].max()) * 3
        spends = np.linspace(0, spend_max, n_pts)

        # Receita incremental em regime estacionário
        receita = np.zeros(n_pts)
        for i, s in enumerate(spends):
            if s > 0:
                adstock_ss = s / (1.0 - decay)
                receita[i] = coef / (1.0 + (half_sat / adstock_ss) ** slope)

        # Ponto atual
        if spend_atual > 0:
            adstock_at = spend_atual / (1.0 - decay)
            receita_at = coef / (1.0 + (half_sat / adstock_at) ** slope)
        else:
            receita_at = 0.0

        cor = _cor(canal)

        # Plotly
        fig.add_trace(go.Scatter(
            x=spends, y=receita, mode="lines",
            line=dict(color=cor, width=2.5), showlegend=False,
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=[spend_atual], y=[receita_at], mode="markers",
            marker=dict(color=cor, size=10, line=dict(color="white", width=2)),
            showlegend=False,
            hovertext=f"Investimento atual: R$ {spend_atual:,.0f}",
        ), row=row, col=col)

        # Matplotlib
        ax.plot(spends, receita, color=cor, linewidth=2.5)
        ax.scatter([spend_atual], [receita_at], color=cor, s=80, zorder=5,
                   edgecolors="white", linewidth=2)
        ax.set_title(_label(canal), fontsize=12, fontweight="bold", color=cor)
        ax.set_xlabel("Investimento Semanal (R$)")
        ax.set_ylabel("Receita Incremental (R$)")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        _estilo_eixos(ax)

        ax.annotate(
            f"Atual\nR$ {spend_atual:,.0f}",
            xy=(spend_atual, receita_at),
            xytext=(spend_atual + spend_max * 0.08, receita_at * 0.82),
            fontsize=8, color=cor,
            arrowprops=dict(arrowstyle="->", color=cor, lw=1),
        )

    # Ocultar o 6º subplot (temos 5 canais)
    axes_flat[5].set_visible(False)
    fig.update_annotations(font=dict(family="Segoe UI, Arial, sans-serif"))

    fig.update_layout(
        title="Curvas de Resposta por Canal — Investimento vs. Receita Incremental",
        font=dict(family="Segoe UI, Arial, sans-serif"),
        plot_bgcolor="white", height=700, showlegend=False,
    )

    fig_mpl.suptitle(
        "Curvas de Resposta por Canal — Investimento vs. Receita Incremental",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig_mpl.tight_layout()
    fig_mpl.savefig(IMAGES_DIR / "curvas_de_resposta.png")
    plt.close(fig_mpl)

    return fig


# ---------------------------------------------------------------------------
# 3. Contribuição ao Longo do Tempo (Stacked Area)
# ---------------------------------------------------------------------------

def plot_channel_contribution_over_time(
    model: MarketingMixModel,
    df: pd.DataFrame,
) -> go.Figure:
    """
    Gráfico de área empilhada com a contribuição semanal de cada canal.

    Decompõe a receita prevista em baseline e contribuição de cada canal
    ao longo das semanas do histórico.

    Parâmetros
    ----------
    model : MarketingMixModel
        Modelo treinado.
    df : pd.DataFrame
        DataFrame original com coluna ``date``.

    Retorna
    -------
    go.Figure
        Figura Plotly interativa.
        PNG salva em ``docs/images/contribuicao_ao_longo_do_tempo.png``.
    """
    _garantir_diretorio()
    model._check_fitted()

    n = len(model._X)
    tem_data = "date" in df.columns
    datas = df["date"].values[:n] if tem_data else np.arange(n)

    # Contribuição semanal de cada canal
    contrib: dict[str, np.ndarray] = {}
    for canal in CHANNELS:
        col = f"spend_{canal}_transformed"
        contrib[canal] = float(model.params[col]) * model._X[col].to_numpy()

    # Baseline semanal (intercepto + variáveis de contexto)
    baseline = np.full(n, float(model.params["const"]))
    for ctx in CONTEXT_FEATURES:
        baseline += float(model.params[ctx]) * model._X[ctx].to_numpy()
    contrib["baseline_e_contexto"] = baseline

    ordem = ["baseline_e_contexto"] + list(CHANNELS)

    # ---- Plotly ----
    fig = go.Figure()
    for canal in ordem:
        fig.add_trace(go.Scatter(
            x=datas, y=contrib[canal],
            name=_label(canal), mode="lines", stackgroup="one",
            line=dict(width=0.5, color=_cor(canal)),
            fillcolor=_cor_clara(canal, 0.8),
        ))

    fig.update_layout(
        title="Contribuição de Cada Canal ao Longo do Tempo",
        xaxis_title="Semana", yaxis_title="Receita (R$)",
        font=dict(family="Segoe UI, Arial, sans-serif"),
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#E5E7EB", tickformat=",.0f"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2,
                    xanchor="center", x=0.5),
    )

    # ---- Matplotlib PNG ----
    fig_mpl, ax = plt.subplots(figsize=(14, 6))

    y_stack = np.array([contrib[c] for c in ordem])
    cores_stack = [_cor(c) for c in ordem]
    labels_stack = [_label(c) for c in ordem]

    ax.stackplot(range(n), y_stack, labels=labels_stack,
                 colors=cores_stack, alpha=0.85)

    if tem_data:
        _ticks_datas(ax, datas, n)

    ax.set_xlabel("Semana")
    ax.set_ylabel("Receita (R$)")
    ax.set_title("Contribuição de Cada Canal ao Longo do Tempo",
                 fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(_fmt_reais)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, frameon=False)
    _estilo_eixos(ax)

    fig_mpl.tight_layout()
    fig_mpl.savefig(IMAGES_DIR / "contribuicao_ao_longo_do_tempo.png")
    plt.close(fig_mpl)

    return fig


# ---------------------------------------------------------------------------
# 4. ROI por Canal
# ---------------------------------------------------------------------------

def plot_roi_by_channel(contributions: pd.DataFrame) -> go.Figure:
    """
    Barras horizontais com o ROI de cada canal, ordenado do maior ao menor.

    Parâmetros
    ----------
    contributions : pd.DataFrame
        DataFrame retornado por ``MarketingMixModel.get_channel_contributions()``.

    Retorna
    -------
    go.Figure
        Figura Plotly interativa.
        PNG salva em ``docs/images/roi_por_canal.png``.
    """
    _garantir_diretorio()

    df_roi = (
        contributions[contributions["canal"] != "baseline_e_contexto"]
        .dropna(subset=["roi"])
        .sort_values("roi", ascending=True)
    )

    canais = df_roi["canal"].tolist()
    rois = df_roi["roi"].tolist()
    cores = [_cor(c) for c in canais]
    labels = [_label(c) for c in canais]

    # ---- Plotly ----
    fig = go.Figure(go.Bar(
        x=rois, y=labels, orientation="h",
        marker_color=cores,
        text=[f"{r:.2f}x" for r in rois], textposition="outside",
    ))
    fig.update_layout(
        title="ROI por Canal de Marketing",
        xaxis_title="ROI (Retorno sobre Investimento)",
        font=dict(family="Segoe UI, Arial, sans-serif"),
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#E5E7EB"),
        showlegend=False, height=400,
    )

    # ---- Matplotlib PNG ----
    fig_mpl, ax = plt.subplots(figsize=(10, 5))

    y_pos = range(len(canais))
    bars = ax.barh(y_pos, rois, color=cores, edgecolor="white", height=0.6)

    for i, (bar, roi) in enumerate(zip(bars, rois)):
        ax.text(bar.get_width() + max(rois) * 0.02, i, f"{roi:.2f}x",
                va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.set_xlabel("ROI (Retorno sobre Investimento)")
    ax.set_title("ROI por Canal de Marketing", fontsize=14, fontweight="bold")
    _estilo_eixos(ax, grid_axis="x")

    fig_mpl.tight_layout()
    fig_mpl.savefig(IMAGES_DIR / "roi_por_canal.png")
    plt.close(fig_mpl)

    return fig


# ---------------------------------------------------------------------------
# 5. Comparação de Budget (Atual vs. Otimizado)
# ---------------------------------------------------------------------------

def plot_budget_optimization_comparison(
    comparison_df: pd.DataFrame,
) -> go.Figure:
    """
    Barras agrupadas comparando a alocação atual com a otimizada.

    Parâmetros
    ----------
    comparison_df : pd.DataFrame
        DataFrame retornado por ``BudgetOptimizer.compare()``, contendo
        as colunas ``canal``, ``spend_atual`` e ``spend_otimizado``.

    Retorna
    -------
    go.Figure
        Figura Plotly interativa.
        PNG salva em ``docs/images/comparacao_budget_otimizado.png``.
    """
    _garantir_diretorio()

    df_c = comparison_df[comparison_df["canal"] != "TOTAL"].copy()

    canais = df_c["canal"].tolist()
    spend_atual = df_c["spend_atual"].tolist()
    spend_otim = df_c["spend_otimizado"].tolist()
    cores = [_cor(c) for c in canais]
    labels = [_label(c) for c in canais]

    # ---- Plotly ----
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=spend_atual, name="Alocação Atual",
        marker_color=[_cor_clara(c) for c in canais],
        text=[f"R$ {v:,.0f}" for v in spend_atual], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=labels, y=spend_otim, name="Alocação Otimizada",
        marker_color=cores,
        text=[f"R$ {v:,.0f}" for v in spend_otim], textposition="outside",
    ))

    fig.update_layout(
        title="Comparação de Budget — Alocação Atual vs. Otimizada",
        yaxis_title="Investimento Mensal (R$)",
        font=dict(family="Segoe UI, Arial, sans-serif"),
        barmode="group", plot_bgcolor="white",
        yaxis=dict(gridcolor="#E5E7EB", tickformat=",.0f"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15,
                    xanchor="center", x=0.5),
    )

    # ---- Matplotlib PNG ----
    fig_mpl, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(canais))
    largura = 0.35

    ax.bar(x - largura / 2, spend_atual, largura, label="Alocação Atual",
           color=[_cor_clara_mpl(c) for c in canais], edgecolor="white")
    ax.bar(x + largura / 2, spend_otim, largura, label="Alocação Otimizada",
           color=cores, edgecolor="white")

    topo = max(max(spend_atual), max(spend_otim))
    for i in range(len(canais)):
        ax.text(i - largura / 2, spend_atual[i] + topo * 0.01,
                f"R$ {spend_atual[i]:,.0f}", ha="center", va="bottom",
                fontsize=7, rotation=45)
        ax.text(i + largura / 2, spend_otim[i] + topo * 0.01,
                f"R$ {spend_otim[i]:,.0f}", ha="center", va="bottom",
                fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Investimento Mensal (R$)")
    ax.set_title("Comparação de Budget — Alocação Atual vs. Otimizada",
                 fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(_fmt_reais)
    ax.legend(frameon=False)
    _estilo_eixos(ax)

    fig_mpl.tight_layout()
    fig_mpl.savefig(IMAGES_DIR / "comparacao_budget_otimizado.png")
    plt.close(fig_mpl)

    return fig


# ---------------------------------------------------------------------------
# 6. Efeito do Adstock
# ---------------------------------------------------------------------------

def plot_adstock_effect(
    df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    channel: str,
) -> go.Figure:
    """
    Compara o investimento original com o investimento após adstock.

    Mostra como o efeito de carryover suaviza e prolonga o impacto do
    investimento ao longo das semanas.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame original com colunas ``spend_<canal>`` e ``date``.
    transformed_df : pd.DataFrame
        DataFrame com transformações aplicadas (``apply_all_transformations``).
        Usado apenas para manter a assinatura consistente; o adstock é
        recalculado internamente para isolar o efeito puro.
    channel : str
        Nome do canal (ex.: ``'meta_ads'``).

    Retorna
    -------
    go.Figure
        Figura Plotly interativa.
        PNG salva em ``docs/images/efeito_adstock_{channel}.png``.
    """
    _garantir_diretorio()

    params = DEFAULT_CHANNEL_PARAMS[channel]
    original = df[f"spend_{channel}"].values
    adstocked = geometric_adstock(df[f"spend_{channel}"], params["decay"]).values
    n = len(original)
    tem_data = "date" in df.columns
    datas = df["date"].values if tem_data else np.arange(n)
    cor = _cor(channel)
    label_canal = _label(channel)

    # ---- Plotly ----
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=datas, y=original, name="Investimento Original",
        mode="lines", line=dict(color=cor, width=1.5, dash="dot"), opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=datas, y=adstocked, name="Com Adstock Aplicado",
        mode="lines", line=dict(color=cor, width=2.5),
    ))

    fig.update_layout(
        title=f"Efeito do Adstock — {label_canal} (decay = {params['decay']})",
        xaxis_title="Semana", yaxis_title="Investimento (R$)",
        font=dict(family="Segoe UI, Arial, sans-serif"),
        plot_bgcolor="white",
        yaxis=dict(gridcolor="#E5E7EB", tickformat=",.0f"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15,
                    xanchor="center", x=0.5),
    )

    # ---- Matplotlib PNG ----
    fig_mpl, ax = plt.subplots(figsize=(14, 5))

    x_range = range(n)
    ax.plot(x_range, original, color=cor, linewidth=1.5, linestyle=":",
            alpha=0.6, label="Investimento Original")
    ax.plot(x_range, adstocked, color=cor, linewidth=2.5,
            label="Com Adstock Aplicado")
    ax.fill_between(x_range, original, adstocked, color=cor, alpha=0.12)

    if tem_data:
        _ticks_datas(ax, datas, n)

    ax.set_xlabel("Semana")
    ax.set_ylabel("Investimento (R$)")
    ax.set_title(
        f"Efeito do Adstock — {label_canal} (decay = {params['decay']})",
        fontsize=14, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(_fmt_reais)
    ax.legend(frameon=False)
    _estilo_eixos(ax)

    fig_mpl.tight_layout()
    fig_mpl.savefig(IMAGES_DIR / f"efeito_adstock_{channel}.png")
    plt.close(fig_mpl)

    return fig


# ---------------------------------------------------------------------------
# 7. Diagnósticos do Modelo (Grid 2×2)
# ---------------------------------------------------------------------------

def plot_model_diagnostics(model: MarketingMixModel) -> go.Figure:
    """
    Grid 2×2 com diagnósticos visuais do modelo de regressão.

    Gráficos gerados:
        1. **Resíduos vs. Valores Preditos** — verifica homocedasticidade.
        2. **QQ-Plot dos Resíduos** — verifica normalidade.
        3. **Valores Reais vs. Preditos** — avalia aderência geral.
        4. **Histograma dos Resíduos** — distribuição dos erros.

    Parâmetros
    ----------
    model : MarketingMixModel
        Modelo treinado.

    Retorna
    -------
    go.Figure
        Figura Plotly interativa.
        PNG salva em ``docs/images/diagnosticos_modelo.png``.
    """
    _garantir_diretorio()
    model._check_fitted()

    y = model._y.to_numpy()
    y_hat = model.fitted_values
    residuos = model.residuals

    cor_pts = "#1877F2"
    cor_ref = "#EF4444"

    # Quantis para QQ-Plot
    residuos_pad = (residuos - residuos.mean()) / residuos.std()
    quantis_teo = stats.norm.ppf(
        (np.arange(1, len(residuos_pad) + 1) - 0.5) / len(residuos_pad)
    )
    residuos_ord = np.sort(residuos_pad)
    qq_min = min(float(quantis_teo.min()), float(residuos_ord.min()))
    qq_max = max(float(quantis_teo.max()), float(residuos_ord.max()))

    y_min = min(float(y.min()), float(y_hat.min()))
    y_max = max(float(y.max()), float(y_hat.max()))

    # ---- Plotly ----
    titulos_sub = [
        "Resíduos vs. Valores Preditos",
        "QQ-Plot dos Resíduos",
        "Valores Reais vs. Preditos",
        "Histograma dos Resíduos",
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=titulos_sub,
                        horizontal_spacing=0.1, vertical_spacing=0.12)

    # (1) Resíduos vs. Preditos
    fig.add_trace(go.Scatter(
        x=y_hat, y=residuos, mode="markers",
        marker=dict(color=cor_pts, size=5, opacity=0.6), showlegend=False,
    ), row=1, col=1)
    fig.add_hline(y=0, line=dict(color=cor_ref, dash="dash", width=1),
                  row=1, col=1)

    # (2) QQ-Plot
    fig.add_trace(go.Scatter(
        x=quantis_teo, y=residuos_ord, mode="markers",
        marker=dict(color=cor_pts, size=5, opacity=0.6), showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[qq_min, qq_max], y=[qq_min, qq_max], mode="lines",
        line=dict(color=cor_ref, dash="dash", width=1), showlegend=False,
    ), row=1, col=2)

    # (3) Reais vs. Preditos
    fig.add_trace(go.Scatter(
        x=y, y=y_hat, mode="markers",
        marker=dict(color=cor_pts, size=5, opacity=0.6), showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[y_min, y_max], y=[y_min, y_max], mode="lines",
        line=dict(color=cor_ref, dash="dash", width=1), showlegend=False,
    ), row=2, col=1)

    # (4) Histograma
    fig.add_trace(go.Histogram(
        x=residuos, nbinsx=25,
        marker_color=cor_pts, opacity=0.7, showlegend=False,
    ), row=2, col=2)

    fig.update_xaxes(title_text="Valores Preditos (R$)", row=1, col=1)
    fig.update_yaxes(title_text="Resíduos (R$)", row=1, col=1)
    fig.update_xaxes(title_text="Quantis Teóricos", row=1, col=2)
    fig.update_yaxes(title_text="Quantis Observados", row=1, col=2)
    fig.update_xaxes(title_text="Valores Reais (R$)", row=2, col=1)
    fig.update_yaxes(title_text="Valores Preditos (R$)", row=2, col=1)
    fig.update_xaxes(title_text="Resíduos (R$)", row=2, col=2)
    fig.update_yaxes(title_text="Frequência", row=2, col=2)

    fig.update_layout(
        title="Diagnósticos do Modelo de Regressão",
        font=dict(family="Segoe UI, Arial, sans-serif"),
        plot_bgcolor="white", height=700, showlegend=False,
    )

    # ---- Matplotlib PNG ----
    fig_mpl, axes = plt.subplots(2, 2, figsize=(12, 10))
    marker_kw = dict(color=cor_pts, alpha=0.6, s=20,
                     edgecolors="white", linewidth=0.3)

    # (1) Resíduos vs. Preditos
    ax = axes[0, 0]
    ax.scatter(y_hat, residuos, **marker_kw)
    ax.axhline(0, color=cor_ref, linestyle="--", linewidth=1)
    ax.set_xlabel("Valores Preditos (R$)")
    ax.set_ylabel("Resíduos (R$)")
    ax.set_title("Resíduos vs. Valores Preditos", fontweight="bold")
    ax.xaxis.set_major_formatter(_fmt_reais)
    ax.yaxis.set_major_formatter(_fmt_reais)
    _estilo_eixos(ax, grid_axis="both")

    # (2) QQ-Plot
    ax = axes[0, 1]
    ax.scatter(quantis_teo, residuos_ord, **marker_kw)
    ax.plot([qq_min, qq_max], [qq_min, qq_max],
            color=cor_ref, linestyle="--", linewidth=1)
    ax.set_xlabel("Quantis Teóricos")
    ax.set_ylabel("Quantis Observados")
    ax.set_title("QQ-Plot dos Resíduos", fontweight="bold")
    _estilo_eixos(ax, grid_axis="both")

    # (3) Reais vs. Preditos
    ax = axes[1, 0]
    ax.scatter(y, y_hat, **marker_kw)
    ax.plot([y_min, y_max], [y_min, y_max],
            color=cor_ref, linestyle="--", linewidth=1)
    ax.set_xlabel("Valores Reais (R$)")
    ax.set_ylabel("Valores Preditos (R$)")
    ax.set_title("Valores Reais vs. Preditos", fontweight="bold")
    ax.xaxis.set_major_formatter(_fmt_reais)
    ax.yaxis.set_major_formatter(_fmt_reais)
    _estilo_eixos(ax, grid_axis="both")

    # (4) Histograma
    ax = axes[1, 1]
    ax.hist(residuos, bins=25, color=cor_pts, alpha=0.7, edgecolor="white")
    ax.axvline(0, color=cor_ref, linestyle="--", linewidth=1)
    ax.set_xlabel("Resíduos (R$)")
    ax.set_ylabel("Frequência")
    ax.set_title("Histograma dos Resíduos", fontweight="bold")
    ax.xaxis.set_major_formatter(_fmt_reais)
    _estilo_eixos(ax, grid_axis="both")

    fig_mpl.suptitle("Diagnósticos do Modelo de Regressão",
                     fontsize=14, fontweight="bold", y=1.02)
    fig_mpl.tight_layout()
    fig_mpl.savefig(IMAGES_DIR / "diagnosticos_modelo.png")
    plt.close(fig_mpl)

    return fig
