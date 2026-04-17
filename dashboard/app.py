"""
Dashboard interativo de Marketing Mix Modeling (MMM).

Quatro páginas com visualizações, simulador de budget e diagnósticos
do modelo. Executar com:

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Setup de path — garante que src/ seja importável
# ---------------------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.model import CHANNELS, CONTEXT_FEATURES, MarketingMixModel
from src.optimizer import (
    BudgetOptimizer,
    DEFAULT_MAX_PER_CHANNEL,
    DEFAULT_MIN_PER_CHANNEL,
    WEEKS_PER_MONTH,
    current_allocation_from_history,
)
from src.transformations import (
    DEFAULT_CHANNEL_PARAMS,
    apply_all_transformations,
    geometric_adstock,
)

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

PLOTLY_LAYOUT = dict(
    font=dict(family="Segoe UI, Arial, sans-serif"),
    plot_bgcolor="white",
    margin=dict(l=40, r=20, t=50, b=40),
)


def _label(canal: str) -> str:
    return CHANNEL_LABELS.get(canal, canal)


def _cor(canal: str) -> str:
    return CHANNEL_COLORS.get(canal, "#333333")


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _cor_rgba(canal: str, alpha: float = 0.25) -> str:
    r, g, b = _hex_to_rgb(_cor(canal))
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Cache de dados e modelo
# ---------------------------------------------------------------------------

@st.cache_data
def carregar_dados() -> pd.DataFrame:
    """Carrega o CSV de dados de marketing."""
    caminho = os.path.join(_ROOT, "data", "raw", "marketing_data.csv")
    return pd.read_csv(caminho, parse_dates=["date"])


@st.cache_resource
def treinar_modelo() -> MarketingMixModel:
    """Treina o modelo MMM (cacheado para não retreinar a cada interação)."""
    df = carregar_dados()
    mmm = MarketingMixModel()
    mmm.fit(df)
    return mmm


def obter_modelo() -> tuple[pd.DataFrame, MarketingMixModel]:
    """Retorna dados e modelo treinado, usando cache do Streamlit."""
    df = carregar_dados()
    mmm = treinar_modelo()
    return df, mmm


# ---------------------------------------------------------------------------
# Configuração geral da página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Marketing Mix Modeling Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — Navegação
# ---------------------------------------------------------------------------

st.sidebar.title("📊 MMM Dashboard")
st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "Navegação",
    [
        "Visão Geral",
        "Análise por Canal",
        "Simulador de Budget",
        "Diagnósticos do Modelo",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption("Marketing Mix Modeling — Projeto de Portfólio")


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — Visão Geral
# ═══════════════════════════════════════════════════════════════════════════

def pagina_visao_geral() -> None:
    """Renderiza a página de visão geral com KPIs, waterfall e stacked area."""
    st.title("📊 Marketing Mix Modeling Dashboard")
    st.markdown("### Visão Geral — Decomposição da Receita")

    df, mmm = obter_modelo()
    contributions = mmm.get_channel_contributions()

    # ------ KPIs no topo ------
    receita_total = float(df["revenue"].sum())
    canais_df = contributions[contributions["canal"] != "baseline_e_contexto"]
    receita_mkt = float(canais_df["contribuicao_total"].sum())
    roi_medio = float(canais_df["roi"].dropna().mean())
    melhor_canal_row = canais_df.loc[canais_df["roi"].idxmax()]
    melhor_canal = _label(melhor_canal_row["canal"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Receita Total", f"R$ {receita_total:,.0f}")
    col2.metric("Receita de Marketing", f"R$ {receita_mkt:,.0f}")
    col3.metric("ROI Médio", f"{roi_medio:.2f}x")
    col4.metric("Melhor Canal (ROI)", melhor_canal)

    st.markdown("---")

    # ------ Waterfall ------
    st.subheader("Decomposição da Receita — Waterfall")

    mask_base = contributions["canal"] == "baseline_e_contexto"
    baseline_val = float(contributions.loc[mask_base, "contribuicao_total"].iloc[0])
    canais_sorted = canais_df.sort_values("contribuicao_total", ascending=False)

    nomes = ["Baseline"]
    valores = [baseline_val]
    cores = [_cor("baseline_e_contexto")]
    for _, row in canais_sorted.iterrows():
        nomes.append(_label(row["canal"]))
        valores.append(float(row["contribuicao_total"]))
        cores.append(_cor(row["canal"]))

    total = sum(valores)

    fig_wf = go.Figure()
    acum = 0.0
    for nome, val, cor in zip(nomes, valores, cores):
        fig_wf.add_trace(go.Bar(
            x=[nome], y=[val], base=[acum],
            marker_color=cor, text=f"R$ {val:,.0f}",
            textposition="outside", showlegend=False, width=0.55,
        ))
        acum += val

    fig_wf.add_trace(go.Bar(
        x=["Receita Total"], y=[total], base=[0],
        marker_color="#1E293B", text=f"R$ {total:,.0f}",
        textposition="outside", showlegend=False, width=0.55,
    ))

    acum = 0.0
    for i in range(len(valores) - 1):
        acum += valores[i]
        fig_wf.add_shape(
            type="line", x0=i + 0.3, x1=i + 0.7, y0=acum, y1=acum,
            line=dict(color="#94A3B8", width=1, dash="dot"),
        )

    fig_wf.update_layout(
        yaxis_title="Receita (R$)",
        yaxis=dict(gridcolor="#E5E7EB", tickformat=",.0f"),
        height=450, **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # ------ Stacked area ------
    st.subheader("Contribuição de Cada Canal ao Longo do Tempo")

    n = len(mmm._X)
    datas = df["date"].values[:n]

    contrib_dict: dict[str, np.ndarray] = {}
    for canal in CHANNELS:
        col_t = f"spend_{canal}_transformed"
        contrib_dict[canal] = float(mmm.params[col_t]) * mmm._X[col_t].to_numpy()

    baseline_ts = np.full(n, float(mmm.params["const"]))
    for ctx in CONTEXT_FEATURES:
        baseline_ts += float(mmm.params[ctx]) * mmm._X[ctx].to_numpy()
    contrib_dict["baseline_e_contexto"] = baseline_ts

    ordem = ["baseline_e_contexto"] + list(CHANNELS)

    fig_area = go.Figure()
    for canal in ordem:
        fig_area.add_trace(go.Scatter(
            x=datas, y=contrib_dict[canal],
            name=_label(canal), mode="lines", stackgroup="one",
            line=dict(width=0.5, color=_cor(canal)),
            fillcolor=_cor_rgba(canal, 0.8),
        ))

    fig_area.update_layout(
        xaxis_title="Semana", yaxis_title="Receita (R$)",
        yaxis=dict(gridcolor="#E5E7EB", tickformat=",.0f"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2,
                    xanchor="center", x=0.5),
        height=450, **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_area, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — Análise por Canal
# ═══════════════════════════════════════════════════════════════════════════

def pagina_analise_canal() -> None:
    """Renderiza a página de análise detalhada por canal."""
    st.title("📊 Marketing Mix Modeling Dashboard")
    st.markdown("### Análise por Canal")

    df, mmm = obter_modelo()
    contributions = mmm.get_channel_contributions()

    # Dropdown
    opcoes = {_label(c): c for c in CHANNELS}
    canal_display = st.selectbox("Selecione o canal:", list(opcoes.keys()))
    canal = opcoes[canal_display]
    cor = _cor(canal)

    st.markdown("---")

    # ------ Métricas do canal ------
    row_canal = contributions[contributions["canal"] == canal].iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Contribuição Total", f"R$ {row_canal['contribuicao_total']:,.0f}")
    col2.metric("% da Receita", f"{row_canal['contribuicao_pct']:.1f}%")
    col3.metric("Investimento Total", f"R$ {row_canal['investimento_total']:,.0f}")
    roi_val = row_canal["roi"]
    col4.metric("ROI", f"{roi_val:.2f}x" if pd.notna(roi_val) else "—")

    st.markdown("---")

    # ------ Curva de resposta ------
    col_esq, col_dir = st.columns(2)

    with col_esq:
        st.subheader("Curva de Resposta")

        params = mmm.channel_params[canal]
        coef = float(mmm.params[f"spend_{canal}_transformed"])
        decay = params["decay"]
        half_sat = params["half_saturation"]
        slope = params["slope"]

        spend_atual = float(mmm._df_prepared[f"spend_{canal}"].mean())
        spend_max = float(mmm._df_prepared[f"spend_{canal}"].max()) * 3
        spends = np.linspace(0, spend_max, 200)

        receita = np.zeros(200)
        for i, s in enumerate(spends):
            if s > 0:
                adstock_ss = s / (1.0 - decay)
                receita[i] = coef / (1.0 + (half_sat / adstock_ss) ** slope)

        if spend_atual > 0:
            adstock_at = spend_atual / (1.0 - decay)
            receita_at = coef / (1.0 + (half_sat / adstock_at) ** slope)
        else:
            receita_at = 0.0

        fig_rc = go.Figure()
        fig_rc.add_trace(go.Scatter(
            x=spends, y=receita, mode="lines",
            line=dict(color=cor, width=3), name="Curva de Resposta",
        ))
        fig_rc.add_trace(go.Scatter(
            x=[spend_atual], y=[receita_at], mode="markers+text",
            marker=dict(color=cor, size=12, line=dict(color="white", width=2)),
            text=[f"Atual: R$ {spend_atual:,.0f}"], textposition="top center",
            name="Investimento Atual",
        ))
        fig_rc.update_layout(
            xaxis_title="Investimento Semanal (R$)",
            yaxis_title="Receita Incremental (R$)",
            showlegend=False, height=400, **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_rc, use_container_width=True)

    # ------ Efeito do adstock ------
    with col_dir:
        st.subheader(f"Efeito do Adstock (decay = {decay})")

        original = df[f"spend_{canal}"].values
        adstocked = geometric_adstock(
            df[f"spend_{canal}"], decay
        ).values
        datas = df["date"].values

        fig_ads = go.Figure()
        fig_ads.add_trace(go.Scatter(
            x=datas, y=original, name="Investimento Original",
            mode="lines", line=dict(color=cor, width=1.5, dash="dot"),
            opacity=0.6,
        ))
        fig_ads.add_trace(go.Scatter(
            x=datas, y=adstocked, name="Com Adstock",
            mode="lines", line=dict(color=cor, width=2.5),
        ))
        fig_ads.update_layout(
            xaxis_title="Semana", yaxis_title="Investimento (R$)",
            yaxis=dict(gridcolor="#E5E7EB", tickformat=",.0f"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                        xanchor="center", x=0.5),
            height=400, **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_ads, use_container_width=True)

    # ------ Parâmetros do canal ------
    st.subheader("Parâmetros do Canal")
    params_col1, params_col2, params_col3 = st.columns(3)
    params_col1.metric("Decay (Adstock)", f"{decay:.2f}")
    params_col2.metric("Half-Saturation", f"R$ {half_sat:,.0f}")
    params_col3.metric("Slope (Hill)", f"{slope:.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — Simulador de Budget
# ═══════════════════════════════════════════════════════════════════════════

def pagina_simulador() -> None:
    """Renderiza o simulador de budget com sliders e otimização."""
    st.title("📊 Marketing Mix Modeling Dashboard")
    st.markdown("### Simulador de Budget")

    df, mmm = obter_modelo()
    opt = BudgetOptimizer(mmm)

    alocacao_atual = current_allocation_from_history(df, reference_weeks=4)
    budget_atual = sum(alocacao_atual.values())

    # ------ Slider de budget total ------
    budget_total = st.slider(
        "Budget Total Mensal (R$)",
        min_value=100_000, max_value=500_000,
        value=int(budget_atual), step=5_000,
        format="R$ %d",
    )

    st.markdown("---")

    # ------ Estado dos sliders de canais ------
    if "sliders_canal" not in st.session_state:
        st.session_state.sliders_canal = {
            canal: int(alocacao_atual.get(canal, 0))
            for canal in CHANNELS
        }

    # ------ Botão de otimização ------
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        otimizar = st.button("⚡ Otimizar Automaticamente", type="primary")

    if otimizar:
        try:
            alocacao_otima = opt.optimize(total_budget=float(budget_total))
            st.session_state.sliders_canal = {
                canal: int(round(val))
                for canal, val in alocacao_otima.items()
            }
            with col_info:
                st.success("Budget otimizado com sucesso!")
        except (ValueError, RuntimeError) as e:
            with col_info:
                st.error(f"Erro na otimização: {e}")

    # ------ Sliders por canal ------
    st.subheader("Alocação por Canal")
    valores_canal: dict[str, float] = {}

    cols = st.columns(len(CHANNELS))
    for i, canal in enumerate(CHANNELS):
        with cols[i]:
            piso = int(DEFAULT_MIN_PER_CHANNEL[canal])
            teto = min(int(DEFAULT_MAX_PER_CHANNEL[canal]), budget_total)
            default = min(max(st.session_state.sliders_canal[canal], piso), teto)

            val = st.slider(
                _label(canal),
                min_value=piso, max_value=teto,
                value=default, step=1_000,
                format="R$ %d",
                key=f"slider_{canal}",
            )
            valores_canal[canal] = float(val)

    soma_canais = sum(valores_canal.values())
    diferenca = budget_total - soma_canais

    # Indicador de soma
    if abs(diferenca) < 100:
        st.info(f"Soma dos canais: **R$ {soma_canais:,.0f}** = Budget total")
    elif diferenca > 0:
        st.warning(
            f"Soma dos canais: **R$ {soma_canais:,.0f}** — "
            f"Faltam **R$ {diferenca:,.0f}** para atingir o budget total."
        )
    else:
        st.error(
            f"Soma dos canais: **R$ {soma_canais:,.0f}** — "
            f"Excede o budget total em **R$ {-diferenca:,.0f}**."
        )

    st.markdown("---")

    # ------ Receita estimada ------
    spends_array = np.array(
        [valores_canal[c] for c in CHANNELS], dtype=float
    )
    receita_estimada = opt._predict_monthly_revenue(spends_array)

    # Receita atual para comparação
    spends_atual_arr = np.array(
        [alocacao_atual.get(c, 0) for c in CHANNELS], dtype=float
    )
    receita_atual = opt._predict_monthly_revenue(spends_atual_arr)
    delta_receita = receita_estimada - receita_atual

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Receita Mensal Estimada", f"R$ {receita_estimada:,.0f}")
    col_r2.metric(
        "vs. Alocação Atual",
        f"R$ {delta_receita:+,.0f}",
        delta=f"{delta_receita / receita_atual * 100:+.1f}%" if receita_atual > 0 else "—",
    )
    col_r3.metric("Budget Utilizado", f"R$ {soma_canais:,.0f}")

    # ------ Gráfico comparativo ------
    st.subheader("Alocação Atual vs. Simulação")

    fig_comp = go.Figure()
    labels = [_label(c) for c in CHANNELS]
    cores = [_cor(c) for c in CHANNELS]
    spend_at_list = [alocacao_atual.get(c, 0) for c in CHANNELS]
    spend_sim_list = [valores_canal[c] for c in CHANNELS]

    fig_comp.add_trace(go.Bar(
        x=labels, y=spend_at_list, name="Alocação Atual",
        marker_color=[_cor_rgba(c, 0.35) for c in CHANNELS],
    ))
    fig_comp.add_trace(go.Bar(
        x=labels, y=spend_sim_list, name="Simulação",
        marker_color=cores,
    ))
    fig_comp.update_layout(
        barmode="group",
        yaxis_title="Investimento Mensal (R$)",
        yaxis=dict(gridcolor="#E5E7EB", tickformat=",.0f"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15,
                    xanchor="center", x=0.5),
        height=420, **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_comp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — Diagnósticos do Modelo
# ═══════════════════════════════════════════════════════════════════════════

def pagina_diagnosticos() -> None:
    """Renderiza a página de diagnósticos do modelo."""
    st.title("📊 Marketing Mix Modeling Dashboard")
    st.markdown("### Diagnósticos do Modelo")

    df, mmm = obter_modelo()
    diag = mmm.get_model_diagnostics()

    # ------ Métricas ------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{diag['r2']:.4f}")
    col2.metric("MAPE", f"{diag['mape']:.2f}%")
    col3.metric("MAE", f"R$ {diag['mae']:,.0f}")
    col4.metric("Durbin-Watson", f"{diag['durbin_watson']:.3f}")

    st.markdown("---")

    # ------ Informações extras ------
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("R² Ajustado", f"{diag['adj_r2']:.4f}")
    info_col2.metric("RMSE", f"R$ {diag['rmse']:,.0f}")
    info_col3.metric(
        "Método de Ajuste",
        "OLS com Bounds" if diag["is_constrained"] else "OLS (Statsmodels)",
    )

    st.markdown("---")

    # ------ Grid 2x2 de diagnósticos ------
    st.subheader("Gráficos de Diagnóstico")

    y = mmm._y.to_numpy()
    y_hat = mmm.fitted_values
    residuos = mmm.residuals

    cor_pts = "#1877F2"
    cor_ref = "#EF4444"

    from scipy import stats as sp_stats

    residuos_pad = (residuos - residuos.mean()) / residuos.std()
    quantis_teo = sp_stats.norm.ppf(
        (np.arange(1, len(residuos_pad) + 1) - 0.5) / len(residuos_pad)
    )
    residuos_ord = np.sort(residuos_pad)
    qq_min = min(float(quantis_teo.min()), float(residuos_ord.min()))
    qq_max = max(float(quantis_teo.max()), float(residuos_ord.max()))
    y_min = min(float(y.min()), float(y_hat.min()))
    y_max = max(float(y.max()), float(y_hat.max()))

    fig_diag = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Resíduos vs. Valores Preditos",
            "QQ-Plot dos Resíduos",
            "Valores Reais vs. Preditos",
            "Histograma dos Resíduos",
        ],
        horizontal_spacing=0.1, vertical_spacing=0.12,
    )

    marker_kw = dict(color=cor_pts, size=5, opacity=0.6)

    # (1) Resíduos vs Preditos
    fig_diag.add_trace(go.Scatter(
        x=y_hat, y=residuos, mode="markers",
        marker=marker_kw, showlegend=False,
    ), row=1, col=1)
    fig_diag.add_hline(
        y=0, line=dict(color=cor_ref, dash="dash", width=1), row=1, col=1
    )

    # (2) QQ-Plot
    fig_diag.add_trace(go.Scatter(
        x=quantis_teo, y=residuos_ord, mode="markers",
        marker=marker_kw, showlegend=False,
    ), row=1, col=2)
    fig_diag.add_trace(go.Scatter(
        x=[qq_min, qq_max], y=[qq_min, qq_max], mode="lines",
        line=dict(color=cor_ref, dash="dash", width=1), showlegend=False,
    ), row=1, col=2)

    # (3) Reais vs Preditos
    fig_diag.add_trace(go.Scatter(
        x=y, y=y_hat, mode="markers",
        marker=marker_kw, showlegend=False,
    ), row=2, col=1)
    fig_diag.add_trace(go.Scatter(
        x=[y_min, y_max], y=[y_min, y_max], mode="lines",
        line=dict(color=cor_ref, dash="dash", width=1), showlegend=False,
    ), row=2, col=1)

    # (4) Histograma
    fig_diag.add_trace(go.Histogram(
        x=residuos, nbinsx=25,
        marker_color=cor_pts, opacity=0.7, showlegend=False,
    ), row=2, col=2)

    fig_diag.update_xaxes(title_text="Valores Preditos (R$)", row=1, col=1)
    fig_diag.update_yaxes(title_text="Resíduos (R$)", row=1, col=1)
    fig_diag.update_xaxes(title_text="Quantis Teóricos", row=1, col=2)
    fig_diag.update_yaxes(title_text="Quantis Observados", row=1, col=2)
    fig_diag.update_xaxes(title_text="Valores Reais (R$)", row=2, col=1)
    fig_diag.update_yaxes(title_text="Valores Preditos (R$)", row=2, col=1)
    fig_diag.update_xaxes(title_text="Resíduos (R$)", row=2, col=2)
    fig_diag.update_yaxes(title_text="Frequência", row=2, col=2)

    fig_diag.update_layout(
        height=700, showlegend=False, **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_diag, use_container_width=True)

    # ------ VIF ------
    st.subheader("Multicolinearidade (VIF)")

    vif_data = []
    for feature, vif_val in diag["vif"].items():
        nome = feature.replace("spend_", "").replace("_transformed", "")
        nome_display = CHANNEL_LABELS.get(nome, nome)
        status = "Alto" if vif_val > 10 else "OK"
        vif_data.append({
            "Feature": nome_display,
            "VIF": round(vif_val, 2),
            "Status": status,
        })

    vif_df = pd.DataFrame(vif_data)
    st.dataframe(vif_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Roteamento de páginas
# ---------------------------------------------------------------------------

PAGINAS = {
    "Visão Geral": pagina_visao_geral,
    "Análise por Canal": pagina_analise_canal,
    "Simulador de Budget": pagina_simulador,
    "Diagnósticos do Modelo": pagina_diagnosticos,
}

PAGINAS[pagina]()
