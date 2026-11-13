"""Dashboard do case de fraude (PT/EN): visão de jan/2026, fila priorizada com tratativa, simulador e drift."""
import json
from pathlib import Path

import altair as alt
import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Radar de Fraude · Fraud Radar", page_icon=":material/shield:", layout="wide")

BASE = Path(__file__).parent
# paleta validada (daltonismo + contraste): azul = padrão/revisar, magenta = risco/bloquear, violeta = 3ª série
AZUL, MAGENTA, VIOLETA = "#3346d3", "#e0005f", "#8a3fc7"
CINZA, TINTA, TINTA2, GRADE = "#c4c6d4", "#11163a", "#5a5f7a", "#e8e9f0"
GRAD = "linear-gradient(90deg, #2b39c4 0%, #8a3fc7 55%, #e0005f 100%)"
GRAD_CAPA = "linear-gradient(120deg, #08137a 0%, #1b1a8f 25%, #521a8c 58%, #a10a74 82%, #e0005f 100%)"
ACOES = {"bloquear": MAGENTA, "revisar": AZUL, "aprovar": CINZA}
FONTE = "IBM Plex Sans, Segoe UI, sans-serif"

# ---------------------------------------------------------------------------- idioma
if "lang" not in st.session_state:
    st.session_state.lang = "PT"


def EN():
    return st.session_state.get("lang") == "EN"


def T(pt, en):
    """Texto no idioma escolhido no botão PT/EN."""
    return en if EN() else pt


ACAO_EN = {"bloquear": "block", "revisar": "review", "aprovar": "approve"}
RESULTADOS = ["Fraude confirmada", "Não é fraude", "Inconclusivo"]  # valores gravados (sempre em PT)
RES_EN = {"Fraude confirmada": "Confirmed fraud", "Não é fraude": "Not fraud", "Inconclusivo": "Inconclusive", "Pendente": "Pending"}
FAIXA_EN = {"estável": "stable", "atenção": "attention", "mudança": "shift"}
REG_EN = {"Sudeste": "Southeast", "Sul": "South", "Nordeste": "Northeast"}


def acao_txt(a):
    return ACAO_EN.get(a, a) if EN() else a


def res_txt(r):
    return RES_EN.get(r, r) if EN() else r


def nomes():
    return {"hora": T("Horário", "Hour"), "is_new_device": T("Dispositivo novo", "New device"),
            "nome_estabelecimento": T("Estabelecimento", "Merchant"), "cidade": T("Cidade", "City"),
            "turno": T("Turno", "Shift"), "cidade_regiao": T("Região", "Region"), "valor_transacao": T("Valor", "Amount"),
            "device_type": T("Tipo de dispositivo", "Device type"), "valor_log": T("Valor (log)", "Amount (log)")}


st.html(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
.hero {{ background: {GRAD_CAPA}; color: #fff; padding: 34px 40px 30px; margin-bottom: 6px; position: relative; overflow: hidden;
         border-radius: 18px; box-shadow: 0 18px 40px rgba(20,24,80,.22) }}
.hero::after {{ content: ''; position: absolute; width: 520px; height: 520px; right: -140px; top: -200px; border-radius: 50%;
               background: radial-gradient(circle at 35% 35%, rgba(255,255,255,.22), rgba(255,255,255,0) 60%);
               border: 1px solid rgba(255,255,255,.18) }}
.hero .k {{ font-size: 12px; font-weight: 600; letter-spacing: .16em; text-transform: uppercase; color: #ffb3d1 }}
.hero h1 {{ font-family: {FONTE}; font-weight: 300; font-size: 44px; line-height: 1.1; margin: 8px 0 0; color: #fff; padding: 0 }}
.hero .r {{ height: 2px; width: 40%; background: rgba(255,255,255,.85); margin: 20px 0 16px -40px }}
.hero p {{ color: rgba(255,255,255,.85); font-size: 15px; margin: 0 }}
.hero .nums {{ display: flex; flex-wrap: wrap; gap: 0; margin-top: 26px }}
.hero .nums div {{ padding: 0 34px; border-left: 1px solid rgba(255,255,255,.3) }}
.hero .nums div:first-child {{ padding-left: 0; border-left: 0 }}
.hero .nums b {{ display: block; font-size: 38px; font-weight: 500; line-height: 1.1 }}
.hero .nums span {{ font-size: 13px; color: rgba(255,255,255,.8) }}
.cab .k {{ font-size: 12px; font-weight: 600; letter-spacing: .16em; text-transform: uppercase; color: {MAGENTA} }}
.cab h1 {{ font-family: {FONTE}; font-weight: 300; font-size: 36px; margin: 4px 0 0; padding: 0; color: {TINTA} }}
.cab .r {{ height: 3px; width: 220px; background: {GRAD}; margin: 14px 0 12px }}
.cab p {{ color: {TINTA2}; font-size: 14.5px; margin: 0 0 8px }}
.tc b {{ font-size: 15px; font-weight: 500; color: {TINTA} }} .tc span {{ display: block; font-size: 12.5px; color: {TINTA2} }}
.big {{ font-size: 64px; font-weight: 500; line-height: 1; background: {GRAD}; -webkit-background-clip: text;
        background-clip: text; -webkit-text-fill-color: transparent; margin: 6px 0 4px }}
.selo {{ display: inline-block; color: #fff; font-size: 12px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase;
         padding: 5px 12px; border-radius: 3px }}
.gauge {{ position: relative; height: 12px; border-radius: 6px; margin: 30px 0 8px; display: flex; overflow: visible }}
.gauge i {{ display: block; height: 100% }}
.gauge .m {{ position: absolute; top: -9px; width: 3px; height: 30px; background: {TINTA}; border-radius: 2px }}
.gauge .m::after {{ content: attr(data-v); position: absolute; top: -22px; left: 50%; transform: translateX(-50%);
                   font-size: 12px; font-weight: 600; color: {TINTA}; white-space: nowrap }}
.escala {{ display: flex; font-size: 11.5px; color: {TINTA2} }}
.mini {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 6px 0 2px }}
.mini div {{ background: #f6f7fb; border-radius: 12px; padding: 10px 14px; border-left: 3px solid {VIOLETA} }}
.mini span {{ display: block; font-size: 12px; color: {TINTA2} }} .mini b {{ font-size: 22px; font-weight: 500; color: {TINTA} }}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{
  background: #fff; border-radius: 16px !important; border-color: #eceef4 !important; box-shadow: 0 6px 24px rgba(17,22,58,.06) }}
[data-testid="stMetric"] {{ background: #fff; border-radius: 14px !important; box-shadow: 0 6px 24px rgba(17,22,58,.06);
  border-color: #eceef4 !important; position: relative; overflow: hidden; padding-left: 16px !important }}
[data-testid="stMetric"]::before {{ content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: linear-gradient(#3346d3, #e0005f) }}
[data-testid="stMetricValue"] {{ font-weight: 500 }}
</style>""")


@st.cache_data
def carregar_dados():
    df = pd.read_csv(BASE / "dados" / "jan2026_classificadas.csv", parse_dates=["datahora_transacao"])
    if "motivo_alerta_en" not in df:
        df["motivo_alerta_en"] = df.motivo_alerta
    drift = pd.read_csv(BASE / "dados" / "drift.csv")
    meta = json.loads((BASE / "meta.json").read_text(encoding="utf-8"))
    return df, drift, meta


@st.cache_resource
def carregar_modelo():
    return lgb.Booster(model_file=str(BASE / "modelo_lgbm.txt"))


df, drift, meta = carregar_dados()
modelo = carregar_modelo()
T_REV, T_BLOQ = meta["t_rev"], meta["t_bloq"]

REGIAO = {"Belo Horizonte": "Sudeste", "Rio de Janeiro": "Sudeste", "São Paulo": "Sudeste", "Curitiba": "Sul",
          "Porto Alegre": "Sul", "Recife": "Nordeste", "Salvador": "Nordeste"}
DERIVADAS = {"valor_transacao", "valor_log", "hora", "is_new_device", "turno", "cidade_regiao"}


def n_br(v, d=0):
    s = f"{v:,.{d}f}"
    return s if EN() else s.replace(",", "X").replace(".", ",").replace("X", ".")


def brl2(v):
    return "R$ " + n_br(v, 2)


def pct_br(v, d=1):
    s = f"{v:.{d}%}"
    return s if EN() else s.replace(".", ",")


def turno_da_hora(h):
    return "madrugada" if h < 6 else "manha" if h < 12 else "tarde" if h < 18 else "noite"


def acao(p):
    return "bloquear" if p >= T_BLOQ else ("revisar" if p >= T_REV else "aprovar")


def estilo(chart, altura):
    """Aplica a identidade visual a um gráfico Altair."""
    return (chart.properties(height=altura, background="transparent")
            .configure(font=FONTE)
            .configure_view(stroke=None)
            .configure_axis(labelColor=TINTA2, titleColor=TINTA2, labelFontSize=11, titleFontSize=11, titleFontWeight=400,
                            gridColor=GRADE, domainColor=CINZA, tickColor=CINZA, tickSize=0, labelPadding=6)
            .configure_legend(labelColor=TINTA2, titleColor=TINTA2, labelFontSize=11, symbolType="square",
                              orient="top", direction="horizontal", title=None))


def grafico(chart, altura=260):
    st.altair_chart(estilo(chart, altura), width="stretch", theme=None)


def cabecalho(kicker, titulo, texto=""):
    st.html(f"<div class='cab'><div class='k'>{kicker}</div><h1>{titulo}</h1><div class='r'></div>"
            f"{f'<p>{texto}</p>' if texto else ''}</div>")


def titulo_card(t, sub=""):
    st.html(f"<div class='tc'><b>{t}</b>{f'<span>{sub}</span>' if sub else ''}</div>")


# ---------------------------------------------------------------------------
def pagina_visao():
    alerta = df[df.acao_recomendada != "aprovar"].assign(acao=lambda d: d.acao_recomendada.map(acao_txt))
    n_rev, n_bloq = (df.acao_recomendada == "revisar").sum(), (df.acao_recomendada == "bloquear").sum()
    m = meta["metricas_teste"]
    nums = [(n_br(len(df)), T("transações em jan/2026", "transactions in Jan/2026")),
            (n_br(n_rev), T("para revisar · ", "to review · ") + pct_br(n_rev / len(df))),
            (n_br(n_bloq), T("bloqueio automático", "automatic block")),
            ("R$ " + n_br((df.prob_fraude * df.valor_transacao).sum()), T("fraude esperada", "expected fraud")),
            (n_br(m["pr_auc"], 2), T("PR-AUC · score atual ", "PR-AUC · current score ") + n_br(m["auc_score_interno"], 2) + " AUC")]
    st.html(f"""<div class="hero"><div class="k">{T("Radar de Fraude · Janeiro/2026", "Fraud Radar · January/2026")}</div>
      <h1>{T("Quais transações merecem atenção agora", "Which transactions need attention now")}</h1><div class="r"></div>
      <p>{T(f"Modelo LightGBM treinado em 2025 · revisar se a probabilidade for ≥ {T_REV:.0%} · bloquear se for ≥ {T_BLOQ:.0%}",
            f"LightGBM model trained on 2025 · review if probability ≥ {T_REV:.0%} · block if ≥ {T_BLOQ:.0%}")}</p>
      <div class="nums">{''.join(f'<div><b>{v}</b><span>{r}</span></div>' for v, r in nums)}</div></div>""")

    c1, c2 = st.columns([1.35, 1])
    with c1, st.container(border=True):
        titulo_card(T("Alertas por hora do dia", "Alerts by hour of day"),
                    T("a fraude se concentra à noite: o pico é às 22h–23h", "fraud concentrates at night: the peak is at 10–11 pm"))
        por_hora = alerta.groupby(["hora", "acao"]).size().reset_index(name="n")
        grafico(alt.Chart(por_hora).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, stroke="white", strokeWidth=1).encode(
            x=alt.X("hora:O", title=None, axis=alt.Axis(labelExpr="datum.value + 'h'", labelAngle=0)),
            y=alt.Y("n:Q", title=T("transações", "transactions")),
            color=alt.Color("acao:N", title=None,
                            scale=alt.Scale(domain=[acao_txt("revisar"), acao_txt("bloquear")], range=[AZUL, MAGENTA])),
            order=alt.Order("acao:N", sort="descending"),
            tooltip=[alt.Tooltip("hora:O", title=T("hora", "hour")), alt.Tooltip("acao:N", title=T("ação", "action")),
                     alt.Tooltip("n:Q", title=T("transações", "transactions"), format=",")]), 270)
    with c2, st.container(border=True):
        titulo_card(T("Alertas por estabelecimento", "Alerts by merchant"),
                    T("volume de transações sinalizadas", "number of flagged transactions"))
        por_estab = alerta.groupby("nome_estabelecimento").size().reset_index(name="n")
        base = alt.Chart(por_estab).encode(y=alt.Y("nome_estabelecimento:N", sort="-x", title=None))
        grafico(base.mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3, height=16).encode(
            x=alt.X("n:Q", title=None, axis=alt.Axis(labels=False, grid=False)),
            color=alt.Color("n:Q", scale=alt.Scale(range=["#b9c0f2", AZUL]), legend=None),
            tooltip=[alt.Tooltip("nome_estabelecimento:N", title=T("estabelecimento", "merchant")),
                     alt.Tooltip("n:Q", title=T("alertas", "alerts"))])
            + base.mark_text(align="left", dx=5, color=TINTA2, fontSize=11).encode(x="n:Q", text=alt.Text("n:Q", format=",")),
            270)

    with st.container(border=True):
        titulo_card(T("Onde e quando: taxa de alerta por estabelecimento × hora", "Where and when: alert rate by merchant × hour"),
                    T("cada estabelecimento tem a sua janela de risco · quanto mais escuro, maior a % de transações em alerta",
                      "each merchant has its own risk window · the darker, the higher the % of transactions flagged"))
        mapa = (df.assign(alerta=df.acao_recomendada != "aprovar")
                .groupby(["nome_estabelecimento", "hora"]).alerta.mean().reset_index())
        ordem = mapa.groupby("nome_estabelecimento").alerta.mean().sort_values(ascending=False).index.tolist()
        grafico(alt.Chart(mapa).mark_rect(cornerRadius=2, stroke="white", strokeWidth=2).encode(
            x=alt.X("hora:O", title=None, axis=alt.Axis(labelExpr="datum.value + 'h'", labelAngle=0, domain=False)),
            y=alt.Y("nome_estabelecimento:N", sort=ordem, title=None, axis=alt.Axis(domain=False)),
            color=alt.Color("alerta:Q", title=T("% em alerta", "% flagged"),
                            scale=alt.Scale(range=["#f3f4f8", "#f9d3e2", "#ef6a9c", MAGENTA, "#7a0034"]),
                            legend=alt.Legend(format="%", orient="right", direction="vertical", gradientLength=180)),
            tooltip=[alt.Tooltip("nome_estabelecimento:N", title=T("estabelecimento", "merchant")),
                     alt.Tooltip("hora:O", title=T("hora", "hour")),
                     alt.Tooltip("alerta:Q", title=T("% em alerta", "% flagged"), format=".1%")]), 330)

    c3, c4 = st.columns([1.35, 1])
    with c3, st.container(border=True):
        titulo_card(T("Alertas por dia", "Alerts by day"),
                    T("volume diário de transações sinalizadas em janeiro", "daily flagged transactions in January"))
        por_dia = alerta.groupby(alerta.datahora_transacao.dt.date).size().reset_index(name="n")
        por_dia.columns = ["dia", "n"]
        base = alt.Chart(por_dia).encode(x=alt.X("dia:T", title=None, axis=alt.Axis(format=T("%d/%m", "%b %d"), grid=False)))
        grafico(base.mark_area(line={"color": AZUL, "strokeWidth": 2}, interpolate="monotone", color=alt.Gradient(
            gradient="linear", x1=1, x2=1, y1=1, y2=0,
            stops=[alt.GradientStop(color="rgba(51,70,211,0.02)", offset=0), alt.GradientStop(color="rgba(51,70,211,0.25)", offset=1)]))
            .encode(y=alt.Y("n:Q", title=T("alertas", "alerts")))
            + base.mark_point(filled=True, size=40, color=AZUL, opacity=0).encode(
                y="n:Q", tooltip=[alt.Tooltip("dia:T", title=T("dia", "day"), format=T("%d/%m", "%b %d")),
                                  alt.Tooltip("n:Q", title=T("alertas", "alerts"))]),
            240)
    with c4, st.container(border=True):
        titulo_card(T("Taxa de alerta por cidade", "Alert rate by city"),
                    T("% das transações da cidade que foram sinalizadas", "% of the city's transactions that were flagged"))
        por_cid = (df.acao_recomendada != "aprovar").groupby(df.cidade).mean().reset_index(name="taxa")
        base = alt.Chart(por_cid).encode(y=alt.Y("cidade:N", sort="-x", title=None))
        grafico(base.mark_bar(color=VIOLETA, cornerRadiusTopRight=3, cornerRadiusBottomRight=3, height=16).encode(
            x=alt.X("taxa:Q", title=None, axis=alt.Axis(labels=False, grid=False)),
            tooltip=["cidade", alt.Tooltip("taxa:Q", format=".1%", title=T("% em alerta", "% flagged"))])
            + base.mark_text(align="left", dx=5, color=TINTA2, fontSize=11).encode(
                x="taxa:Q", text=alt.Text("taxa:Q", format=".1%")), 240)


# ---------------------------------------------------------------------------
ARQ_TRATATIVAS = BASE / "dados" / "tratativas.csv"
COLS_TRAT = ["id", "resultado", "analista", "observacao", "registrado_em"]


def ler_tratativas():
    if ARQ_TRATATIVAS.exists():
        return pd.read_csv(ARQ_TRATATIVAS, dtype={"id": int})
    return pd.DataFrame(columns=COLS_TRAT)


def salvar_tratativa(reg):
    t = ler_tratativas()
    t = pd.concat([t[t.id != reg["id"]], pd.DataFrame([reg])], ignore_index=True)  # última análise vale
    t.to_csv(ARQ_TRATATIVAS, index=False)


def fatores_shap(linha):
    """Contribuição SHAP de cada variável para uma transação da base (mesmas derivações do treino)."""
    feats, cats = meta["features"], meta["categorias"]
    r = {"hora": int(linha.hora), "is_new_device": int(linha.is_new_device), "nome_estabelecimento": linha.nome_estabelecimento,
         "cidade": linha.cidade, "turno": turno_da_hora(int(linha.hora)), "cidade_regiao": linha.cidade_regiao,
         "valor_transacao": float(linha.valor_transacao), "device_type": linha.device_type,
         "valor_log": float(np.log1p(linha.valor_transacao))}
    X = pd.DataFrame([{f: r[f] for f in feats}])
    for f in feats:
        X[f] = pd.Categorical(X[f], categories=cats[f]) if f in cats else X[f].astype(float)
    c = modelo.predict(X, pred_contrib=True)[0][:-1]
    return pd.DataFrame({"variavel": [nomes().get(f, f) for f in feats], "contrib": c})


def pagina_fila():
    cabecalho(T("Área de negócio", "Business team"), T("Fila priorizada de análise", "Prioritized review queue"),
              T("Os casos estão em ordem de <b>perda esperada = risco de fraude × valor da transação</b>. Assim, um valor alto "
                "com risco médio sobe para o topo, à frente de um valor baixo com risco alto. Selecione um caso para ver "
                "por que ele foi sinalizado e registrar a conclusão da análise.",
                "Cases are sorted by <b>expected loss = fraud risk × transaction amount</b>, so a high amount with medium risk "
                "rises above a low amount with high risk. Select a case to see why it was flagged and record the outcome."))
    trat = ler_tratativas()
    fila = df[df.acao_recomendada != "aprovar"].copy()
    fila["id"] = fila.index
    fila["perda_esperada"] = fila.prob_fraude * fila.valor_transacao
    fila = fila.sort_values("perda_esperada", ascending=False)
    fila["prioridade"] = np.arange(1, len(fila) + 1)
    fila = fila.merge(trat[["id", "resultado"]], on="id", how="left")
    fila["resultado"] = fila.resultado.fillna("Pendente")

    tratados = fila[fila.resultado != "Pendente"]
    conclusivos = tratados[tratados.resultado != "Inconclusivo"]
    with st.container(horizontal=True):
        st.metric(T("Casos na fila", "Cases in queue"), n_br(len(fila)), border=True)
        st.metric(T("Perda esperada total", "Total expected loss"), "R$ " + n_br(fila.perda_esperada.sum()), border=True)
        st.metric(T("Já analisados", "Already reviewed"), n_br(len(tratados)),
                  help=pct_br(len(tratados) / max(len(fila), 1)) + T(" da fila", " of the queue"), border=True)
        st.metric(T("Fraude confirmada", "Confirmed fraud"),
                  pct_br((conclusivos.resultado == "Fraude confirmada").mean()) if len(conclusivos) else "–",
                  help=T("Precisão da fila: % dos casos analisados (excluindo inconclusivos) que eram fraude. "
                         "É o principal indicador para calibrar a próxima versão do modelo.",
                         "Queue precision: % of reviewed cases (excluding inconclusive) that were fraud. "
                         "The main indicator to calibrate the next model version."), border=True)

    with st.container(horizontal=True):
        status = st.segmented_control("Status", ["pend", "feito", "todos"], default="pend",
                                      format_func=lambda s: {"pend": T("Pendentes", "Pending"), "feito": T("Analisados", "Reviewed"),
                                                             "todos": T("Todos", "All")}[s])
        acoes = st.pills(T("Ação do modelo", "Model action"), ["bloquear", "revisar"], default=["bloquear", "revisar"],
                         selection_mode="multi", format_func=acao_txt)
        estabs = st.multiselect(T("Estabelecimento", "Merchant"), sorted(df.nome_estabelecimento.unique()),
                                placeholder=T("Todos", "All"))
    vis = fila[fila.acao_recomendada.isin(acoes or [])]
    if status == "pend":
        vis = vis[vis.resultado == "Pendente"]
    elif status == "feito":
        vis = vis[vis.resultado != "Pendente"]
    if estabs:
        vis = vis[vis.nome_estabelecimento.isin(estabs)]
    vis = vis.reset_index(drop=True)
    tabela = vis.assign(acao=vis.acao_recomendada.map(acao_txt), analise=vis.resultado.map(res_txt))

    c1, c2 = st.columns([1.55, 1])
    with c1:
        sel = st.dataframe(
            tabela[["prioridade", "perda_esperada", "prob_fraude", "valor_transacao", "acao", "nome_estabelecimento",
                    "datahora_transacao", "analise"]],
            hide_index=True, height=620, on_select="rerun", selection_mode="single-row", key="fila_sel",
            column_config={
                "prioridade": st.column_config.NumberColumn("#", width="small"),
                "perda_esperada": st.column_config.NumberColumn(T("Perda esperada", "Expected loss"), format="R$ %.2f",
                                                                help=T("probabilidade de fraude × valor", "fraud probability × amount")),
                "prob_fraude": st.column_config.ProgressColumn(T("Risco", "Risk"), min_value=0, max_value=1, format="percent",
                                                               width="small"),
                "valor_transacao": st.column_config.NumberColumn(T("Valor", "Amount"), format="R$ %.2f"),
                "acao": st.column_config.TextColumn(T("Ação", "Action"), width="small"),
                "nome_estabelecimento": T("Estabelecimento", "Merchant"),
                "datahora_transacao": st.column_config.DatetimeColumn(T("Data/hora", "Date/time"), format="DD/MM HH:mm"),
                "analise": T("Análise", "Outcome"),
            })
    linhas = sel.selection.rows if sel and sel.selection else []
    with c2, st.container(border=True):
        if vis.empty:
            st.caption(T("Nenhum caso com esses filtros.", "No cases match these filters."))
            return
        caso = vis.iloc[linhas[0] if linhas else 0]
        if not linhas:
            st.caption(T("Mostrando o 1º da lista. Clique em uma linha para trocar de caso.",
                         "Showing the first case. Click a row to switch."))
        a = caso.acao_recomendada
        st.html(f"""<div class='tc'><span>{T("Prioridade", "Priority")} #{caso.prioridade} · {caso.datahora_transacao:%d/%m %H:%M}</span></div>
          <div style="display:flex;gap:28px;align-items:flex-end;margin:6px 0 4px">
            <div><div class="big" style="font-size:44px">{brl2(caso.perda_esperada)}</div>
              <span style="font-size:12.5px;color:{TINTA2}">{T("perda esperada", "expected loss")}</span></div>
            <div style="padding-bottom:18px"><span class="selo" style="background:{ACOES[a]}">{acao_txt(a)}</span></div></div>""")
        st.html("<div class='mini'>" + "".join(
            f"<div><span>{r}</span><b>{v}</b></div>" for r, v in
            [(T("Risco", "Risk"), pct_br(caso.prob_fraude)), (T("Valor", "Amount"), brl2(caso.valor_transacao)),
             (T("Hora", "Hour"), f"{int(caso.hora)}h")]) + "</div>")
        regiao = REG_EN.get(caso.cidade_regiao, caso.cidade_regiao) if EN() else caso.cidade_regiao
        st.markdown(f"**{caso.nome_estabelecimento}** · {caso.cidade} ({regiao}) · {caso.device_type}"
                    + (f" · :red-badge[{T('dispositivo novo', 'new device')}]" if caso.is_new_device
                       else f" · {T('dispositivo conhecido', 'known device')}"))
        st.markdown(f"> {caso.motivo_alerta_en if EN() else caso.motivo_alerta}")
        titulo_card(T("Por que o risco está alto", "Why the risk is high"),
                    T("contribuição de cada característica · magenta aumenta, azul reduz",
                      "contribution of each feature · magenta increases, blue decreases"))
        cd = fatores_shap(caso)
        cd["direcao"] = np.where(cd.contrib > 0, "+", "-")
        grafico(alt.Chart(cd).mark_bar(cornerRadius=3, height=12).encode(
            y=alt.Y("variavel:N", sort=alt.EncodingSortField("contrib", op="sum", order="descending"), title=None,
                    axis=alt.Axis(domain=False)),
            x=alt.X("contrib:Q", title=None),
            color=alt.Color("direcao:N", scale=alt.Scale(domain=["+", "-"], range=[MAGENTA, AZUL]), legend=None),
            tooltip=["variavel", alt.Tooltip("contrib:Q", format="+.3f", title=T("contribuição", "contribution"))]), 200)

        anterior = trat[trat.id == caso.id]
        with st.form(f"trat_{caso.id}", clear_on_submit=False, border=False):
            titulo_card(T("Registrar análise", "Record outcome"),
                        T("a conclusão vira rótulo para calibrar a próxima versão do modelo",
                          "the outcome becomes a label to calibrate the next model version"))
            idx = RESULTADOS.index(anterior.resultado.iloc[0]) if len(anterior) else None
            resultado = st.radio(T("Conclusão", "Outcome"), RESULTADOS, index=idx, horizontal=True, format_func=res_txt)
            analista = st.text_input(T("Analista", "Analyst"), value=anterior.analista.iloc[0] if len(anterior) else "")
            obs = st.text_area(T("Observação", "Notes"), value=anterior.observacao.iloc[0] if len(anterior) and pd.notna(
                anterior.observacao.iloc[0]) else "", placeholder=T("Ex.: cliente confirmou a compra por telefone",
                                                                   "E.g. customer confirmed the purchase by phone"), height=80)
            if st.form_submit_button(T("Salvar análise", "Save outcome"), type="primary", icon=":material/check:"):
                if resultado is None:
                    st.warning(T("Escolha uma conclusão antes de salvar.", "Choose an outcome before saving."))
                else:
                    salvar_tratativa({"id": int(caso.id), "resultado": resultado, "analista": analista.strip(),
                                      "observacao": obs.strip(), "registrado_em": pd.Timestamp.now().isoformat(timespec="seconds")})
                    st.toast(T("Análise registrada", "Outcome saved"), icon=":material/check_circle:")
                    st.rerun()

    if len(trat):
        exp = trat.merge(df.reset_index(names="id"), on="id", how="left")
        st.download_button(T("Baixar análises registradas (CSV)", "Download recorded outcomes (CSV)"),
                           exp.to_csv(index=False).encode("utf-8-sig"), "tratativas_fraude.csv", "text/csv",
                           icon=":material/download:")


# ---------------------------------------------------------------------------
def pagina_simulador():
    cabecalho(T("Explicabilidade", "Explainability"), T("Simulador de transação", "Transaction simulator"),
              T("Monte uma transação e veja o risco calculado pelo modelo e o peso de cada variável (SHAP).",
                "Build a transaction and see the model's risk and the weight of each feature (SHAP)."))
    feats, cats, med = meta["features"], meta["categorias"], meta["medianas"]
    linha = {}
    with st.container(border=True):
        cols = st.columns(3)
        valor = cols[0].number_input(T("Valor (R$)", "Amount (R$)"), 0.5, 1000.0, 35.0, step=1.0)
        hora = cols[1].slider(T("Hora", "Hour"), 0, 23, 22, format="%dh")
        novo = cols[2].toggle(T("Dispositivo novo", "New device"), True)
        outros = [f for f in feats if f not in DERIVADAS]
        padrao = {"nome_estabelecimento": "Gas Station", "cidade": "São Paulo", "device_type": "mobile"}
        cols = st.columns(3)
        for i, f in enumerate(outros):
            c = cols[i % 3]
            if f in cats:
                opcoes = cats[f]
                idx = opcoes.index(padrao[f]) if padrao.get(f) in opcoes else 0
                linha[f] = c.selectbox(nomes().get(f, f), opcoes, index=idx)
            else:
                linha[f] = c.number_input(f, value=float(med.get(f, 0)))
    # variáveis derivadas, calculadas como nos dados reais (evita combinações impossíveis)
    linha.update({"valor_transacao": valor, "valor_log": np.log1p(valor), "hora": hora, "is_new_device": int(novo),
                  "turno": turno_da_hora(hora), "cidade_regiao": REGIAO.get(linha.get("cidade"), "Sudeste")})
    X = pd.DataFrame([{f: linha[f] for f in feats}])
    for f in cats:
        X[f] = pd.Categorical(X[f], categories=cats[f])
    for f in feats:
        if f not in cats:
            X[f] = X[f].astype(float)
    p = float(modelo.predict(X)[0])
    contrib = modelo.predict(X, pred_contrib=True)[0][:-1]

    a = acao(p)
    c1, c2 = st.columns([1, 1.7])
    with c1, st.container(border=True):
        titulo_card(T("Probabilidade de fraude", "Fraud probability"))
        st.html(f"""<div class="big">{pct_br(p)}</div>
          <span class="selo" style="background:{ACOES[a] if a != 'aprovar' else '#6b7090'}">{acao_txt(a)}</span>
          <div class="gauge"><i style="width:{T_REV*100}%;background:#dfe1ea"></i>
            <i style="width:{(T_BLOQ-T_REV)*100}%;background:{AZUL}"></i><i style="flex:1;background:{MAGENTA}"></i>
            <div class="m" data-v="{pct_br(p, 0)}" style="left:calc({min(max(p, 0.005), 0.995)*100}% - 1px)"></div></div>
          <div class="escala"><span style="width:{T_REV*100}%">{acao_txt("aprovar")}</span>
            <span style="width:{(T_BLOQ-T_REV)*100}%">{acao_txt("revisar")} · ≥ {T_REV:.0%}</span>
            <span>{acao_txt("bloquear")} · ≥ {T_BLOQ:.0%}</span></div>""")
    with c2, st.container(border=True):
        titulo_card(T("O que pesou na decisão", "What drove the decision"),
                    T("contribuição SHAP de cada variável (log-odds) · magenta aumenta o risco, azul reduz",
                      "SHAP contribution of each feature (log-odds) · magenta increases the risk, blue decreases"))
        cd = pd.DataFrame({"variavel": [nomes().get(f, f) for f in feats], "contrib": contrib})
        cd["direcao"] = np.where(cd.contrib > 0, "+", "-")
        base = alt.Chart(cd).encode(y=alt.Y("variavel:N", sort=alt.EncodingSortField("contrib", op="sum", order="descending"),
                                            title=None, axis=alt.Axis(domain=False)))
        grafico(base.mark_bar(cornerRadius=3, height=16).encode(
            x=alt.X("contrib:Q", title=None),
            color=alt.Color("direcao:N", scale=alt.Scale(domain=["+", "-"], range=[MAGENTA, AZUL]), legend=None),
            tooltip=["variavel", alt.Tooltip("contrib:Q", format="+.3f", title=T("contribuição", "contribution"))])
            + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=TINTA2).encode(x="x:Q"), 270)


# ---------------------------------------------------------------------------
def pagina_drift():
    cabecalho(T("Governança", "Governance"), T("Monitoramento de drift", "Drift monitoring"),
              T("PSI entre 2025 (treino) e janeiro/2026. Abaixo de 0,10 é estável; de 0,10 a 0,25 pede atenção; "
                "acima de 0,25 indica mudança relevante.",
                "PSI between 2025 (training) and January/2026. Below 0.10 is stable; 0.10 to 0.25 needs attention; "
                "above 0.25 means a relevant shift."))
    d = drift.copy()
    d["faixa"] = pd.cut(d.psi, [-1, 0.10, 0.25, 99], labels=["estável", "atenção", "mudança"]).astype(str)
    d["faixa_txt"] = d.faixa.map(lambda f: FAIXA_EN.get(f, f) if EN() else f)
    d["status_txt"] = d.status.map(lambda s: s.replace(s.split(" ", 1)[-1], FAIXA_EN.get(s.split(" ", 1)[-1], "")) if EN() else s)
    d["psi_plot"] = d.psi.clip(lower=1e-5)  # escala log não aceita zero
    d["folga"] = 0.10 / d.psi_plot  # quantas vezes abaixo do limite de atenção
    d["inicio"] = 1e-5
    n_est = int((d.faixa == "estável").sum())
    with st.container(horizontal=True):
        st.metric(T("Variáveis estáveis", "Stable features"), f"{n_est} {T('de', 'of')} {len(d)}", border=True)
        st.metric(T("Maior PSI", "Highest PSI"), n_br(d.psi.max(), 4),
                  help=T("variável: ", "feature: ") + d.loc[d.psi.idxmax(), "variavel"], border=True)
        st.metric(T("Folga até o limite de atenção", "Headroom to the attention limit"), f"{n_br(d.folga.min())}x",
                  help=T("o maior PSI ainda está essa quantidade de vezes abaixo de 0,10",
                         "the highest PSI is still this many times below 0.10"), border=True)
    with st.container(border=True):
        titulo_card(T("PSI por variável · escala logarítmica", "PSI by feature · log scale"),
                    T("os valores são tão pequenos que só aparecem em escala log · linhas = limites de atenção (0,10) e de mudança (0,25)",
                      "values are so small they only show on a log scale · lines = attention (0.10) and shift (0.25) limits"))
        x_log = alt.Scale(type="log", domain=[1e-5, 1])
        faixas = pd.DataFrame({"ini": [0.10, 0.25], "fim": [0.25, 1], "cor": ["#efe7fa", "#fde3ee"]})
        fundo = alt.Chart(faixas).mark_rect(opacity=0.8).encode(
            x=alt.X("ini:Q", scale=x_log), x2="fim:Q", color=alt.Color("cor:N", scale=None))
        limites = pd.DataFrame({"x": [0.10, 0.25], "rot": [T("atenção 0,10", "attention 0.10"), T("mudança 0,25", "shift 0.25")]})
        linhas = alt.Chart(limites).mark_rule(color=TINTA2, strokeDash=[4, 4]).encode(x=alt.X("x:Q", scale=x_log))
        rot_lim = alt.Chart(limites).mark_text(align="left", dx=4, dy=-6, color=TINTA2, fontSize=11).encode(
            x=alt.X("x:Q", scale=x_log), y=alt.value(0), text="rot:N")
        base = alt.Chart(d).encode(y=alt.Y("variavel:N", sort=alt.EncodingSortField("psi", order="descending"), title=None,
                                           axis=alt.Axis(domain=False, grid=True)))
        hastes = base.mark_rule(color="#c9cdf3", strokeWidth=2).encode(x=alt.X("inicio:Q", scale=x_log), x2="psi_plot:Q")
        pontos = base.mark_circle(size=140, opacity=1, stroke="white", strokeWidth=2).encode(
            x=alt.X("psi_plot:Q", title=T("PSI (escala log)", "PSI (log scale)"), scale=x_log, axis=alt.Axis(format="~g")),
            color=alt.Color("faixa:N", scale=alt.Scale(domain=["estável", "atenção", "mudança"], range=[AZUL, VIOLETA, MAGENTA]),
                            legend=None),
            tooltip=["variavel", alt.Tooltip("psi:Q", format=".5f", title="PSI"), alt.Tooltip("faixa_txt:N", title="status")])
        rotulos = base.mark_text(align="left", dx=10, color=TINTA2, fontSize=11).encode(
            x="psi_plot:Q", text=alt.Text("psi:Q", format=".5f"))
        grafico(fundo + linhas + rot_lim + hastes + pontos + rotulos, 360)
    st.dataframe(d[["variavel", "psi", "status_txt"]], hide_index=True,
                 column_config={"variavel": T("Variável", "Feature"), "psi": st.column_config.NumberColumn("PSI", format="%.5f"),
                                "status_txt": "Status"})
    st.info(T("Janeiro/2026 está estável em relação a 2025 em todas as variáveis do modelo. Mas PSI estável não "
              "garante que o padrão de fraude não mudou: em dez/25 surgiu fraude às 20h–21h sem mudar a distribuição das "
              "variáveis. Por isso a regra combina PSI com a **precisão da fila manual**: re-treinar todo mês, ou antes "
              "se o PSI do score passar de 0,25 ou a precisão da fila cair mais de 20%.",
              "January/2026 is stable versus 2025 on every model feature. But a stable PSI does not guarantee the fraud "
              "pattern has not changed: in Dec/25 fraud appeared at 8–9 pm without shifting the feature distributions. "
              "That is why the rule combines PSI with **manual queue precision**: retrain every month, or sooner if the "
              "score's PSI goes above 0.25 or queue precision drops more than 20%."), icon=":material/info:")


pagina = st.navigation([
    st.Page(pagina_visao, title=T("Visão jan/2026", "Jan/2026 overview"), icon=":material/dashboard:", url_path="visao", default=True),
    st.Page(pagina_fila, title=T("Fila priorizada", "Prioritized queue"), icon=":material/assignment:", url_path="fila"),
    st.Page(pagina_simulador, title=T("Simulador", "Simulator"), icon=":material/science:", url_path="simulador"),
    st.Page(pagina_drift, title="Drift", icon=":material/monitoring:", url_path="drift"),
], position="top")
with st.container(horizontal=True, horizontal_alignment="right"):
    st.segmented_control("Idioma / Language", ["PT", "EN"], key="lang", label_visibility="collapsed",
                         format_func=lambda x: {"PT": "Português", "EN": "English"}[x])
pagina.run()
