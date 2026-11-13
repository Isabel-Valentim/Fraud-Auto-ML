"""Dashboard do case de fraude: visão de jan/2026, fila do analista, simulador e drift."""
import json
from pathlib import Path

import altair as alt
import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Radar de Fraude", page_icon=":material/shield:", layout="wide")

BASE = Path(__file__).parent
# paleta validada (daltonismo + contraste): azul = padrão/revisar, magenta = risco/bloquear, violeta = 3ª série
AZUL, MAGENTA, VIOLETA = "#3346d3", "#e0005f", "#8a3fc7"
CINZA, TINTA, TINTA2, GRADE = "#c4c6d4", "#11163a", "#5a5f7a", "#e8e9f0"
GRAD = "linear-gradient(90deg, #2b39c4 0%, #8a3fc7 55%, #e0005f 100%)"
GRAD_CAPA = "linear-gradient(120deg, #08137a 0%, #1b1a8f 25%, #521a8c 58%, #a10a74 82%, #e0005f 100%)"
ACOES = {"bloquear": MAGENTA, "revisar": AZUL, "aprovar": CINZA}
FONTE = "IBM Plex Sans, Segoe UI, sans-serif"

st.html(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
.hero {{ background: {GRAD_CAPA}; color: #fff; border-radius: 6px; padding: 34px 40px 30px; margin-bottom: 6px }}
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
.hero {{ position: relative; overflow: hidden; border-radius: 18px; box-shadow: 0 18px 40px rgba(20,24,80,.22) }}
.hero::after {{ content: ''; position: absolute; width: 520px; height: 520px; right: -140px; top: -200px; border-radius: 50%;
               background: radial-gradient(circle at 35% 35%, rgba(255,255,255,.22), rgba(255,255,255,0) 60%);
               border: 1px solid rgba(255,255,255,.18) }}
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
NOMES = {"hora": "Horário", "is_new_device": "Dispositivo novo", "nome_estabelecimento": "Estabelecimento",
         "cidade": "Cidade", "turno": "Turno", "cidade_regiao": "Região", "valor_transacao": "Valor",
         "device_type": "Tipo de dispositivo", "valor_log": "Valor (log)"}


def n_br(v):
    return f"{v:,.0f}".replace(",", ".")


def brl2(v):
    return "R$ " + f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct_br(v, d=1):
    return f"{v:.{d}%}".replace(".", ",")


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
    alerta = df[df.acao_recomendada != "aprovar"]
    n_rev, n_bloq = (df.acao_recomendada == "revisar").sum(), (df.acao_recomendada == "bloquear").sum()
    m = meta["metricas_teste"]
    nums = [(n_br(len(df)), "transações em jan/2026"), (n_br(n_rev), f"para revisar · {pct_br(n_rev / len(df))}"),
            (n_br(n_bloq), "bloqueio automático"),
            ("R$ " + n_br((df.prob_fraude * df.valor_transacao).sum()), "fraude esperada"),
            (f"{m['pr_auc']:.2f}".replace(".", ","), f"PR-AUC · score atual {m['auc_score_interno']:.2f} AUC".replace(".", ","))]
    st.html(f"""<div class="hero"><div class="k">Radar de Fraude · Janeiro/2026</div>
      <h1>Quais transações merecem atenção agora</h1><div class="r"></div>
      <p>Modelo LightGBM treinado em 2025 · revisar se a probabilidade for ≥ {T_REV:.0%} · bloquear se for ≥ {T_BLOQ:.0%}</p>
      <div class="nums">{''.join(f'<div><b>{v}</b><span>{r}</span></div>' for v, r in nums)}</div></div>""")

    c1, c2 = st.columns([1.35, 1])
    with c1, st.container(border=True):
        titulo_card("Alertas por hora do dia", "a fraude se concentra à noite: o pico é às 22h–23h")
        por_hora = alerta.groupby(["hora", "acao_recomendada"]).size().reset_index(name="n")
        grafico(alt.Chart(por_hora).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, stroke="white", strokeWidth=1).encode(
            x=alt.X("hora:O", title=None, axis=alt.Axis(labelExpr="datum.value + 'h'", labelAngle=0)),
            y=alt.Y("n:Q", title="transações"),
            color=alt.Color("acao_recomendada:N", title=None,
                            scale=alt.Scale(domain=["revisar", "bloquear"], range=[AZUL, MAGENTA])),
            order=alt.Order("acao_recomendada:N", sort="descending"),
            tooltip=[alt.Tooltip("hora:O", title="hora"), alt.Tooltip("acao_recomendada:N", title="ação"),
                     alt.Tooltip("n:Q", title="transações", format=",")]), 270)
    with c2, st.container(border=True):
        titulo_card("Alertas por estabelecimento", "volume de transações sinalizadas")
        por_estab = alerta.groupby("nome_estabelecimento").size().reset_index(name="n")
        base = alt.Chart(por_estab).encode(y=alt.Y("nome_estabelecimento:N", sort="-x", title=None))
        grafico(base.mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3, height=16).encode(
            x=alt.X("n:Q", title=None, axis=alt.Axis(labels=False, grid=False)),
            color=alt.Color("n:Q", scale=alt.Scale(range=["#b9c0f2", AZUL]), legend=None),
            tooltip=[alt.Tooltip("nome_estabelecimento:N", title="estabelecimento"), alt.Tooltip("n:Q", title="alertas")])
            + base.mark_text(align="left", dx=5, color=TINTA2, fontSize=11).encode(x="n:Q", text=alt.Text("n:Q", format=",")),
            270)

    with st.container(border=True):
        titulo_card("Onde e quando: taxa de alerta por estabelecimento × hora",
                    "cada estabelecimento tem a sua janela de risco · quanto mais escuro, maior a % de transações em alerta")
        mapa = (df.assign(alerta=df.acao_recomendada != "aprovar")
                .groupby(["nome_estabelecimento", "hora"]).alerta.mean().reset_index())
        ordem = mapa.groupby("nome_estabelecimento").alerta.mean().sort_values(ascending=False).index.tolist()
        grafico(alt.Chart(mapa).mark_rect(cornerRadius=2, stroke="white", strokeWidth=2).encode(
            x=alt.X("hora:O", title=None, axis=alt.Axis(labelExpr="datum.value + 'h'", labelAngle=0, domain=False)),
            y=alt.Y("nome_estabelecimento:N", sort=ordem, title=None, axis=alt.Axis(domain=False)),
            color=alt.Color("alerta:Q", title="% em alerta", scale=alt.Scale(range=["#f3f4f8", "#f9d3e2", "#ef6a9c", MAGENTA, "#7a0034"]),
                            legend=alt.Legend(format="%", orient="right", direction="vertical", gradientLength=180)),
            tooltip=[alt.Tooltip("nome_estabelecimento:N", title="estabelecimento"), alt.Tooltip("hora:O", title="hora"),
                     alt.Tooltip("alerta:Q", title="% em alerta", format=".1%")]), 330)

    c3, c4 = st.columns([1.35, 1])
    with c3, st.container(border=True):
        titulo_card("Alertas por dia", "volume diário de transações sinalizadas em janeiro")
        por_dia = alerta.groupby(alerta.datahora_transacao.dt.date).size().reset_index(name="n")
        por_dia.columns = ["dia", "n"]
        base = alt.Chart(por_dia).encode(x=alt.X("dia:T", title=None, axis=alt.Axis(format="%d/%m", grid=False)))
        grafico(base.mark_area(line={"color": AZUL, "strokeWidth": 2}, interpolate="monotone", color=alt.Gradient(
            gradient="linear", x1=1, x2=1, y1=1, y2=0,
            stops=[alt.GradientStop(color="rgba(51,70,211,0.02)", offset=0), alt.GradientStop(color="rgba(51,70,211,0.25)", offset=1)]))
            .encode(y=alt.Y("n:Q", title="alertas"))
            + base.mark_point(filled=True, size=40, color=AZUL, opacity=0).encode(
                y="n:Q", tooltip=[alt.Tooltip("dia:T", title="dia", format="%d/%m"), alt.Tooltip("n:Q", title="alertas")]),
            240)
    with c4, st.container(border=True):
        titulo_card("Taxa de alerta por cidade", "% das transações da cidade que foram sinalizadas")
        por_cid = (df.acao_recomendada != "aprovar").groupby(df.cidade).mean().reset_index(name="taxa")
        base = alt.Chart(por_cid).encode(y=alt.Y("cidade:N", sort="-x", title=None))
        grafico(base.mark_bar(color=VIOLETA, cornerRadiusTopRight=3, cornerRadiusBottomRight=3, height=16).encode(
            x=alt.X("taxa:Q", title=None, axis=alt.Axis(labels=False, grid=False)),
            tooltip=["cidade", alt.Tooltip("taxa:Q", format=".1%", title="% em alerta")])
            + base.mark_text(align="left", dx=5, color=TINTA2, fontSize=11).encode(
                x="taxa:Q", text=alt.Text("taxa:Q", format=".1%")), 240)


# ---------------------------------------------------------------------------
ARQ_TRATATIVAS = BASE / "dados" / "tratativas.csv"
COLS_TRAT = ["id", "resultado", "analista", "observacao", "registrado_em"]
RESULTADOS = ["Fraude confirmada", "Não é fraude", "Inconclusivo"]


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
    return pd.DataFrame({"variavel": [NOMES.get(f, f) for f in feats], "contrib": c})


def pagina_fila():
    cabecalho("Área de negócio", "Fila priorizada de análise",
              "Os casos estão em ordem de <b>perda esperada = risco de fraude × valor da transação</b>. Assim, um valor alto "
              "com risco médio sobe para o topo, à frente de um valor baixo com risco alto. Selecione um caso para ver "
              "por que ele foi sinalizado e registrar a conclusão da análise.")
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
        st.metric("Casos na fila", n_br(len(fila)), border=True)
        st.metric("Perda esperada total", "R$ " + n_br(fila.perda_esperada.sum()), border=True)
        st.metric("Já analisados", n_br(len(tratados)), help=f"{pct_br(len(tratados) / max(len(fila), 1))} da fila",
                  border=True)
        st.metric("Fraude confirmada", pct_br((conclusivos.resultado == "Fraude confirmada").mean()) if len(conclusivos) else "–",
                  help="Precisão da fila: % dos casos analisados (excluindo inconclusivos) que eram fraude. "
                       "É o principal indicador para calibrar a próxima versão do modelo.", border=True)

    with st.container(horizontal=True):
        status = st.segmented_control("Status", ["Pendentes", "Analisados", "Todos"], default="Pendentes")
        acoes = st.pills("Ação do modelo", ["bloquear", "revisar"], default=["bloquear", "revisar"], selection_mode="multi")
        estabs = st.multiselect("Estabelecimento", sorted(df.nome_estabelecimento.unique()), placeholder="Todos")
    vis = fila[fila.acao_recomendada.isin(acoes or [])]
    if status == "Pendentes":
        vis = vis[vis.resultado == "Pendente"]
    elif status == "Analisados":
        vis = vis[vis.resultado != "Pendente"]
    if estabs:
        vis = vis[vis.nome_estabelecimento.isin(estabs)]
    vis = vis.reset_index(drop=True)

    c1, c2 = st.columns([1.55, 1])
    with c1:
        sel = st.dataframe(
            vis[["prioridade", "perda_esperada", "prob_fraude", "valor_transacao", "acao_recomendada", "nome_estabelecimento",
                 "datahora_transacao", "resultado"]],
            hide_index=True, height=620, on_select="rerun", selection_mode="single-row", key="fila_sel",
            column_config={
                "prioridade": st.column_config.NumberColumn("#", width="small"),
                "perda_esperada": st.column_config.NumberColumn("Perda esperada", format="R$ %.2f",
                                                                help="probabilidade de fraude × valor"),
                "prob_fraude": st.column_config.ProgressColumn("Risco", min_value=0, max_value=1, format="percent", width="small"),
                "valor_transacao": st.column_config.NumberColumn("Valor", format="R$ %.2f"),
                "acao_recomendada": st.column_config.TextColumn("Ação", width="small"),
                "nome_estabelecimento": "Estabelecimento",
                "datahora_transacao": st.column_config.DatetimeColumn("Data/hora", format="DD/MM HH:mm"),
                "resultado": "Análise",
            })
    linhas = sel.selection.rows if sel and sel.selection else []
    with c2, st.container(border=True):
        if vis.empty:
            st.caption("Nenhum caso com esses filtros.")
            return
        caso = vis.iloc[linhas[0] if linhas else 0]
        if not linhas:
            st.caption("Mostrando o 1º da lista. Clique em uma linha para trocar de caso.")
        a = caso.acao_recomendada
        st.html(f"""<div class='tc'><span>Prioridade #{caso.prioridade} · {caso.datahora_transacao:%d/%m %H:%M}</span></div>
          <div style="display:flex;gap:28px;align-items:flex-end;margin:6px 0 4px">
            <div><div class="big" style="font-size:44px">{brl2(caso.perda_esperada)}</div>
              <span style="font-size:12.5px;color:{TINTA2}">perda esperada</span></div>
            <div style="padding-bottom:18px"><span class="selo" style="background:{ACOES[a]}">{a}</span></div></div>""")
        st.html("<div class='mini'>" + "".join(
            f"<div><span>{r}</span><b>{v}</b></div>" for r, v in
            [("Risco", pct_br(caso.prob_fraude)), ("Valor", brl2(caso.valor_transacao)), ("Hora", f"{int(caso.hora)}h")]) + "</div>")
        st.markdown(f"**{caso.nome_estabelecimento}** · {caso.cidade} ({caso.cidade_regiao}) · {caso.device_type}"
                    f"{' · :red-badge[dispositivo novo]' if caso.is_new_device else ' · dispositivo conhecido'}")
        st.markdown(f"> {caso.motivo_alerta}")
        titulo_card("Por que o risco está alto", "contribuição de cada característica · magenta aumenta, azul reduz")
        cd = fatores_shap(caso)
        cd["direcao"] = np.where(cd.contrib > 0, "aumenta", "reduz")
        grafico(alt.Chart(cd).mark_bar(cornerRadius=3, height=12).encode(
            y=alt.Y("variavel:N", sort=alt.EncodingSortField("contrib", op="sum", order="descending"), title=None,
                    axis=alt.Axis(domain=False)),
            x=alt.X("contrib:Q", title=None),
            color=alt.Color("direcao:N", scale=alt.Scale(domain=["aumenta", "reduz"], range=[MAGENTA, AZUL]), legend=None),
            tooltip=["variavel", alt.Tooltip("contrib:Q", format="+.3f", title="contribuição")]), 200)

        anterior = trat[trat.id == caso.id]
        with st.form(f"trat_{caso.id}", clear_on_submit=False, border=False):
            titulo_card("Registrar análise", "a conclusão vira rótulo para calibrar a próxima versão do modelo")
            idx = RESULTADOS.index(anterior.resultado.iloc[0]) if len(anterior) else None
            resultado = st.radio("Conclusão", RESULTADOS, index=idx, horizontal=True)
            analista = st.text_input("Analista", value=anterior.analista.iloc[0] if len(anterior) else "")
            obs = st.text_area("Observação", value=anterior.observacao.iloc[0] if len(anterior) and pd.notna(
                anterior.observacao.iloc[0]) else "", placeholder="Ex.: cliente confirmou a compra por telefone", height=80)
            if st.form_submit_button("Salvar análise", type="primary", icon=":material/check:"):
                if resultado is None:
                    st.warning("Escolha uma conclusão antes de salvar.")
                else:
                    salvar_tratativa({"id": int(caso.id), "resultado": resultado, "analista": analista.strip(),
                                      "observacao": obs.strip(), "registrado_em": pd.Timestamp.now().isoformat(timespec="seconds")})
                    st.toast("Análise registrada", icon=":material/check_circle:")
                    st.rerun()

    if len(trat):
        exp = trat.merge(df.reset_index(names="id"), on="id", how="left")
        st.download_button("Baixar análises registradas (CSV)", exp.to_csv(index=False).encode("utf-8-sig"),
                           "tratativas_fraude.csv", "text/csv", icon=":material/download:")


# ---------------------------------------------------------------------------
def pagina_simulador():
    cabecalho("Explicabilidade", "Simulador de transação",
              "Monte uma transação e veja o risco calculado pelo modelo e o peso de cada variável (SHAP).")
    feats, cats, med = meta["features"], meta["categorias"], meta["medianas"]
    linha = {}
    with st.container(border=True):
        cols = st.columns(3)
        valor = cols[0].number_input("Valor (R$)", 0.5, 1000.0, 35.0, step=1.0)
        hora = cols[1].slider("Hora", 0, 23, 22, format="%dh")
        novo = cols[2].toggle("Dispositivo novo", True)
        outros = [f for f in feats if f not in DERIVADAS]
        padrao = {"nome_estabelecimento": "Gas Station", "cidade": "São Paulo", "device_type": "mobile"}
        cols = st.columns(3)
        for i, f in enumerate(outros):
            c = cols[i % 3]
            if f in cats:
                opcoes = cats[f]
                idx = opcoes.index(padrao[f]) if padrao.get(f) in opcoes else 0
                linha[f] = c.selectbox(NOMES.get(f, f), opcoes, index=idx)
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
        titulo_card("Probabilidade de fraude")
        st.html(f"""<div class="big">{pct_br(p)}</div>
          <span class="selo" style="background:{ACOES[a] if a != 'aprovar' else '#6b7090'}">{a}</span>
          <div class="gauge"><i style="width:{T_REV*100}%;background:#dfe1ea"></i>
            <i style="width:{(T_BLOQ-T_REV)*100}%;background:{AZUL}"></i><i style="flex:1;background:{MAGENTA}"></i>
            <div class="m" data-v="{pct_br(p, 0)}" style="left:calc({min(max(p, 0.005), 0.995)*100}% - 1px)"></div></div>
          <div class="escala"><span style="width:{T_REV*100}%">aprovar</span>
            <span style="width:{(T_BLOQ-T_REV)*100}%">revisar · ≥ {T_REV:.0%}</span><span>bloquear · ≥ {T_BLOQ:.0%}</span></div>""")
    with c2, st.container(border=True):
        titulo_card("O que pesou na decisão", "contribuição SHAP de cada variável (log-odds) · magenta aumenta o risco, azul reduz")
        cd = pd.DataFrame({"variavel": [NOMES.get(f, f) for f in feats], "contrib": contrib})
        cd["direcao"] = np.where(cd.contrib > 0, "aumenta o risco", "reduz o risco")
        base = alt.Chart(cd).encode(y=alt.Y("variavel:N", sort=alt.EncodingSortField("contrib", op="sum", order="descending"),
                                            title=None, axis=alt.Axis(domain=False)))
        grafico(base.mark_bar(cornerRadius=3, height=16).encode(
            x=alt.X("contrib:Q", title=None),
            color=alt.Color("direcao:N", scale=alt.Scale(domain=["aumenta o risco", "reduz o risco"], range=[MAGENTA, AZUL]),
                            legend=None),
            tooltip=["variavel", alt.Tooltip("contrib:Q", format="+.3f", title="contribuição")])
            + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=TINTA2).encode(x="x:Q"), 270)


# ---------------------------------------------------------------------------
def pagina_drift():
    cabecalho("Governança", "Monitoramento de drift",
              "PSI entre 2025 (treino) e janeiro/2026. Abaixo de 0,10 é estável; de 0,10 a 0,25 pede atenção; acima de 0,25 indica mudança relevante.")
    d = drift.copy()
    d["faixa"] = pd.cut(d.psi, [-1, 0.10, 0.25, 99], labels=["estável", "atenção", "mudança"]).astype(str)
    lim = max(0.3, d.psi.max() * 1.15)
    with st.container(border=True):
        titulo_card("PSI por variável", "as faixas ao fundo marcam os limites de atenção (0,10) e de mudança (0,25)")
        faixas = pd.DataFrame({"ini": [0, 0.10, 0.25], "fim": [0.10, 0.25, lim], "faixa": ["estável", "atenção", "mudança"]})
        fundo = alt.Chart(faixas).mark_rect(opacity=0.5).encode(
            x="ini:Q", x2="fim:Q", color=alt.Color("faixa:N", scale=alt.Scale(domain=["estável", "atenção", "mudança"],
                                                                                range=["#f4f5fa", "#ece6f8", "#fbe0ea"]), legend=None))
        base = alt.Chart(d).encode(y=alt.Y("variavel:N", sort="-x", title=None, axis=alt.Axis(domain=False)))
        barras = base.mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3, height=14).encode(
            x=alt.X("psi:Q", title="PSI", scale=alt.Scale(domain=[0, lim])),
            color=alt.Color("faixa:N", scale=alt.Scale(domain=["estável", "atenção", "mudança"], range=[AZUL, VIOLETA, MAGENTA]),
                            legend=alt.Legend(title=None)),
            tooltip=["variavel", alt.Tooltip("psi:Q", format=".3f"), "faixa"])
        rotulos = base.mark_text(align="left", dx=5, color=TINTA2, fontSize=11).encode(x="psi:Q", text=alt.Text("psi:Q", format=".3f"))
        grafico(fundo + barras + rotulos, 340)
    st.dataframe(d[["variavel", "psi", "status"]], hide_index=True,
                 column_config={"variavel": "Variável", "psi": st.column_config.NumberColumn("PSI", format="%.3f"),
                                "status": "Status"})
    st.info("Janeiro/2026 está estável em relação a 2025 em todas as variáveis do modelo. Mas PSI estável não "
            "garante que o padrão de fraude não mudou: em dez/25 surgiu fraude às 20h–21h sem mudar a distribuição das "
            "variáveis. Por isso a regra combina PSI com a **precisão da fila manual**: re-treinar todo mês, ou antes "
            "se o PSI do score passar de 0,25 ou a precisão da fila cair mais de 20%.", icon=":material/info:")


st.navigation([
    st.Page(pagina_visao, title="Visão jan/2026", icon=":material/dashboard:", url_path="visao", default=True),
    st.Page(pagina_fila, title="Fila priorizada", icon=":material/assignment:", url_path="fila"),
    st.Page(pagina_simulador, title="Simulador", icon=":material/science:", url_path="simulador"),
    st.Page(pagina_drift, title="Drift", icon=":material/monitoring:", url_path="drift"),
], position="top").run()
