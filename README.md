# Radar de Fraude

**Detecção de fraude em transações com decisão por custo, fila priorizada para a área de negócio e explicação de cada alerta.**

Hoje a triagem de fraude é quase toda manual, e o score usado para ajudar tem AUC 0,50: equivale a um sorteio.
Este projeto troca esse processo por um modelo que dá a cada transação uma **probabilidade de fraude**, uma **ação**
(aprovar, revisar ou bloquear) e o **motivo** em português. Os casos chegam à área de negócio numa fila ordenada por
**perda esperada (risco × valor)**, e cada análise registrada vira rótulo para a próxima versão do modelo.

| | |
|---|---|
| **R$ 42.351** de economia no mês de teste | −58% no custo da fraude em relação a aprovar tudo |
| **83%** das fraudes capturadas | revisando só 10% das transações |
| **PR-AUC 0,65** | contra 0,05 do score atual (12x) e 0,25 de uma regra de especialista |
| **79%** do valor fraudado evitado | com 7,9% das transações na fila de análise |

> As métricas são do **teste em dezembro/2025**, um mês que o modelo nunca viu no treino e que traz um padrão novo de
> fraude. É o último mês com rótulo; para janeiro/2026 (sem rótulo) existe só a estimativa de fraude esperada.

![Impacto financeiro](artefatos/figs/impacto_financeiro.png)

---

## Como funciona

```mermaid
flowchart LR
    A[Transações 2025<br/>523 mil] --> B[Variáveis<br/>contexto + feriados BrasilAPI]
    B --> C[Validação temporal<br/>treino jan–out · valid nov · teste dez]
    C --> D[AutoGluon<br/>20 modelos como régua]
    D --> E[LightGBM extra_trees<br/>campeão leve e explicável]
    E --> F[Limiares por custo<br/>R$ 4 análise · R$ 15 bloqueio indevido]
    F --> G[Aprovar · Revisar · Bloquear]
    G --> H[SHAP<br/>motivo de cada alerta]
    H --> I[Fila priorizada<br/>risco × valor]
    I --> J[Tratativa da área de negócio<br/>fraude · não é fraude · inconclusivo]
    J -->|rótulos novos| K[Re-treino mensal<br/>+ recalibração]
    K --> E
    I --> L[Monitoramento<br/>PSI + precisão da fila]
    L --> K
```

## O que os dados mostraram

| Achado | Evidência |
|---|---|
| O horário é o sinal mais forte | 22h–23h: ~30% das transações são fraude; das 9h às 11h e das 15h às 19h, nenhuma |
| Dispositivo novo multiplica o risco | ~6x; junto com 22h–23h chega a 60% |
| Cada estabelecimento tem sua janela | posto de madrugada, restaurantes no almoço; posto + SP/RJ + dispositivo novo de madrugada ≈ 78% |
| O score atual não separa nada | AUC 0,50 |
| A fraude muda com o tempo | dezembro: 5,4% de fraude e um padrão novo às 20h–21h (*concept drift*) |
| Histórico do cliente e feriados não ajudam | testados e descartados pela importância por permutação (23 → 9 variáveis) |

![Janela de fraude por estabelecimento](artefatos/figs/eda_estab_hora.png)

## Fila priorizada e plano de calibração

Ordenar só pelo risco põe no topo compras de R$ 20 com risco alto, enquanto uma compra alta com risco médio fica
escondida. A fila do painel é ordenada por **perda esperada = probabilidade × valor**: os 10% primeiros concentram
28% da perda esperada (ordenando só pelo risco seriam 21%), e 45% deles têm risco médio.

![Fila priorizada](artefatos/figs/fila_priorizada.png)

No painel, a área de negócio vê a fila ordenada, as características de cada caso, **por que o risco está alto** (SHAP)
e registra a **tratativa**. Essas conclusões são o plano de calibração das próximas versões:

- **Precisão da fila** (% de fraude confirmada entre os casos analisados) é o termômetro principal. Uma queda acima de 20% antecipa o re-treino.
- Os limiares de revisar e bloquear são recalibrados com o custo real observado.
- Casos "não é fraude" com risco alto mostram onde o modelo erra e entram no treino.
- Uma versão nova só entra depois de um mês em *shadow mode*, comparada com a anterior.

![Painel da área de negócio](artefatos/figs/painel_fila.png)

## Estrutura

```
├── case_fraude.ipynb              # todo o raciocínio: EDA, modelagem, decisão, explicabilidade, drift
├── app/                           # dashboard Streamlit
│   ├── streamlit_app.py           # visão jan/26 · fila priorizada com tratativa · simulador · drift
│   ├── modelo_lgbm.txt            # modelo final
│   └── dados/                     # jan/26 classificado e PSI
├── transacoes_historico_2025.csv  # base de treino (523 mil transações, com rótulo de fraude)
├── transacoes_jan2026.csv         # base a classificar (44 mil transações, sem rótulo)
├── entrega/                       # jan/26 classificado (probabilidade, ação e motivo)
├── artefatos/                     # métricas, gráficos e resultados salvos do AutoGluon
├── dados_externos/                # cache da API de feriados (BrasilAPI)
└── extras/
    └── fontes/                    # fonte dos gráficos (licença OFL)
```


## Como rodar

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

1. Execute `case_fraude.ipynb`. Com `USAR_AUTOGLUON = False` (padrão), ele lê os resultados salvos do benchmark e roda
   em poucos minutos. Com `True`, re-treina os 20 modelos (~25 min).
2. Dashboard: `streamlit run app/streamlit_app.py`

**Reprodutibilidade:** o notebook foi re-executado do zero num ambiente limpo, com outra versão de Python e das
bibliotecas, e as 139 métricas salvas e o CSV de entrega saíram idênticos.

## Decisões principais

| Decisão | Por quê |
|---|---|
| PR-AUC em vez de acurácia | Com ~4% de fraude, aprovar tudo já dá 96% de acurácia |
| Validação temporal, teste em dezembro | O modelo é usado no futuro; dezembro tem drift e mede a robustez |
| AutoGluon como régua, LightGBM como modelo | Mesma performance do melhor modelo (diferença de 0,006 de PR-AUC), mas leve e explicável |
| Limiares por custo em R$ | Decisão de negócio explícita e ajustável sem re-treinar |
| Fila por risco × valor | Quem analisa olha primeiro onde está o dinheiro |

## Limitações e próximos passos

- Os dados são sintéticos: padrões muito limpos inflam as métricas em relação ao mundo real.
- As premissas de custo precisam ser validadas com a área de negócio.
- Próximos sinais a incorporar: geolocalização da compra × endereço do cliente, velocidade entre cidades
  (*impossible travel*) e *device fingerprint*.
- Copiloto com LLM: reescrever o motivo de cada alerta em linguagem natural, usando só os fatores do SHAP.

**Stack:** Python · pandas · LightGBM · AutoGluon · SHAP · Streamlit · Altair · Matplotlib · BrasilAPI
