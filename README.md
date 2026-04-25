# 🤖 Modelo Preditivo de Vendas com Machine Learning

> Comparativo de algoritmos de regressão para previsão de receita mensal por categoria — com feature engineering temporal, GridSearchCV e análise de importância de variáveis

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=flat&logo=scikitlearn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-007BFF?style=flat)](https://xgboost.readthedocs.io)
[![Status](https://img.shields.io/badge/Status-Completo-28a745?style=flat)]()

---

## 🎯 Contexto de Negócio

Erros de previsão de vendas geram estoque excessivo, ruptura de produtos, alocação ineficiente da equipe comercial e frustração de clientes. Este projeto desenvolve e compara modelos preditivos para prever a **receita mensal por categoria de produto** em uma rede de varejo PME, com horizonte de previsão de 3 meses.

**Impacto esperado:** Redução do erro de planejamento de ±25% para ±8% (MAPE), permitindo decisões mais assertivas de compra, estoque e metas comerciais.

---

## 🗂️ Estrutura do Projeto

```
projeto2_ml/
├── modelo_preditivo_vendas.py    # Pipeline completo de ML
├── data/
│   └── pedidos.csv               # Dataset base (gerado no Projeto 1)
└── outputs/
    ├── fig5_comparativo_modelos.png   # RMSE, R², Real vs Predito, Feature Importance
    └── fig6_serie_temporal_categorias.png  # Série real vs prevista por categoria
```

---

## 🔧 Pipeline de Machine Learning

```
Dados brutos
    ↓
Agregação mensal por categoria
    ↓
Feature Engineering (lags, MMs, sazonalidade)
    ↓
Split temporal (treino < jun/2024 | teste ≥ jun/2024)
    ↓
Treinamento: Ridge | Random Forest | XGBoost + GridSearchCV
    ↓
Avaliação: RMSE | MAE | R² | MAPE
    ↓
Seleção do melhor modelo
    ↓
Previsão 3 meses futuros
```

---

## 🧠 Feature Engineering

| Feature | Tipo | Descrição |
|---|---|---|
| `receita_lag1/2/3/6` | Lag | Receita dos meses anteriores |
| `receita_mm3/6/12` | Média móvel | Tendência de curto/médio prazo |
| `receita_delta_pct` | Derivada | Variação percentual MoM |
| `mes_sin / mes_cos` | Cíclica | Componentes de sazonalidade (Fourier) |
| `is_q4 / is_novdez` | Dummy | Indicadores de período sazonal forte |
| `ticket_lag1` | Lag | Ticket médio do mês anterior |
| `margem_lag1` | Lag | Margem do mês anterior |

> **Por que features cíclicas?** `mes_sin` e `mes_cos` preservam a continuidade entre dezembro e janeiro, algo que uma variável linear de mês (1–12) não consegue capturar.

---

## 📊 Resultados

### Comparativo de Modelos (conjunto de teste)

| Modelo | RMSE | MAE | R² | MAPE |
|---|---|---|---|---|
| Ridge Regression | R$ 15.538 | R$ 12.957 | 0.95 | 43,1% |
| **Random Forest** | **R$ 14.862** | **R$ 10.729** | **0.95** | **14,6%** |
| XGBoost | R$ 17.856 | R$ 12.615 | 0.93 | 18,7% |

**🏆 Random Forest** vence em RMSE e MAPE. Ridge tem MAPE alto por sensibilidade a outliers sazonais.

### Previsão de Receita — Jan/Fev/Mar 2025

| Categoria | Jan/25 | Fev/25 | Mar/25 |
|---|---|---|---|
| Eletrodomésticos | R$ 161.668 | R$ 164.179 | R$ 169.067 |
| Móveis | R$ 103.727 | R$ 103.382 | R$ 107.471 |
| Eletrônicos | R$ 89.523 | R$ 88.753 | R$ 92.540 |
| Vestuário | R$ 47.305 | R$ 44.859 | R$ 45.100 |
| Beleza e Saúde | R$ 20.357 | R$ 20.160 | R$ 20.162 |
| Alimentos | R$ 9.767 | R$ 9.625 | R$ 9.634 |
| **Total** | **R$ 432.346** | **R$ 430.958** | **R$ 443.974** |

---

## ⚙️ Parâmetros Ótimos (XGBoost via GridSearchCV)

```python
XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror"
)
```

Validação cruzada com `TimeSeriesSplit(n_splits=4)` — evita data leakage em séries temporais.

---

## 🚀 Como Executar

```bash
# Pré-requisito: rodar o Projeto 1 para gerar data/pedidos.csv

pip install pandas numpy scikit-learn xgboost matplotlib seaborn

python modelo_preditivo_vendas.py
```

---

## 📐 Decisões Técnicas

**Por que não usar ARIMA/Prophet?**  
Com múltiplas categorias e features externas (ticket, margem, vendedor), modelos de regressão supervisionada capturam relações multivariadas que modelos univariados de série temporal não conseguem. Para um próximo passo, `Prophet` pode ser comparado como baseline univariado.

**Por que TimeSeriesSplit e não KFold?**  
Em séries temporais, usar dados futuros para treinar e testar em dados passados causa data leakage. O `TimeSeriesSplit` garante que cada fold de validação usa apenas dados posteriores ao treino.

---

## 🛠️ Stack Técnica

| Ferramenta | Uso |
|---|---|
| Python 3.10+ | Linguagem base |
| Scikit-learn | Pipeline, modelos, métricas |
| XGBoost | Gradient boosting otimizado |
| Pandas / NumPy | Feature engineering |
| Matplotlib / Seaborn | Visualizações |

---

## 👤 Autor

**Marcos Brito** — Consultor 360° | Ciência de Dados  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-marcos--brito-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/marcosrpbrito)
[![GitHub](https://img.shields.io/badge/GitHub-BritoM063-181717?style=flat&logo=github)](https://github.com/BritoM063)
