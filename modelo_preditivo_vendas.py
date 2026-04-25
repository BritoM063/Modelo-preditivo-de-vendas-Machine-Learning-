"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PROJETO 2 — MODELO PREDITIVO DE VENDAS (MACHINE LEARNING)             ║
║  Autor: Marcos Brito | github.com/BritoM063                             ║
║  Stack: Python · Scikit-learn · XGBoost · GridSearchCV · Matplotlib     ║
╚══════════════════════════════════════════════════════════════════════════╝

CONTEXTO DE NEGÓCIO
-------------------
A previsão de vendas é um dos maiores desafios estratégicos de PMEs.
Erros de previsão geram excesso de estoque, ruptura de produtos, alocação
ineficiente de equipe de vendas e frustração de clientes.

OBJETIVO
--------
Desenvolver modelos preditivos de regressão para prever a receita mensal
por categoria de produto, comparar algoritmos e selecionar o melhor
modelo para suportar o planejamento comercial.

PIPELINE
--------
1. Feature Engineering (lags, médias móveis, sazonalidade)
2. Regressão Linear (baseline)
3. Random Forest Regressor
4. XGBoost com GridSearchCV
5. Comparativo de modelos (RMSE, MAE, R²)
6. Análise de importância de features
7. Previsão para os próximos 3 meses
"""

# ── 0. IMPORTS ───────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
import os
from datetime import timedelta

from sklearn.linear_model   import LinearRegression, Ridge
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection      import permutation_importance
import xgboost as xgb

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 130, "figure.facecolor": "white"})
os.makedirs("outputs", exist_ok=True)

print("=" * 65)
print("PROJETO 2 — MODELO PREDITIVO DE VENDAS COM MACHINE LEARNING")
print("=" * 65)


# ── 1. CARREGAMENTO E PREPARAÇÃO ─────────────────────────────────────────────
print("\n[1/7] Carregando e preparando dados...")

df = pd.read_csv("data/pedidos.csv", parse_dates=["data"])

# Agrega receita líquida mensal por categoria
mensal = (df.groupby(["ano", "mes", "categoria"])
            .agg(receita=("receita_liquida","sum"),
                 n_pedidos=("pedido_id","count"),
                 ticket_medio=("ticket_medio","mean"),
                 margem_media=("margem_pct","mean"),
                 qtd_total=("qtd_itens","sum"))
            .reset_index()
            .sort_values(["categoria","ano","mes"]))

mensal["periodo"] = pd.to_datetime(
    mensal["ano"].astype(str) + "-" + mensal["mes"].astype(str).str.zfill(2) + "-01"
)

print(f"✓ {len(mensal)} observações mensais por categoria")
print(f"  Período: {mensal['periodo'].min().date()} → {mensal['periodo'].max().date()}")
print(f"  Categorias: {mensal['categoria'].unique().tolist()}")


# ── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────
print("\n[2/7] Feature Engineering...")

def criar_features(df_cat):
    """Cria features temporais, lags e médias móveis para uma categoria."""
    d = df_cat.sort_values("periodo").copy()

    # Lags de receita (1, 2, 3, 6 meses)
    for lag in [1, 2, 3, 6]:
        d[f"receita_lag{lag}"] = d["receita"].shift(lag)

    # Médias móveis
    d["receita_mm3"]  = d["receita"].shift(1).rolling(3).mean()
    d["receita_mm6"]  = d["receita"].shift(1).rolling(6).mean()
    d["receita_mm12"] = d["receita"].shift(1).rolling(12).mean()

    # Variação percentual mês a mês
    d["receita_delta_pct"] = d["receita"].pct_change().shift(1) * 100

    # Features sazonais
    d["mes"]          = d["periodo"].dt.month
    d["trimestre"]    = d["periodo"].dt.quarter
    d["mes_sin"]      = np.sin(2 * np.pi * d["mes"] / 12)   # componente cíclico
    d["mes_cos"]      = np.cos(2 * np.pi * d["mes"] / 12)
    d["is_q4"]        = (d["trimestre"] == 4).astype(int)
    d["is_novdez"]    = d["mes"].isin([11, 12]).astype(int)
    d["mes_do_ano"]   = d["mes"]

    # Lags de outras features
    d["ticket_lag1"]  = d["ticket_medio"].shift(1)
    d["margem_lag1"]  = d["margem_media"].shift(1)
    d["n_pedidos_lag1"] = d["n_pedidos"].shift(1)

    return d.dropna()

dfs_feat = []
for cat, grp in mensal.groupby("categoria"):
    feat = criar_features(grp)
    feat["categoria_enc"] = hash(cat) % 100   # encoding simples
    feat["cat_label"] = cat
    dfs_feat.append(feat)

df_feat = pd.concat(dfs_feat).reset_index(drop=True)

FEATURES = [
    "receita_lag1","receita_lag2","receita_lag3","receita_lag6",
    "receita_mm3","receita_mm6","receita_mm12","receita_delta_pct",
    "mes_sin","mes_cos","is_q4","is_novdez","mes_do_ano",
    "ticket_lag1","margem_lag1","n_pedidos_lag1","categoria_enc"
]
TARGET = "receita"

print(f"✓ Features criadas: {len(FEATURES)}")
print(f"  {FEATURES}")
print(f"  Dataset final: {len(df_feat)} amostras × {len(FEATURES)} features")


# ── 3. SPLIT TEMPORAL ────────────────────────────────────────────────────────
print("\n[3/7] Split temporal treino/teste...")

CORTE = pd.Timestamp("2024-06-01")
df_train = df_feat[df_feat["periodo"] <  CORTE]
df_test  = df_feat[df_feat["periodo"] >= CORTE]

X_train = df_train[FEATURES]
y_train = df_train[TARGET]
X_test  = df_test[FEATURES]
y_test  = df_test[TARGET]

print(f"  Treino: {len(X_train)} amostras (até {CORTE.date()})")
print(f"  Teste:  {len(X_test)}  amostras (após {CORTE.date()})")
print(f"  Receita média treino: R$ {y_train.mean():,.2f}")
print(f"  Receita média teste:  R$ {y_test.mean():,.2f}")


# ── 4. MODELOS ───────────────────────────────────────────────────────────────
print("\n[4/7] Treinando modelos...")

def avaliar_modelo(nome, modelo, X_tr, y_tr, X_te, y_te):
    """Treina, prediz e retorna métricas."""
    modelo.fit(X_tr, y_tr)
    pred_tr = modelo.predict(X_tr)
    pred_te = modelo.predict(X_te)

    def metricas(y_real, y_pred, label):
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))
        mae  = mean_absolute_error(y_real, y_pred)
        r2   = r2_score(y_real, y_pred)
        mape = np.mean(np.abs((y_real - y_pred) / (y_real + 1))) * 100
        return {"set": label, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

    res = pd.DataFrame([
        metricas(y_tr, pred_tr, "Treino"),
        metricas(y_te, pred_te, "Teste"),
    ])
    print(f"\n  {'─'*45}")
    print(f"  {nome}")
    print(res[["set","RMSE","MAE","R2","MAPE"]].to_string(index=False))
    return modelo, pred_te, res

resultados = {}

# 4.1 Baseline — Regressão Linear com Ridge
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  Ridge(alpha=10))
])
m_lr, pred_lr, res_lr = avaliar_modelo("Ridge Regression (baseline)", pipe_lr,
                                        X_train, y_train, X_test, y_test)
resultados["Ridge"] = {"modelo": m_lr, "pred": pred_lr, "res": res_lr}

# 4.2 Random Forest
rf = RandomForestRegressor(
    n_estimators=300, max_depth=8, min_samples_leaf=2,
    max_features="sqrt", random_state=42, n_jobs=-1
)
m_rf, pred_rf, res_rf = avaliar_modelo("Random Forest Regressor", rf,
                                        X_train, y_train, X_test, y_test)
resultados["Random Forest"] = {"modelo": m_rf, "pred": pred_rf, "res": res_rf}

# 4.3 XGBoost com GridSearchCV
print(f"\n  {'─'*45}")
print("  XGBoost + GridSearchCV (pode levar ~30s)...")

tscv = TimeSeriesSplit(n_splits=4)
param_grid = {
    "n_estimators":    [200, 400],
    "max_depth":       [4, 6],
    "learning_rate":   [0.05, 0.1],
    "subsample":       [0.8, 1.0],
    "colsample_bytree":[0.8, 1.0],
}
xgb_base = xgb.XGBRegressor(
    objective="reg:squarederror", random_state=42,
    eval_metric="rmse", verbosity=0
)
grid_xgb = GridSearchCV(
    xgb_base, param_grid, cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1, verbose=0
)
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
print(f"  Melhores parâmetros: {grid_xgb.best_params_}")

m_xgb, pred_xgb, res_xgb = avaliar_modelo("XGBoost (best params)", best_xgb,
                                            X_train, y_train, X_test, y_test)
resultados["XGBoost"] = {"modelo": m_xgb, "pred": pred_xgb, "res": res_xgb}


# ── 5. COMPARATIVO DE MODELOS ────────────────────────────────────────────────
print("\n[5/7] Comparativo final de modelos...")

comparativo = []
for nome, v in resultados.items():
    linha = v["res"][v["res"]["set"] == "Teste"].copy()
    linha["Modelo"] = nome
    comparativo.append(linha)

df_comp = pd.concat(comparativo).set_index("Modelo")[["RMSE","MAE","R2","MAPE"]]
print("\n  ┌─ COMPARATIVO (conjunto de teste) ─────────────────────┐")
print(df_comp.round(2).to_string())
print("  └────────────────────────────────────────────────────────┘")

melhor_modelo_nome = df_comp["RMSE"].idxmin()
melhor_pred        = resultados[melhor_modelo_nome]["pred"]
melhor_modelo_obj  = resultados[melhor_modelo_nome]["modelo"]
print(f"\n  🏆 Melhor modelo (menor RMSE no teste): {melhor_modelo_nome}")


# ── 6. VISUALIZAÇÕES ─────────────────────────────────────────────────────────
print("\n[6/7] Gerando visualizações...")

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Análise Comparativa de Modelos Preditivos — Receita Mensal",
             fontsize=15, fontweight="bold")

cores_modelo = {"Ridge": "#888", "Random Forest": "#014034", "XGBoost": "#C8973A"}

# 6.1 Comparativo de RMSE
ax = axes[0, 0]
bars = ax.bar(df_comp.index, df_comp["RMSE"] / 1000,
              color=[cores_modelo[m] for m in df_comp.index], width=0.5)
ax.set_title("RMSE por Modelo (R$ mil) — menor é melhor", fontweight="bold")
ax.set_ylabel("RMSE (R$ mil)")
for bar, val in zip(bars, df_comp["RMSE"] / 1000):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"R${val:.1f}k", ha="center", va="bottom", fontweight="bold", fontsize=10)

# 6.2 R² por modelo
ax = axes[0, 1]
bars = ax.bar(df_comp.index, df_comp["R2"],
              color=[cores_modelo[m] for m in df_comp.index], width=0.5)
ax.axhline(0.8, color="#E74C3C", linestyle="--", label="R²=0.80 (benchmark)")
ax.set_title("R² por Modelo — maior é melhor", fontweight="bold")
ax.set_ylabel("R²")
ax.set_ylim(0, 1.05)
ax.legend()
for bar, val in zip(bars, df_comp["R2"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

# 6.3 Real vs Predito (melhor modelo)
ax = axes[1, 0]
ax.scatter(y_test, melhor_pred, alpha=0.5, color="#014034", s=30)
lim_max = max(y_test.max(), melhor_pred.max()) * 1.05
ax.plot([0, lim_max], [0, lim_max], "r--", linewidth=2, label="Predição perfeita")
ax.set_xlabel("Receita Real (R$)")
ax.set_ylabel("Receita Prevista (R$)")
ax.set_title(f"Real vs Predito — {melhor_modelo_nome}", fontweight="bold")
ax.legend()
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"R${x/1000:.0f}k"))
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"R${x/1000:.0f}k"))

# 6.4 Importância de features (XGBoost)
ax = axes[1, 1]
if hasattr(best_xgb, "feature_importances_"):
    fi = pd.Series(best_xgb.feature_importances_, index=FEATURES)
    fi = fi.sort_values(ascending=True).tail(12)
    colors_fi = ["#C8973A" if "lag" in f or "mm" in f
                 else "#014034" if "mes" in f or "q4" in f
                 else "#D9946C" for f in fi.index]
    fi.plot(kind="barh", ax=ax, color=colors_fi)
    ax.set_title("Importância de Features — XGBoost\n(top 12)", fontweight="bold")
    ax.set_xlabel("Feature Importance (gain)")

plt.tight_layout()
plt.savefig("outputs/fig5_comparativo_modelos.png", bbox_inches="tight", dpi=130)
plt.close()
print("✓ outputs/fig5_comparativo_modelos.png")

# 6.5 Série temporal: real vs predito por categoria
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
axes = axes.flatten()
cats = df_test["cat_label"].unique()

for idx, cat in enumerate(cats[:6]):
    ax = axes[idx]
    mask_tr = df_train["cat_label"] == cat
    mask_te = df_test["cat_label"]  == cat

    ax.plot(df_train.loc[mask_tr, "periodo"],
            df_train.loc[mask_tr, "receita"] / 1000,
            color="#012623", linewidth=1.5, label="Histórico")
    ax.plot(df_test.loc[mask_te, "periodo"],
            df_test.loc[mask_te, "receita"] / 1000,
            color="#014034", linewidth=2, label="Real (teste)")

    X_cat  = df_test.loc[mask_te, FEATURES]
    p_cat  = melhor_modelo_obj.predict(X_cat)
    ax.plot(df_test.loc[mask_te, "periodo"],
            p_cat / 1000,
            color="#C8973A", linewidth=2, linestyle="--", label="Previsto")

    ax.axvline(CORTE, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.set_title(cat, fontweight="bold", fontsize=10)
    ax.set_ylabel("R$ mil")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}k"))

fig.suptitle(f"Receita Real vs Prevista por Categoria — {melhor_modelo_nome}",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/fig6_serie_temporal_categorias.png", bbox_inches="tight", dpi=130)
plt.close()
print("✓ outputs/fig6_serie_temporal_categorias.png")


# ── 7. PREVISÃO FUTURA ───────────────────────────────────────────────────────
print("\n[7/7] Previsão para os próximos 3 meses...")

# Última observação por categoria → projetar jan-mar 2025
ultima_obs = df_feat.sort_values("periodo").groupby("cat_label").last().reset_index()

previsoes = []
for _, row in ultima_obs.iterrows():
    for delta in [1, 2, 3]:
        prox_periodo = row["periodo"] + pd.DateOffset(months=delta)
        mes          = prox_periodo.month
        feat_row = {
            "receita_lag1":     row["receita"],
            "receita_lag2":     row["receita_lag1"],
            "receita_lag3":     row["receita_lag2"],
            "receita_lag6":     row["receita_lag3"],
            "receita_mm3":      row["receita_mm3"],
            "receita_mm6":      row["receita_mm6"],
            "receita_mm12":     row["receita_mm12"],
            "receita_delta_pct":row["receita_delta_pct"],
            "mes_sin":          np.sin(2 * np.pi * mes / 12),
            "mes_cos":          np.cos(2 * np.pi * mes / 12),
            "is_q4":            int((mes // 3 + 1) == 4),
            "is_novdez":        int(mes in [11, 12]),
            "mes_do_ano":       mes,
            "ticket_lag1":      row["ticket_lag1"],
            "margem_lag1":      row["margem_lag1"],
            "n_pedidos_lag1":   row["n_pedidos_lag1"],
            "categoria_enc":    row["categoria_enc"],
        }
        X_fut = pd.DataFrame([feat_row])
        prev  = melhor_modelo_obj.predict(X_fut)[0]
        previsoes.append({
            "categoria":   row["cat_label"],
            "periodo":     prox_periodo.strftime("%Y-%m"),
            "previsao_R$": round(prev, 2),
        })

df_prev = pd.DataFrame(previsoes)
pivot_prev = df_prev.pivot(index="categoria", columns="periodo", values="previsao_R$")
print("\n  📅 Previsão de Receita por Categoria (R$):")
print(pivot_prev.to_string())
total_prev = df_prev.groupby("periodo")["previsao_R$"].sum()
print(f"\n  📊 Previsão de Receita Total:")
for per, val in total_prev.items():
    print(f"     {per}: R$ {val:,.2f}")

print(f"\n✅ Projeto 2 concluído. Outputs em /outputs/")
print(f"   Melhor modelo: {melhor_modelo_nome}")
print(df_comp.round(2).to_string())
