"""
Gera gráficos comparativos entre Regressão Logística e Random Forest
para o modelo de risco de crédito.

Uso:
    python src/model_comparison_plot.py
"""

import matplotlib
matplotlib.use("Agg")  # permite rodar sem display

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from utils import load_data
from risk_analysis import (
    preparar_features_ml,
    criar_target,
    treinar_modelo_risco,
    _FEATURE_COLS,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)


def _preparar_dados():
    clientes, transacoes, demonstracoes, _, _ = load_data()
    feat = preparar_features_ml(clientes, transacoes, demonstracoes)
    target = criar_target(transacoes)

    df = feat.merge(target, on="id_cliente", how="left")
    df["inadimplente"] = df["inadimplente"].fillna(0).astype(int)

    feature_cols = [c for c in _FEATURE_COLS if c in df.columns]
    X = df[feature_cols].values
    y = df["inadimplente"].values
    return X, y, feature_cols


def plot_comparacao_modelos():
    clientes, transacoes, demonstracoes, _, _ = load_data()
    resultado = treinar_modelo_risco(clientes, transacoes, demonstracoes)

    X = resultado["df_features"][[c for c in _FEATURE_COLS if c in resultado["df_features"].columns]].values
    y = resultado["df_features"]["inadimplente"].values
    feature_cols = resultado["feature_names"]

    lr = resultado["logistic_regression"]
    rf = resultado["random_forest"]

    n_splits = min(5, int(y.sum()), int((len(y) - y.sum())))
    n_splits = max(n_splits, 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Probabilidades via cross_val_predict para curvas mais honestas
    lr_proba = cross_val_predict(lr, X, y, cv=cv, method="predict_proba")[:, 1]
    rf_proba = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]
    lr_pred  = cross_val_predict(lr, X, y, cv=cv)
    rf_pred  = cross_val_predict(rf, X, y, cv=cv)

    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Comparação de Modelos — Risco de Crédito", fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # 1. Curvas ROC
    ax1 = fig.add_subplot(gs[0, 0])
    for proba, label, color in [
        (lr_proba, "Regressão Logística", "#4C72B0"),
        (rf_proba, "Random Forest",       "#DD8452"),
    ]:
        fpr, tpr, _ = roc_curve(y, proba)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, lw=2, color=color, label=f"{label} (AUC = {roc_auc:.2f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.02])
    ax1.set_xlabel("Taxa de Falso Positivo")
    ax1.set_ylabel("Taxa de Verdadeiro Positivo")
    ax1.set_title("Curva ROC")
    ax1.legend(loc="lower right", fontsize=8)

    # 2. Distribuição de probabilidades
    ax2 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, 1, 20)
    ax2.hist(lr_proba[y == 0], bins=bins, alpha=0.55, color="#4C72B0", label="LR — Adimplente")
    ax2.hist(lr_proba[y == 1], bins=bins, alpha=0.55, color="#4C72B0", hatch="//", label="LR — Inadimplente")
    ax2.hist(rf_proba[y == 0], bins=bins, alpha=0.45, color="#DD8452", label="RF — Adimplente")
    ax2.hist(rf_proba[y == 1], bins=bins, alpha=0.45, color="#DD8452", hatch="//", label="RF — Inadimplente")
    ax2.set_xlabel("Probabilidade de inadimplência")
    ax2.set_ylabel("Frequência")
    ax2.set_title("Distribuição de Probabilidades")
    ax2.legend(fontsize=7)

    # 3. AUC-ROC comparativo (barra)
    ax3 = fig.add_subplot(gs[0, 2])
    modelos = ["Reg. Logística", "Random Forest"]
    aucs    = [resultado["lr_auc_cv"], resultado["rf_auc_cv"]]
    bars = ax3.bar(modelos, aucs, color=["#4C72B0", "#DD8452"], width=0.4, zorder=3)
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel("AUC-ROC (CV)")
    ax3.set_title("AUC-ROC — Validação Cruzada")
    ax3.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    for bar, val in zip(bars, aucs):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 4. Matriz de confusão — Regressão Logística
    ax4 = fig.add_subplot(gs[1, 0])
    cm_lr = confusion_matrix(y, lr_pred)
    disp = ConfusionMatrixDisplay(cm_lr, display_labels=["Adimplente", "Inadimplente"])
    disp.plot(ax=ax4, colorbar=False, cmap="Blues")
    ax4.set_title("Conf. Matrix — Reg. Logística")

    # 5. Matriz de confusão — Random Forest
    ax5 = fig.add_subplot(gs[1, 1])
    cm_rf = confusion_matrix(y, rf_pred)
    disp2 = ConfusionMatrixDisplay(cm_rf, display_labels=["Adimplente", "Inadimplente"])
    disp2.plot(ax=ax5, colorbar=False, cmap="Oranges")
    ax5.set_title("Conf. Matrix — Random Forest")

    # 6. Importância das features (Random Forest)
    ax6 = fig.add_subplot(gs[1, 2])
    imp = resultado["feature_importances"].head(10)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(imp)))[::-1]
    ax6.barh(imp.index[::-1], imp.values[::-1], color=colors[::-1])
    ax6.set_xlabel("Importância")
    ax6.set_title("Top-10 Features (Random Forest)")
    ax6.tick_params(axis="y", labelsize=8)

    # ------------------------------------------------------------------ #
    out_path = OUTPUT_DIR / "comparacao_modelos_risco.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico salvo em: {out_path}")
    return out_path


if __name__ == "__main__":
    plot_comparacao_modelos()
