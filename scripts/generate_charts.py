# usage: python generate_charts.py
# Generates publication-ready PNG charts from the XGBoost model results.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "reports", "charts")
os.makedirs(OUT_DIR, exist_ok=True)

# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":          10,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "axes.linewidth":     0.6,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "legend.fontsize":    9,
    "legend.framealpha":  1.0,
    "legend.edgecolor":   "#cccccc",
    "legend.fancybox":    False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.2,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
})

# muted, print-safe palette
C_BLUE   = "#2b6ca3"
C_GREEN  = "#2a9d4e"
C_ORANGE = "#d98c21"
C_RED    = "#c0392b"
C_PURPLE = "#6c8ebf"
C_GREY   = "#7f8c8d"

COLOR_FREE  = C_GREEN
COLOR_SLOW  = C_ORANGE
COLOR_CONG  = C_RED
STATE_COLORS = [COLOR_FREE, COLOR_SLOW, COLOR_CONG]
STATES = ["Free flow", "Slow", "Congested"]


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Traffic state distribution
# ═══════════════════════════════════════════════════════════════════════════════
def chart_state_distribution():
    counts = np.array([6_754_694, 4_519_484, 4_082_462])
    pcts   = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(5, 3.2))
    bars = ax.bar(STATES, counts / 1e6, color=STATE_COLORS, edgecolor="none", width=0.55)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
                f"{pct:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_ylabel("Observations (millions)")
    ax.set_title("Traffic State Distribution")
    ax.set_ylim(0, counts.max() / 1e6 * 1.18)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    fig.tight_layout()
    save(fig, "01_state_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Per-day accuracy & F1 scores
# ═══════════════════════════════════════════════════════════════════════════════
def chart_per_day():
    days   = ["Sun\n08-25", "Mon\n08-26", "Tue\n08-27", "Wed\n08-28",
              "Thu\n08-29", "Fri\n08-30", "Sat\n08-31"]
    acc    = [0.7035, 0.7474, 0.7350, 0.7062, 0.7191, 0.7355, 0.7425]
    f1w    = [0.6949, 0.7474, 0.7309, 0.6974, 0.7114, 0.7304, 0.7436]
    f1m    = [0.6616, 0.7303, 0.7144, 0.6804, 0.6933, 0.7125, 0.7197]
    mae    = [0.1771, 0.1380, 0.1433, 0.1504, 0.1458, 0.1437, 0.1503]

    x = np.arange(len(days))
    w = 0.19

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - 1.5*w, acc, w, label="Accuracy",    color=C_BLUE,   edgecolor="none")
    ax.bar(x - 0.5*w, f1w, w, label="F1 weighted", color=C_GREEN,  edgecolor="none")
    ax.bar(x + 0.5*w, f1m, w, label="F1 macro",    color=C_PURPLE, edgecolor="none")
    ax.bar(x + 1.5*w, mae, w, label="MAE (ratio)", color=C_RED,    edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(days)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.92)
    ax.set_title("Model Performance by Day (Leave-One-Day-Out)")
    ax.axhline(y=0.7270, color=C_BLUE, linestyle="--", linewidth=0.7, alpha=0.5)
    ax.text(6.6, 0.7270 + 0.005, "avg 72.7%", fontsize=7, color=C_BLUE)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.22)
    save(fig, "02_per_day_performance.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Confusion matrices (absolute + normalized, side by side)
# ═══════════════════════════════════════════════════════════════════════════════
def chart_confusion():
    cm = np.array([
        [6_122_193,  600_380,   32_121],
        [1_561_509, 2_933_002,  24_973],
        [  185_834, 1_787_449, 2_109_179],
    ])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    cm_m = cm / 1e6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                    gridspec_kw={"wspace": 0.45})

    for ax, data_fn, title in [
        (ax1, lambda i, j: f"{cm_m[i,j]:.2f}M", "Absolute (millions)"),
        (ax2, lambda i, j: f"{cm_norm[i,j]:.1f}%", "Normalized (%)"),
    ]:
        im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=100, aspect="equal")
        for i in range(3):
            for j in range(3):
                color = "white" if cm_norm[i, j] > 55 else "#222222"
                ax.text(j, i, data_fn(i, j), ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)
        ax.set_xticks(range(3))
        ax.set_xticklabels(STATES, rotation=25, ha="right")
        ax.set_yticks(range(3))
        ax.set_yticklabels(STATES)
        ax.set_xlabel("Predicted")
        ax.set_title(title)
        ax.tick_params(length=0)

    ax1.set_ylabel("Actual")
    fig.suptitle("Confusion Matrix", fontsize=13, fontweight="bold", y=1.01)
    cbar = fig.colorbar(im, ax=[ax1, ax2], shrink=0.75, pad=0.04)
    cbar.set_label("Recall %", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    save(fig, "03_confusion_matrix.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Feature importance (horizontal bar)
# ═══════════════════════════════════════════════════════════════════════════════
def chart_feature_importance():
    features = [
        ("hist_seg_hour",            0.2620),
        ("seg_hour_vs_city_ratio",   0.2392),
        ("sample_size",              0.1931),
        ("seg_vs_global_hour",       0.1441),
        ("seg_hour_q25",             0.0518),
        ("hist_seg_dow",             0.0243),
        ("seg_hour_cong_rate",       0.0226),
        ("hist_seg_hour_std",        0.0083),
        ("seg_hour_min",             0.0064),
        ("global_dow_hour_mean",     0.0049),
        ("speed_limit",              0.0047),
        ("is_weekend",               0.0038),
        ("segment_std_speed",        0.0038),
        ("segment_slow_rate",        0.0035),
        ("road_capacity",            0.0032),
        ("seg_vs_global_dow",        0.0032),
        ("is_peak",                  0.0030),
        ("hist_seg_weekend",         0.0027),
        ("day_of_week_num",          0.0025),
        ("segment_congestion_rate",  0.0023),
        ("time_slot",                0.0017),
        ("segment_mean_speed",       0.0017),
        ("global_hour_mean",         0.0016),
        ("time_cos",                 0.0015),
        ("segment_q10_speed",        0.0013),
        ("frc",                      0.0012),
        ("time_sin",                 0.0006),
        ("segment_min_speed",        0.0005),
        ("distance",                 0.0004),
    ]
    names = [f[0] for f in reversed(features)]
    vals  = [f[1] for f in reversed(features)]

    colors = [C_BLUE if v >= 0.05 else "#b0bec5" for v in vals]

    fig, ax = plt.subplots(figsize=(7, 7.5))
    ax.barh(names, vals, color=colors, edgecolor="none", height=0.65)
    for i, v in enumerate(vals):
        ax.text(v + 0.004, i, f"{v:.1%}", va="center", fontsize=7.5)
    ax.set_xlabel("Relative Importance")
    ax.set_title("Feature Importance (XGBoost Regressor)")
    ax.set_xlim(0, max(vals) * 1.15)
    fig.tight_layout()
    save(fig, "04_feature_importance.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Accuracy by hour of day
# ═══════════════════════════════════════════════════════════════════════════════
def chart_accuracy_by_hour():
    hours = list(range(24))
    acc = [0.7813, 0.7876, 0.7914, 0.7995, 0.8079, 0.7928, 0.7611, 0.7138,
           0.6827, 0.6847, 0.6927, 0.6896, 0.6892, 0.6912, 0.6843, 0.6793,
           0.6633, 0.6519, 0.6738, 0.7167, 0.7261, 0.7470, 0.7618, 0.7785]

    fig, ax = plt.subplots(figsize=(8, 4))

    # find best / worst indices
    acc_arr = np.array(acc)
    best_idx = int(acc_arr.argmax())
    worst_idx = int(acc_arr.argmin())

    # line + dots
    ax.plot(hours, acc, "-o", color=C_BLUE, linewidth=1.8, markersize=4.5,
            markerfacecolor=C_BLUE, markeredgecolor="white", markeredgewidth=0.6,
            zorder=3)

    # emphasize best (green)
    ax.plot(hours[best_idx], acc[best_idx], "o", color=C_GREEN, markersize=9,
            markeredgecolor="white", markeredgewidth=1.2, zorder=4)
    ax.annotate(f"Best {acc[best_idx]:.1%}\n({hours[best_idx]:02d}:00)",
                xy=(hours[best_idx], acc[best_idx]),
                xytext=(hours[best_idx] - 2.5, acc[best_idx] + 0.06),
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=0.8),
                fontsize=8, fontweight="bold", color=C_GREEN, ha="center")

    # emphasize worst (red)
    ax.plot(hours[worst_idx], acc[worst_idx], "o", color=C_RED, markersize=9,
            markeredgecolor="white", markeredgewidth=1.2, zorder=4)
    ax.annotate(f"Worst {acc[worst_idx]:.1%}\n({hours[worst_idx]:02d}:00)",
                xy=(hours[worst_idx], acc[worst_idx]),
                xytext=(hours[worst_idx] + 2.5, acc[worst_idx] - 0.07),
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.8),
                fontsize=8, fontweight="bold", color=C_RED, ha="center")

    # rush-hour shading
    for start, end in [(7, 9), (16, 18)]:
        ax.axvspan(start - 0.5, end + 0.5, alpha=0.06, color=C_RED, zorder=0)
    ax.annotate("Morning rush", xy=(8, 0.04), fontsize=7.5,
                color=C_RED, fontstyle="italic", ha="center")
    ax.annotate("Evening rush", xy=(17, 0.04), fontsize=7.5,
                color=C_RED, fontstyle="italic", ha="center")

    ax.axhline(y=0.7270, color=C_GREY, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.text(23.8, 0.7270 + 0.005, "avg 72.7%", fontsize=7, color=C_GREY, ha="right")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification Accuracy by Hour")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}" for h in hours], fontsize=7.5)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    fig.tight_layout()
    save(fig, "05_accuracy_by_hour.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Weekday vs Weekend comparison
# ═══════════════════════════════════════════════════════════════════════════════
def chart_weekday_weekend():
    labels = ["Weekday", "Weekend"]
    acc    = [0.7287, 0.7227]
    f1m    = [0.7068, 0.6924]

    x = np.arange(len(labels))
    w = 0.28

    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    b1 = ax.bar(x - w/2, acc, w, label="Accuracy", color=C_BLUE,   edgecolor="none")
    b2 = ax.bar(x + w/2, f1m, w, label="F1 macro", color=C_PURPLE, edgecolor="none")

    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f"{bar.get_height():.1%}", ha="center", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.92)
    ax.set_title("Weekday vs Weekend")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.subplots_adjust(bottom=0.2)
    save(fig, "06_weekday_weekend.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Per-class precision / recall / F1 summary
# ═══════════════════════════════════════════════════════════════════════════════
def chart_per_class():
    prec   = [0.7780, 0.5512, 0.9736]
    rec    = [0.9064, 0.6490, 0.5166]
    f1     = [0.8373, 0.5961, 0.6751]

    x = np.arange(3)
    w = 0.25

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(x - w, prec, w, label="Precision", color=C_GREEN,  edgecolor="none")
    ax.bar(x,     rec,  w, label="Recall",    color=C_ORANGE, edgecolor="none")
    ax.bar(x + w, f1,   w, label="F1-score",  color=C_BLUE,   edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(STATES)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-Class Precision, Recall, and F1-Score")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    fig.subplots_adjust(bottom=0.2)
    save(fig, "07_per_class_metrics.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Per-segment accuracy distribution (histogram)
# ═══════════════════════════════════════════════════════════════════════════════
def chart_segment_accuracy_hist():
    import pandas as pd
    seg = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data",
                                    "dataframe_segment_accuracy.csv"))
    acc = seg["accuracy"].values

    fig, ax1 = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 51)
    counts, edges, patches = ax1.hist(acc, bins=bins, color=C_BLUE, edgecolor="white",
                                       linewidth=0.3, alpha=0.85)
    ax1.set_xlabel("Segment Accuracy")
    ax1.set_ylabel("Number of Segments")
    ax1.set_title("Per-Segment Accuracy Distribution")

    median = np.median(acc)
    ax1.axvline(median, color=C_RED, linestyle="--", linewidth=1.2)
    ax1.text(median + 0.02, ax1.get_ylim()[1] * 0.92,
             f"Median {median:.1%}", color=C_RED, fontsize=9, fontweight="bold")

    ax1.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    fig.tight_layout()
    save(fig, "08_segment_accuracy_hist.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Actual vs Predicted class distribution
# ═══════════════════════════════════════════════════════════════════════════════
def chart_actual_vs_predicted():
    cm = np.array([
        [6_122_193,  600_380,   32_121],
        [1_561_509, 2_933_002,  24_973],
        [  185_834, 1_787_449, 2_109_179],
    ])
    actual    = cm.sum(axis=1)
    predicted = cm.sum(axis=0)

    x = np.arange(3)
    w = 0.32

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    b1 = ax.bar(x - w/2, actual / 1e6,    w, label="Actual",    color=C_BLUE,   edgecolor="none")
    b2 = ax.bar(x + w/2, predicted / 1e6, w, label="Predicted", color=C_ORANGE, edgecolor="none")

    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                    f"{bar.get_height():.2f}M", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(STATES)
    ax.set_ylabel("Observations (millions)")
    ax.set_title("Actual vs Predicted Class Distribution")
    ax.set_ylim(0, max(actual.max(), predicted.max()) / 1e6 * 1.18)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.subplots_adjust(bottom=0.2)
    save(fig, "09_actual_vs_predicted.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Misclassification breakdown (stacked bars)
# ═══════════════════════════════════════════════════════════════════════════════
def chart_misclassification():
    cm_norm = np.array([
        [90.6,  8.9, 0.5],
        [34.6, 64.9, 0.6],
        [ 4.6, 43.8, 51.7],
    ])

    x = np.arange(3)
    w = 0.5
    pred_labels = ["Free flow", "Slow", "Congested"]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    bottom = np.zeros(3)
    for j, (label, color) in enumerate(zip(pred_labels, STATE_COLORS)):
        vals = cm_norm[:, j]
        ax.bar(x, vals, w, bottom=bottom, label=f"Pred: {label}", color=color,
               edgecolor="white", linewidth=0.4)
        for i in range(3):
            if vals[i] >= 3:
                ax.text(x[i], bottom[i] + vals[i] / 2, f"{vals[i]:.1f}%",
                        ha="center", va="center", fontsize=8, fontweight="bold",
                        color="white" if vals[i] > 15 else "#333333")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"Actual:\n{s}" for s in STATES])
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Classification Breakdown by Actual Class")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    fig.subplots_adjust(bottom=0.22)
    save(fig, "10_misclassification_breakdown.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Class weights
# ═══════════════════════════════════════════════════════════════════════════════
def chart_class_weights():
    weights = [0.758, 1.133, 1.254]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    bars = ax.bar(STATES, weights, color=STATE_COLORS, edgecolor="none", width=0.5)
    for bar, wt in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{wt:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.axhline(y=1.0, color=C_GREY, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(2.35, 1.01, "balanced = 1.0", fontsize=7.5, color=C_GREY)
    ax.set_ylabel("Weight")
    ax.set_title("Training Class Weights")
    ax.set_ylim(0, max(weights) * 1.22)
    fig.tight_layout()
    save(fig, "11_class_weights.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Per-day MAE comparison
# ═══════════════════════════════════════════════════════════════════════════════
def chart_per_day_mae():
    days = ["Sun\n08-25", "Mon\n08-26", "Tue\n08-27", "Wed\n08-28",
            "Thu\n08-29", "Fri\n08-30", "Sat\n08-31"]
    mae  = [0.1771, 0.1380, 0.1433, 0.1504, 0.1458, 0.1437, 0.1503]
    avg_mae = 0.1499

    colors = [C_BLUE if m <= avg_mae else C_RED for m in mae]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(days, mae, color=colors, edgecolor="none", width=0.55)
    for bar, m in zip(bars, mae):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{m:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.axhline(y=avg_mae, color=C_GREY, linestyle="--", linewidth=0.9, alpha=0.6)
    ax.text(6.5, avg_mae + 0.002, f"avg {avg_mae:.4f}", fontsize=7.5, color=C_GREY,
            ha="right")

    ax.set_ylabel("MAE (speed ratio)")
    ax.set_title("Mean Absolute Error by Day")
    ax.set_ylim(0, max(mae) * 1.2)
    fig.tight_layout()
    save(fig, "12_per_day_mae.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Cumulative segment accuracy (CDF)
# ═══════════════════════════════════════════════════════════════════════════════
def chart_segment_accuracy_cdf():
    import pandas as pd
    seg = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data",
                                    "dataframe_segment_accuracy.csv"))
    acc = np.sort(seg["accuracy"].values)
    cdf = np.arange(1, len(acc) + 1) / len(acc)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(acc, cdf, color=C_BLUE, linewidth=1.8)
    ax.fill_between(acc, cdf, alpha=0.08, color=C_BLUE)

    # annotation lines at key thresholds
    for threshold, label_x_off in [(0.50, 0.03), (0.70, 0.03)]:
        pct_above = (acc >= threshold).sum() / len(acc)
        ax.axvline(threshold, color=C_GREY, linestyle=":", linewidth=0.7, alpha=0.6)
        ax.plot(threshold, 1 - pct_above, "o", color=C_RED, markersize=6, zorder=4)
        ax.annotate(f"{pct_above:.1%} of segments\n≥ {threshold:.0%} accuracy",
                    xy=(threshold, 1 - pct_above),
                    xytext=(threshold + label_x_off, 1 - pct_above - 0.12),
                    arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.8),
                    fontsize=8, color=C_RED, fontweight="bold")

    median = np.median(acc)
    ax.axvline(median, color=C_GREEN, linestyle="--", linewidth=1.0)
    ax.text(median + 0.01, 0.05, f"Median {median:.1%}", fontsize=8,
            color=C_GREEN, fontweight="bold")

    ax.set_xlabel("Segment Accuracy")
    ax.set_ylabel("Cumulative Proportion")
    ax.set_title("Cumulative Distribution of Per-Segment Accuracy")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    fig.tight_layout()
    save(fig, "13_segment_accuracy_cdf.png")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating charts...")
    chart_state_distribution()
    chart_per_day()
    chart_confusion()
    chart_feature_importance()
    chart_accuracy_by_hour()
    chart_weekday_weekend()
    chart_per_class()
    chart_segment_accuracy_hist()
    chart_actual_vs_predicted()
    chart_misclassification()
    chart_class_weights()
    chart_per_day_mae()
    chart_segment_accuracy_cdf()
    print(f"\nDone — {len(os.listdir(OUT_DIR))} charts saved to {OUT_DIR}")
