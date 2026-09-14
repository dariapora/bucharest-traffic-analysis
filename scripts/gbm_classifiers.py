import argparse
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)

from features import (
    REPORT_DIR,
    STATE_ORDER,
    FEATURE_COLS,
    LABEL_TAG,
    load_dataset,
    add_features,
    add_features_out_of_fold,
)

N_CLASSES = 3
MAX_ROUNDS = 2000
EARLY_STOPPING = 50
SEED = 42

LGB_PARAMS = dict(
    objective="multiclass",
    num_class=N_CLASSES,
    learning_rate=0.1,
    num_leaves=127,
    min_data_in_leaf=500,
    feature_fraction=0.8,
    bagging_fraction=0.7,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=255,
    num_threads=32,
    seed=SEED,
    verbose=-1,
)

XGB_PARAMS = dict(
    objective="multi:softprob",
    num_class=N_CLASSES,
    device="cuda",
    tree_method="hist",
    learning_rate=0.1,
    max_depth=10,
    min_child_weight=100,
    subsample=0.7,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    max_bin=256,
    eval_metric="mlogloss",
    seed=SEED,
)


def log(msg=""):
    print(msg, flush=True)


def balanced_sample_weights(y):
    counts = np.bincount(y, minlength=N_CLASSES)
    return (len(y) / (N_CLASSES * counts))[y]


def train_lightgbm(X_tr, y_tr, w_tr, X_va, y_va, feature_cols):
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feature_cols)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
    booster = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(100)],
    )
    best = booster.best_iteration
    importance = pd.Series(booster.feature_importance("gain"), index=feature_cols)

    def predict(X):
        return booster.predict(X, num_iteration=best).argmax(axis=1)

    return predict, best, importance


def train_xgboost(X_tr, y_tr, w_tr, X_va, y_va, feature_cols):
    dtrain = xgb.QuantileDMatrix(X_tr, label=y_tr, weight=w_tr, feature_names=feature_cols,
                                 max_bin=XGB_PARAMS["max_bin"])
    dval = xgb.QuantileDMatrix(X_va, label=y_va, feature_names=feature_cols, ref=dtrain)
    booster = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=MAX_ROUNDS,
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=100,
    )
    best = booster.best_iteration
    gain = booster.get_score(importance_type="total_gain")
    importance = pd.Series({f: gain.get(f, 0.0) for f in feature_cols})

    def predict(X):
        dm = xgb.DMatrix(X, feature_names=feature_cols)
        return booster.predict(dm, iteration_range=(0, best + 1)).argmax(axis=1)

    return predict, best, importance


TRAINERS = {
    "lightgbm": train_lightgbm,
    "xgboost": train_xgboost,
}


def build_report(name, results, day_metrics, importances, best_iters, class_counts, n_rows, elapsed, feature_cols):
    lines = []
    p = lines.append

    overall_acc = accuracy_score(results["actual"], results["predicted"])
    f1_w = f1_score(results["actual"], results["predicted"], average="weighted")
    f1_m = f1_score(results["actual"], results["predicted"], average="macro")

    p("=" * 60)
    p(f"{name.upper()} RESULTS")
    p("=" * 60)
    p(f"Overall accuracy:    {overall_acc:.4f} ({overall_acc * 100:.2f}%)")
    p(f"Weighted F1:         {f1_w:.4f}")
    p(f"Macro F1:            {f1_m:.4f}")
    p("")
    p("=" * 60)
    p("CLASSIFICATION REPORT")
    p("=" * 60)
    p(classification_report(results["actual"], results["predicted"],
                            labels=[0, 1, 2], target_names=STATE_ORDER, digits=4))

    cm = confusion_matrix(results["actual"], results["predicted"], labels=[0, 1, 2])
    p("=" * 60)
    p("CONFUSION MATRIX")
    p("Rows = actual")
    p("Columns = predicted")
    p("=" * 60)
    p(pd.DataFrame(cm, index=STATE_ORDER, columns=STATE_ORDER).to_string())
    p("")
    p("=" * 60)
    p("NORMALIZED CONFUSION MATRIX")
    p("Rows sum to 100%")
    p("=" * 60)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    p((pd.DataFrame(cm_norm, index=STATE_ORDER, columns=STATE_ORDER) * 100).round(1).to_string())
    p("")
    p("=" * 60)
    p("PER-DAY RESULTS")
    p("=" * 60)
    dm = pd.DataFrame(day_metrics)
    dm["best_iteration"] = best_iters
    p(dm.to_string(index=False))
    p("")
    p("=" * 60)
    p("FEATURE IMPORTANCE (gain, mean over folds, normalised)")
    p("=" * 60)
    imp = pd.concat(importances, axis=1).mean(axis=1)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    for feat, val in imp.items():
        p(f"{feat:>24s}: {val:.4f} {'#' * int(round(val * 100))}")
    p("")
    p(f"Total time: {elapsed / 60:.1f} min")

    body = "\n".join(lines)

    header = [
        f"# {name}",
        "",
        "Script: `scripts/gbm_classifiers.py`",
        f"Params: `{LGB_PARAMS if name.startswith('lightgbm') else XGB_PARAMS}`",
        f"Max rounds {MAX_ROUNDS}, early stopping {EARLY_STOPPING} on one held-out weekday of the training set (mlogloss), "
        + ("unweighted samples" if "unweighted" in name else "balanced sample weights"),
        "Validation: leave-one-day-out; training-set features computed out-of-fold (same as logistic regression baseline)",
        f"Features ({len(feature_cols)}): {', '.join(feature_cols)}",
        "",
        f"Loaded {n_rows:,} rows, class distribution: " +
        ", ".join(f"{s} {c:,} ({c / n_rows * 100:.1f}%)" for s, c in zip(STATE_ORDER, class_counts)),
        "",
        "```",
        body,
        "```",
        "",
    ]
    return "\n".join(header), body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(TRAINERS), choices=list(TRAINERS))
    parser.add_argument("--days", nargs="*", default=None, help="subset of test days (default: all)")
    parser.add_argument("--with-sample-size", action="store_true",
                        help="add the contemporaneous probe count of the interval as a feature")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="train on unweighted samples (maximise accuracy instead of balancing classes)")
    parser.add_argument("--both-weightings", action="store_true",
                        help="train each model with balanced and with unweighted samples in the same pass")
    args = parser.parse_args()

    feature_cols = FEATURE_COLS + (["sample_size"] if args.with_sample_size else [])
    suffix = ("_with_sample_size" if args.with_sample_size else "") + LABEL_TAG
    if args.both_weightings:
        weightings = ["balanced", "unweighted"]
    else:
        weightings = ["unweighted" if args.no_class_weights else "balanced"]
    runs = [(m, w) for m in args.models for w in weightings]

    def run_name(model, weighting):
        return model + ("_unweighted" if weighting == "unweighted" else "")

    t_start = time.time()
    df = load_dataset()
    n_rows = len(df)
    class_counts = np.bincount(df["target"], minlength=N_CLASSES)
    log(f"Loaded {n_rows:,} rows, {df['segment_id'].nunique():,} segments, {df['date'].nunique()} days")
    for s, c in zip(STATE_ORDER, class_counts):
        log(f"{s:>12s}: {c:>10,} ({c / n_rows * 100:.1f}%)")

    days = sorted(df["date"].unique())
    test_days = args.days or days
    weekend_by_day = df.groupby("date")["is_weekend"].first()

    state = {run_name(m, w): dict(preds=[], day_metrics=[], importances=[], best_iters=[], time=0.0) for m, w in runs}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for test_day in test_days:
        log(f"\n{'#' * 60}\nTesting day: {test_day}")
        t_fold = time.time()

        train_raw = df[df["date"] != test_day]
        test_raw = df[df["date"] == test_day]

        test = add_features(test_raw, train_raw)
        train = add_features_out_of_fold(train_raw)
        log(f"  features built in {time.time() - t_fold:.0f}s")

        train_days = sorted(train["date"].unique())
        val_day = max(d for d in train_days if not weekend_by_day[d])
        is_val = train["date"] == val_day

        X_tr = train.loc[~is_val, feature_cols].to_numpy(dtype=np.float32)
        y_tr = train.loc[~is_val, "target"].to_numpy()
        X_va = train.loc[is_val, feature_cols].to_numpy(dtype=np.float32)
        y_va = train.loc[is_val, "target"].to_numpy()
        X_te = test[feature_cols].to_numpy(dtype=np.float32)
        y_te = test["target"].to_numpy()
        meta_te = test[["date", "time_numeric", "segment_id"]].reset_index(drop=True)
        weights = {"balanced": balanced_sample_weights(y_tr), "unweighted": np.ones(len(y_tr), dtype=np.float32)}
        log(f"  train={len(y_tr):,}  val({val_day})={len(y_va):,}  test={len(y_te):,}")
        del train, test

        for model, weighting in runs:
            name = run_name(model, weighting)
            log(f"\n  --- {name}")
            t_model = time.time()
            predict, best, importance = TRAINERS[model](X_tr, y_tr, weights[weighting], X_va, y_va, feature_cols)
            preds = predict(X_te)
            elapsed = time.time() - t_model

            acc = accuracy_score(y_te, preds)
            f1_w = f1_score(y_te, preds, average="weighted")
            f1_m = f1_score(y_te, preds, average="macro")
            log(f"  {name}: best_iter={best}  time={elapsed:.0f}s")
            log(f"  Accuracy:    {acc:.4f}")
            log(f"  Weighted F1: {f1_w:.4f}")
            log(f"  Macro F1:    {f1_m:.4f}")

            st = state[name]
            st["preds"].append(meta_te.assign(actual=y_te, predicted=preds))
            st["day_metrics"].append(dict(day=test_day, accuracy=acc, weighted_f1=f1_w, macro_f1=f1_m))
            st["importances"].append(importance)
            st["best_iters"].append(best)
            st["time"] += elapsed

            # write a partial report after each fold so a killed run still leaves results
            results = pd.concat(st["preds"], ignore_index=True)
            report_md, _ = build_report(name + suffix, results, st["day_metrics"], st["importances"],
                                        st["best_iters"], class_counts, n_rows, time.time() - t_start, feature_cols)
            (REPORT_DIR / f"{name}{suffix}.md").write_text(report_md, encoding="utf-8")

    for name in state:
        st = state[name]
        results = pd.concat(st["preds"], ignore_index=True)
        report_md, body = build_report(name + suffix, results, st["day_metrics"], st["importances"],
                                       st["best_iters"], class_counts, n_rows, time.time() - t_start, feature_cols)
        (REPORT_DIR / f"{name}{suffix}.md").write_text(report_md, encoding="utf-8")
        results.to_parquet(REPORT_DIR / f"predictions_{name}{suffix}.parquet", index=False)
        log("\n\n" + body)
        log(f"Report saved to {REPORT_DIR / f'{name}{suffix}.md'}")


if __name__ == "__main__":
    main()
