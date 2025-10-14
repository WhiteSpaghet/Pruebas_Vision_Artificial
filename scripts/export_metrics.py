#!/usr/bin/env python3
"""
export_metrics.py

Lee los artefactos en `experiments/` (history_cnn.json, hyperparam_trials.json,
results_ensemble.csv, params_used.json) y exporta:

- experiments/exported_metrics.json  -> JSON con resumen consolidado
- experiments/exported_metrics.csv   -> CSV con filas por "metric" (clave, valor) para consumo rápido

Opciones CLI:
    --experiments-dir DIR   Carpeta experiments (default: ./experiments)
    --out-dir DIR           Carpeta salida (default: same experiments folder)
    --force                 Sobrescribir salidas si existen
    --verbose               Muestra logs DEBUG

Ejemplo:
    python scripts/export_metrics.py --experiments-dir experiments --verbose

Requisitos:
    pandas, numpy (están en requirements.txt del proyecto)
"""
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


DEFAULT_FILES = {
    "history": "history_cnn.json",
    "trials": "hyperparam_trials.json",
    "ensemble": "results_ensemble.csv",
    "params": "params_used.json"
}


def setup_logging(verbose=False):
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=lvl, format="[%(levelname)s] %(message)s")


def load_json(path: Path):
    if not path.exists():
        logging.warning("No existe: %s", path)
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.exception("Error leyendo JSON %s: %s", path, e)
        return None


def load_csv(path: Path):
    if not path.exists():
        logging.warning("No existe: %s", path)
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        logging.exception("Error leyendo CSV %s: %s", path, e)
        return None


def summarize_history(history_obj):
    """
    history_obj expected format similar to earlier examples:
    { "model_name":..., "epochs": N, "history": {"loss":[...], "val_accuracy":[...] ...}, ...}
    """
    if not history_obj:
        return {}
    hist = history_obj.get("history", {})
    out = {}
    out["model_name"] = history_obj.get("model_name") or history_obj.get("model_name", "unknown")
    out["created_at"] = history_obj.get("created_at") or history_obj.get("timestamp") or history_obj.get("created_at") or None
    out["epochs_recorded"] = history_obj.get("epochs") or (len(hist.get("loss", [])) if "loss" in hist else 0)
    # last metrics
    def last_or_none(k):
        arr = hist.get(k)
        if arr and isinstance(arr, (list, tuple)) and len(arr) > 0:
            return arr[-1]
        return None
    out["train_loss_last"] = float(last_or_none("loss")) if last_or_none("loss") is not None else None
    out["train_acc_last"] = float(last_or_none("accuracy") or last_or_none("acc")) if (last_or_none("accuracy") or last_or_none("acc")) is not None else None
    out["val_loss_last"] = float(last_or_none("val_loss")) if last_or_none("val_loss") is not None else None
    out["val_acc_last"] = float(last_or_none("val_accuracy") or last_or_none("val_acc")) if (last_or_none("val_accuracy") or last_or_none("val_acc")) is not None else None

    # best metrics
    try:
        val_accs = hist.get("val_accuracy") or hist.get("val_acc") or []
        if val_accs:
            best_idx = int(np.nanargmax(np.array(val_accs)))
            out["val_accuracy_best"] = float(val_accs[best_idx])
            out["val_accuracy_best_epoch"] = int(best_idx) + 1
        else:
            out["val_accuracy_best"] = None
            out["val_accuracy_best_epoch"] = None
    except Exception:
        out["val_accuracy_best"] = None
        out["val_accuracy_best_epoch"] = None

    return out


def summarize_trials(trials_obj):
    """
    trials_obj expected: dict with 'trials' list and 'best_trial' maybe present.
    We produce summary: n_trials, n_complete, best_value, best_trial_id, best_params (top-1)
    """
    if not trials_obj:
        return {}
    trials = trials_obj.get("trials", [])
    n = len(trials)
    n_complete = sum(1 for t in trials if str(t.get("status", t.get("state", "") or "")).upper() in ("COMPLETE", "COMPLETED", "SUCCESS"))
    best = trials_obj.get("best_trial")
    if not best and trials:
        # find best by value (non-null)
        eligible = [t for t in trials if t.get("value") is not None]
        if eligible:
            best = max(eligible, key=lambda x: float(x.get("value", -np.inf)))
    summary = {
        "study_name": trials_obj.get("study_name"),
        "objective": trials_obj.get("objective"),
        "created_at": trials_obj.get("created_at"),
        "n_trials": n,
        "n_complete": n_complete,
    }
    if best:
        summary.update({
            "best_trial_id": int(best.get("trial_id", best.get("trial_number", best.get("number", -1)))),
            "best_value": float(best.get("value")) if best.get("value") is not None else None,
            "best_params": best.get("params")
        })
    else:
        summary.update({
            "best_trial_id": None,
            "best_value": None,
            "best_params": None
        })
    return summary


def summarize_ensemble(df):
    """
    df: results_ensemble.csv as pandas DataFrame
    Returns best model by val_accuracy and test_accuracy if present
    """
    if df is None or df.empty:
        return {}
    out = {}
    # normalize column names
    cols = [c.lower() for c in df.columns]
    # find val_accuracy column name variants
    for target in ("val_accuracy", "val_acc", "validation_accuracy"):
        if target in cols:
            val_col = df.columns[cols.index(target)]
            break
    else:
        val_col = None
    for target in ("test_accuracy", "test_acc"):
        if target in cols:
            test_col = df.columns[cols.index(target)]
            break
    else:
        test_col = None

    # best by val
    try:
        if val_col:
            best_idx = df[val_col].astype(float).idxmax()
            best_row = df.loc[best_idx].to_dict()
            out["best_by_val"] = { "model": best_row.get("model_name") or best_row.get("model"), "val_accuracy": float(best_row.get(val_col)) }
        else:
            out["best_by_val"] = None
    except Exception:
        out["best_by_val"] = None

    # best by test
    try:
        if test_col:
            best_idx = df[test_col].astype(float).idxmax()
            best_row = df.loc[best_idx].to_dict()
            out["best_by_test"] = { "model": best_row.get("model_name") or best_row.get("model"), "test_accuracy": float(best_row.get(test_col)) }
        else:
            out["best_by_test"] = None
    except Exception:
        out["best_by_test"] = None

    # top-N summary
    try:
        if val_col:
            topk = df.sort_values(val_col, ascending=False).head(5)
            out["top5_by_val"] = topk[["model_name", val_col]].to_dict(orient="records")
        else:
            out["top5_by_val"] = None
    except Exception:
        out["top5_by_val"] = None

    return out


def flatten_dict_to_rows(d: dict):
    """
    Flatten a nested dict into list of (key_path, value) rows for CSV.
    E.g. {"a":{"b":1}, "c":2} => [("a.b",1), ("c",2)]
    """
    rows = []

    def recurse(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                recurse(f"{prefix}.{k}" if prefix else k, v)
        else:
            # convert lists to JSON strings
            if isinstance(obj, (list, tuple)):
                val = json.dumps(obj, ensure_ascii=False)
            else:
                val = obj
            rows.append({"metric": prefix, "value": val})
    recurse("", d)
    return rows


def main():
    parser = argparse.ArgumentParser(prog="export_metrics.py", description="Export consolidated metrics from experiments/")
    parser.add_argument("--experiments-dir", "-e", default="experiments", help="Directory with experiments artifacts")
    parser.add_argument("--out-dir", "-o", default=None, help="Output directory (defaults to experiments dir)")
    parser.add_argument("--force", action="store_true", help="Overwrite outputs if exist")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    exp_dir = Path(args.experiments_dir)
    if not exp_dir.exists():
        logging.error("Experiments dir no existe: %s", exp_dir)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else exp_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # load files
    history_path = exp_dir / DEFAULT_FILES["history"]
    trials_path = exp_dir / DEFAULT_FILES["trials"]
    ensemble_path = exp_dir / DEFAULT_FILES["ensemble"]
    params_path = exp_dir / DEFAULT_FILES["params"]

    history_obj = load_json(history_path)
    trials_obj = load_json(trials_path)
    params_obj = load_json(params_path)
    ensemble_df = load_csv(ensemble_path)

    # produce summaries
    history_summary = summarize_history(history_obj)
    trials_summary = summarize_trials(trials_obj)
    ensemble_summary = summarize_ensemble(ensemble_df)

    # combined report
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "history_summary": history_summary,
        "trials_summary": trials_summary,
        "ensemble_summary": ensemble_summary,
        "params_used": params_obj or {},
        "raw_files": {
            "history_file": str(history_path) if history_path.exists() else None,
            "trials_file": str(trials_path) if trials_path.exists() else None,
            "ensemble_file": str(ensemble_path) if ensemble_path.exists() else None,
            "params_file": str(params_path) if params_path.exists() else None
        }
    }

    # Write JSON output
    out_json = out_dir / "exported_metrics.json"
    if out_json.exists() and not args.force:
        logging.info("%s already exists. Use --force to overwrite", out_json)
    else:
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logging.info("Wrote %s", out_json)

    # Write CSV (flatten the report)
    out_csv = out_dir / "exported_metrics.csv"
    rows = []
    # top-level generated_at
    rows.append({"metric": "generated_at", "value": report["generated_at"]})
    # history flat
    rows.extend(flatten_dict_to_rows({"history": history_summary}))
    # trials flat
    rows.extend(flatten_dict_to_rows({"trials": trials_summary}))
    # ensemble flat
    rows.extend(flatten_dict_to_rows({"ensemble": ensemble_summary}))
    # params used top-level keys (stringify nested)
    rows.extend(flatten_dict_to_rows({"params_used": params_obj}) if params_obj else [])
    df_out = pd.DataFrame(rows)
    if out_csv.exists() and not args.force:
        logging.info("%s already exists. Use --force to overwrite", out_csv)
    else:
        df_out.to_csv(out_csv, index=False)
        logging.info("Wrote %s (%d rows)", out_csv, len(df_out))

    logging.info("Export metrics completed. Outputs in %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
