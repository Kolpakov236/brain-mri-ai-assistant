"""Reproducible comparison; no claims of boundary accuracy without reference masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import CFG, CLASSES
from .model import load_checkpoint
from .explain import Explainer
from .review import analyze_review, draft_report, mask_stats, overlay


def main(n=40):
    out = CFG.reports_dir / "stage2"
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(CFG.reports_dir / "splits.csv")
    test = d[d.split == "test"].reset_index(drop=True)
    p = np.load(CFG.reports_dir / "test_probs.npy")
    y = test.y.to_numpy(); pred = p.argmax(1); low = p.max(1) < .7
    # Include all 17 historical errors, then a balanced deterministic sample.
    errors = np.flatnonzero(pred != y).tolist()
    rng = np.random.default_rng(2050)
    selected = errors[:n]
    while len(selected) < min(n, len(test)):
        cls = len(selected) % 4
        choices = [int(i) for i in np.flatnonzero(y == cls) if i not in selected]
        selected.append(int(rng.choice(choices)))
    model, meta = load_checkpoint(CFG.checkpoint, "cpu")
    explainer = Explainer(model, meta["arch"], "cpu")
    rows = []
    for i in selected:
        row = test.iloc[i]
        case = analyze_review(str(CFG.data_dir.parent / row.path) if not Path(row.path).exists() else row.path, explainer)
        s = mask_stats(case["mask"])
        text = draft_report(case)
        consistent = (f"площадь выделения {s['area_px']} пикселей" in text) if s["bbox"] else ("не выделена" in text or "не выявил" in text)
        record = {"test_index": int(i), "true": row.label, "pred": CLASSES[case["pred"]],
                  "correct": case["pred"] == int(row.y), "confidence": case["confidence"],
                  "agreement": case["agreement"], "low_confidence": case["confidence"] < .7,
                  "review_flag": case["confidence"] < .7 or case["agreement"] < 1,
                  "baseline_area_px": int(case["baseline_mask"].sum()), "refined_area_px": s["area_px"],
                  "uncertain_area_px": int(case["uncertain"].sum()), "text_matches_region": bool(consistent),
                  "elapsed_s": case["elapsed_s"]}
        rows.append(record)
        if len(rows) <= 3 or (case["pred"] == row.y and not any(r.get("demo_correct") for r in rows[:-1])):
            stem = f"case_{i}"
            Image.fromarray(overlay(case["rgb"], case["baseline_mask"], np.zeros_like(case["mask"]))).save(out / f"{stem}_before.png")
            Image.fromarray(overlay(case["rgb"], case["mask"], case["uncertain"])).save(out / f"{stem}_after.png")
            (out / f"{stem}_draft.txt").write_text(text, encoding="utf-8")
            record["demo_correct"] = bool(record["correct"])
        print(f"{len(rows)}/{len(selected)} index={i} correct={record['correct']} agree={case['agreement']:.1f}", flush=True)
    summary = {"sample_n": len(rows), "selection": "all 17 historical errors plus deterministic balanced additions; not representative",
               "full_test_n": len(y), "full_test_accuracy": float((pred == y).mean()),
               "full_test_low_confidence_n": int(low.sum()), "full_test_errors": int((pred != y).sum()),
               "full_test_errors_flagged": int(((pred != y) & low).sum()),
               "sample_errors": sum(not r["correct"] for r in rows),
               "sample_errors_flagged_before": sum(not r["correct"] and r["low_confidence"] for r in rows),
               "sample_errors_flagged_after": sum(not r["correct"] and r["review_flag"] for r in rows),
               "text_matches_region_n": sum(r["text_matches_region"] for r in rows),
               "text_consistency_checks": "presence of region and pixel area; not medical validity",
               "median_latency_s": float(np.median([r["elapsed_s"] for r in rows])),
               "classification_changed": False, "dice_iou": None,
               "limitations": ["No expert masks: segmentation improvement not established.",
                               "Perturbation agreement is a heuristic, not calibrated clinical confidence.",
                               "Error-enriched sample is unsuitable for overall accuracy estimates."], "cases": rows}
    (out / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40)
    main(ap.parse_args().n)
