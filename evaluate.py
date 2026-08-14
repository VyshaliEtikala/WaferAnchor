"""
Evaluate the inference script algorithm against the generated dataset.

Usage:
  python evaluate.py --data_dir data --infer_cmd "python infer.py --ref {ref} --search {search}"
"""
import os
import json
import argparse
import subprocess
import numpy as np
from math import hypot

def run_infer(ref_path, search_path):
    # call local infer function directly to avoid shell overhead
    import infer as infmod
    cx, cy = infmod.detect_center(__import__("cv2").imread(ref_path, 0), __import__("cv2").imread(search_path, 0))
    return cx, cy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--threshold_px", type=float, default=10.0)
    args = parser.parse_args()
    meta_path = os.path.join(args.data_dir, "metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    errors = []
    results = []
    for m in meta:
        ref = os.path.join(args.data_dir, m["ref"])
        search = os.path.join(args.data_dir, m["search"])
        gt = (m["center"]["x"], m["center"]["y"])
        cx, cy = run_infer(ref, search)
        err = hypot(cx - gt[0], cy - gt[1])
        ok = err <= args.threshold_px
        errors.append(err)
        results.append({"ref":m["ref"], "search":m["search"], "gt":gt, "pred":(cx,cy), "err":err, "ok":ok})
        print(f"{m['ref']} | gt={gt} pred=({cx},{cy}) err={err:.2f} ok={ok}")
    errors = np.array(errors)
    print("Summary:")
    print(f"Mean error: {errors.mean():.2f} px")
    print(f"Median error: {np.median(errors):.2f} px")
    print(f"Success rate (<=10px): {(errors <= 10).mean()*100:.1f}%")
    # save detailed results
    with open(os.path.join(args.data_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()