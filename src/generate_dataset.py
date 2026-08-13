#!/usr/bin/env python3
"""SemiCoN PS2 synthetic dataset generator.
Official experiment interface:
python generate_dataset.py --architecture DRAM --num_pairs 30 --output_dir data/generated --seed 42
Each pair contains:
- search: ~1000x1000 low-magnification image
- reference: ~100x100 high-magnification crop of the same physical target
- independent noise and imaging degradation
- known ground-truth target center, bbox, scale, rotation, noise and blur
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import cv2
import numpy as np
def clip8(x):
 return np.clip(x, 0, 255).astype(np.uint8)
def rng_for(seed, i):
 return np.random.default_rng(seed + 104729 * (i + 1))
def edge_brighten(img, strength):
 f = img.astype(np.float32)
 gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
 gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
 mag = cv2.magnitude(gx, gy)
 if mag.max() > 1e-6:
 mag /= mag.max()
 return clip8(f + strength * 95.0 * mag)
def independent_capture(img, rng, noise_sigma_range, blur_sigma_range,
 edge_strength_range, contrast_range=(0.92, 1.10)):
 sigma_b = float(rng.uniform(*blur_sigma_range))
 if sigma_b > 0.05:
 k = max(3, 2 * int(math.ceil(3 * sigma_b)) + 1)
 out = cv2.GaussianBlur(img, (k, k), sigma_b)
 else:
 out = img.copy()
 out = edge_brighten(out, float(rng.uniform(*edge_strength_range)))
 h, w = out.shape
 yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
 x = (xx - w / 2) / w
 y = (yy - h / 2) / h
 gain = 1.0 + float(rng.uniform(-0.10, 0.10))*x + float(rng.uniform(-0.10, 0.10))*y
 out = clip8(out.astype(np.float32) * gain)
 alpha = float(rng.uniform(*contrast_range))
 beta = float(rng.uniform(-8, 8))
 out = clip8(out.astype(np.float32) * alpha + beta)
 # Independent sensor noise: this RNG stream is separate for every capture.
 noise_sigma = float(rng.uniform(*noise_sigma_range))
 out = clip8(out.astype(np.float32) + rng.normal(0, noise_sigma, out.shape))
 return out, noise_sigma, sigma_b
def affine_image(img, angle, scale):
 h, w = img.shape
 M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
 out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
 borderMode=cv2.BORDER_REFLECT)
 return out, M
def draw_dram(h, w, rng):
 img = np.full((h, w), 35, np.uint8)
 pitch_x = int(rng.integers(42, 58))
 pitch_y = int(rng.integers(42, 58))
 # Bit lines and word lines.
 for x in range(12, w-12, pitch_x):
 cv2.line(img, (x, 0), (x, h), int(rng.integers(95, 145)),
 int(rng.integers(2, 4))) for y in range(12, h-12, pitch_y):
 cv2.line(img, (0, y), (w, y), int(rng.integers(110, 165)),
 int(rng.integers(2, 4)))
 # Contacts / vias and small cell rectangles.
 for y in range(pitch_y//2, h, pitch_y):
 for x in range(pitch_x//2, w, pitch_x):
 r = int(rng.integers(3, 6))
 cv2.circle(img, (x, y), r, int(rng.integers(155, 215)), -1)
 if rng.random() < 0.8:
 cv2.rectangle(img, (x-7, y-4), (x+7, y+4),
 int(rng.integers(70, 120)), 1)
 return img
def draw_finfet(h, w, rng):
 img = np.full((h, w), 35, np.uint8)
 pitch = int(rng.integers(22, 32))
 # Dense parallel fins.
 for x in range(12, w-12, pitch):
 cv2.line(img, (x, 0), (x, h), int(rng.integers(125, 185)),
 int(rng.integers(2, 4)))
 # Gate bars.
 for y in range(45, h-45, int(rng.integers(90, 125))):
 cv2.line(img, (0, y), (w, y), int(rng.integers(155, 215)),
 int(rng.integers(3, 6)))
 if rng.random() < 0.5:
 cv2.line(img, (0, y+8), (w, y+8), int(rng.integers(80, 130)), 1)
 # Contacts at crossings.
 for y in range(55, h-55, 100):
 for x in range(25, w-25, pitch*3):
 cv2.circle(img, (x, y), int(rng.integers(3, 6)),
 int(rng.integers(175, 225)), -1)
 return img
def add_target(scene, center, rng):
 """Distinctive but layout-compatible alignment target, ~220-260 px wide."""
 cx, cy = center
 size = int(rng.integers(210, 251))
 half = size // 2
 out = scene.copy()
 # Main alignment cross and asymmetric corner marks.
 cv2.rectangle(out, (cx-half, cy-half), (cx+half, cy+half), 225, 4)
 arm = int(half * 0.72)
 thick = max(5, size//30)
 cv2.line(out, (cx-arm, cy), (cx+arm, cy), 240, thick)
 cv2.line(out, (cx, cy-arm), (cx, cy+arm), 240, thick)
 cv2.rectangle(out, (cx-half, cy-half), (cx-half//2, cy-half//2), 245, -1)
 cv2.rectangle(out, (cx+half//2, cy+half//2), (cx+half, cy+half), 245, -1)
 cv2.circle(out, (cx, cy), max(5, size//18), 30, -1)
 # Secondary ring/markers make the pattern less ambiguous.
 cv2.circle(out, (cx, cy), int(size*0.63), 145, 2)
 cv2.line(out, (cx-int(size*.85), cy-int(size*.35)),
 (cx-int(size*.55), cy-int(size*.35)), 205, 3)
 return out, size
def crop_ref(scene, center, crop_size):
 cx, cy = center
 p = crop_size//2 + 4
 pad = cv2.copyMakeBorder(scene, p, p, p, p, cv2.BORDER_REFLECT)
 px, py = cx+p, cy+p
 return pad[py-crop_size//2:py-crop_size//2+crop_size,
 px-crop_size//2:px-crop_size//2+crop_size].copy()
def make_pair(i, args, out_root):
 rng = rng_for(args.seed, i)
 H = W = args.canvas_size
 base = draw_dram(H, W, rng) if args.architecture == "DRAM" else draw_finfet(H, W, rng)
 # Target is well inside the die so every physical crop is valid.
 cx = int(rng.integers(400, W-400))
 cy = int(rng.integers(400, H-400))
 scene, target_size = add_target(base, (cx, cy), rng)
 # Low-mag search. The target's ~230 px physical size becomes ~96 px at 1000 px.
 search0 = cv2.resize(scene, (args.search_size, args.search_size),
 interpolation=cv2.INTER_AREA)
 sx0 = cx * args.search_size / W
 sy0 = cy * args.search_size / H
 search_angle = float(rng.uniform(-3.0, 3.0)) search_scale = float(rng.uniform(0.97, 1.03))
 search, M = affine_image(search0, search_angle, search_scale)
 pt = np.array([sx0, sy0, 1.0], np.float32)
 search_center = (M @ pt).astype(float)
 search, search_noise, search_blur = independent_capture(
 search, rng, (4.0, 9.0), (0.9, 2.2), (0.20, 0.42),
 contrast_range=(0.88, 1.08)
 )
 # High-mag reference: crop the same physical region (~240 px) then resize to 100.
 crop_physical = int(rng.integers(225, 255))
 ref0 = crop_ref(scene, (cx, cy), crop_physical)
 reference = cv2.resize(ref0, (args.reference_size, args.reference_size),
 interpolation=cv2.INTER_CUBIC)
 ref_angle = float(rng.uniform(-2.0, 2.0))
 ref_scale = float(rng.uniform(0.96, 1.04))
 reference, _ = affine_image(reference, ref_angle, ref_scale)
 reference, ref_noise, ref_blur = independent_capture(
 reference, rng, (1.0, 3.5), (0.10, 0.75), (0.22, 0.48),
 contrast_range=(0.95, 1.15)
 )
 img_dir = out_root / "images"
 img_dir.mkdir(parents=True, exist_ok=True)
 search_rel = f"images/search_{i:05d}.png"
 ref_rel = f"images/ref_{i:05d}.png"
 cv2.imwrite(str(out_root/search_rel), search)
 cv2.imwrite(str(out_root/ref_rel), reference)
 target_search_size = target_size * args.search_size / W
 bbox = [
 float(search_center[0]-target_search_size/2),
 float(search_center[1]-target_search_size/2),
 float(search_center[0]+target_search_size/2),
 float(search_center[1]+target_search_size/2),
 ]
 return {
 "id": i,
 "architecture": args.architecture,
 "ref": ref_rel,
 "search": search_rel,
 "center": {"x": float(search_center[0]), "y": float(search_center[1])},
 "bbox": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
 "scale": {"search": search_scale, "reference": ref_scale},
 "rotation_deg": {"search": search_angle, "reference": ref_angle},
 "noise_sigma": {"search": search_noise, "reference": ref_noise},
 "blur_sigma": {"search": search_blur, "reference": ref_blur},
 "search_size": [args.search_size, args.search_size],
 "reference_size": [args.reference_size, args.reference_size],
 "seed": args.seed + 104729*(i+1),
 "independent_captures": True,
 "edge_brightening": True,
 }
def main():
 p = argparse.ArgumentParser()
 p.add_argument("--architecture", choices=["DRAM", "FinFET"], default="DRAM")
 p.add_argument("--num_pairs", type=int, default=30)
 p.add_argument("--output_dir", default="data/generated")
 p.add_argument("--seed", type=int, default=42)
 p.add_argument("--canvas_size", type=int, default=2400)
 p.add_argument("--search_size", type=int, default=1000)
 p.add_argument("--reference_size", type=int, default=100)
 args = p.parse_args()
 if args.num_pairs < 30:
 raise SystemExit("ERROR: --num_pairs must be at least 30.")
 out = Path(args.output_dir)
 out.mkdir(parents=True, exist_ok=True)
 records = [make_pair(i, args, out) for i in range(args.num_pairs)]
 (out/"metadata.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
 readme = f"""# Generated {args.architecture} dataset
Pairs: {args.num_pairs}
Search: {args.search_size}x{args.search_size}
Reference: {args.reference_size}x{args.reference_size}
Seed: {args.seed}
Every pair is generated from the same physical target and then captured
independently. `metadata.json` contains ground-truth center/bbox and all
randomized degradation parameters. """
 (out/"README.txt").write_text(readme, encoding="utf-8")
 print(f"Generated {args.num_pairs} {args.architecture} pairs in {out}")
 print("Ground truth: metadata.json")
if __name__ == "__main__":
 main()
