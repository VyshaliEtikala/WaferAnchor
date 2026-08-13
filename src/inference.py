#!/usr/bin/env python3
"""Standalone PS2 reference-to-search localization.
CLI:
python src/inference.py --reference path/to/ref.png --search path/to/search.png
Output:
Predicted center: (x, y)
"""
from __future__ import annotations
import argparse
import cv2
import numpy as np
def _prep(img):
 img = img.astype(np.float32)
 img = cv2.GaussianBlur(img, (3,3), 0)
 return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
def _edge(img):
 g = cv2.GaussianBlur(img, (3,3), 0)
 gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
 gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
 mag = cv2.magnitude(gx, gy)
 return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
def _match(template, image):
 """Multi-scale normalized correlation; return candidate centers and scores."""
 th, tw = template.shape
 ih, iw = image.shape
 candidates = []
 # The nominal reference/search scale is ~1.0, but randomized scale exists.
 for s in np.linspace(0.82, 1.18, 19):
 nw = max(16, int(round(tw*s)))
 nh = max(16, int(round(th*s)))
 if nw >= iw or nh >= ih:
 continue
 t = cv2.resize(template, (nw, nh), interpolation=cv2.INTER_LINEAR)
 r = cv2.matchTemplate(image, t, cv2.TM_CCOEFF_NORMED)
 _, score, _, loc = cv2.minMaxLoc(r)
 x, y = loc
 candidates.append((float(score), x+nw/2.0, y+nh/2.0, nw, nh))
 return candidates
def detect_center(reference, search):
 """Return (x,y) in search-image coordinates.
 Two-stage search keeps inference practical on a phone: a coarse multi-scale
 search finds candidate regions, then full-resolution intensity/edge
 matching refines them. Near-tied candidates use the center-closest rule.
 """
 ref = _prep(reference)
 sea = _prep(search)
 h, w = sea.shape
 # Coarse stage at half resolution.
 ref_c = cv2.resize(ref, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
 sea_c = cv2.resize(sea, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
 ref_ec, sea_ec = _edge(ref_c), _edge(sea_c)
 coarse = []
 for t, im, weight in ((ref_c, sea_c, 0.65), (ref_ec, sea_ec, 0.35)):
 th, tw = t.shape
 for s in np.linspace(0.88, 1.12, 9):
 nw, nh = int(round(tw*s)), int(round(th*s))
 if nw < 20 or nh < 20 or nw >= im.shape[1] or nh >= im.shape[0]:
 continue
 tt = cv2.resize(t, (nw, nh), interpolation=cv2.INTER_LINEAR)
 r = cv2.matchTemplate(im, tt, cv2.TM_CCOEFF_NORMED)
 _, score, _, loc = cv2.minMaxLoc(r)
 coarse.append((weight*float(score), (loc[0]+nw/2)*2.0, (loc[1]+nh/2)*2.0))
 if not coarse:
 raise RuntimeError("No valid template-match candidate.")
 coarse.sort(reverse=True)
 # Keep several coarse candidates to avoid committing to a repeated cell. candidates=[]
 for c in coarse:
 if all((c[1]-q[1])**2+(c[2]-q[2])**2 > 45**2 for q in candidates):
 candidates.append(c)
 if len(candidates) >= 8:
 break
 # Full-resolution local refinement around each coarse candidate.
 refined=[]
 ref_e=_edge(ref)
 for base_score, cx, cy in candidates:
 r=90
 x1=max(0,int(round(cx))-r); y1=max(0,int(round(cy))-r)
 x2=min(w,int(round(cx))+r); y2=min(h,int(round(cy))+r)
 roi=sea[y1:y2,x1:x2]
 roi_e=_edge(roi)
 if roi.shape[0] <= ref.shape[0] or roi.shape[1] <= ref.shape[1]:
 continue
 ri=cv2.matchTemplate(roi, ref, cv2.TM_CCOEFF_NORMED)
 re=cv2.matchTemplate(roi_e, ref_e, cv2.TM_CCOEFF_NORMED)
 _, si, _, li=cv2.minMaxLoc(ri)
 _, se, _, le=cv2.minMaxLoc(re)
 score=0.65*float(si)+0.35*float(se)
 tw,th=ref.shape[1],ref.shape[0]
 px=x1+li[0]+tw/2
 py=y1+li[1]+th/2
 refined.append((score,px,py))
 if not refined:
 return float(candidates[0][1]), float(candidates[0][2])
 refined.sort(key=lambda z:z[0], reverse=True)
 best=refined[0][0]
 eligible=[c for c in refined if c[0] >= best-0.03]
 cx0,cy0=w/2,h/2
 chosen=min(eligible,key=lambda c:(c[1]-cx0)**2+(c[2]-cy0)**2)
 return float(chosen[1]), float(chosen[2])
def main():
 p = argparse.ArgumentParser()
 p.add_argument("--reference", required=True)
 p.add_argument("--search", required=True)
 args = p.parse_args()
 ref = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
 search = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)
 if ref is None:
 raise FileNotFoundError(args.reference)
 if search is None:
 raise FileNotFoundError(args.search)
 x, y = detect_center(ref, search)
 print(f"Predicted center: ({x:.2f}, {y:.2f})")
if __name__ == "__main__":
 main()
