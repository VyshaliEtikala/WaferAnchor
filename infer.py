"""
Inference script for locating reference pattern inside search image.

Usage:
  python infer.py --ref path/to/ref.png --search path/to/search.png

Outputs a single line: (x,y)
"""
import argparse
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ----------------- Helpers -----------------
def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def top_k_peaks(response, k=10, min_dist=16, threshold=0.5):
    # Simple non-max suppression to find peaks
    coords = []
    resp = response.copy()
    for _ in range(k):
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(resp)
        if maxVal < threshold:
            break
        coords.append((maxLoc[0], maxLoc[1], maxVal))
        x,y = maxLoc
        x0 = max(0, x - min_dist)
        x1 = min(resp.shape[1]-1, x + min_dist)
        y0 = max(0, y - min_dist)
        y1 = min(resp.shape[0]-1, y + min_dist)
        resp[y0:y1+1, x0:x1+1] = 0
    return coords

def orb_ransac_inliers(template, patch, max_matches=300):
    # ORB descriptors + BFMatcher + RANSAC homography inlier count
    orb = cv2.ORB_create(nfeatures=500)
    kps1, des1 = orb.detectAndCompute(template, None)
    kps2, des2 = orb.detectAndCompute(patch, None)
    if des1 is None or des2 is None or len(kps1) < 6 or len(kps2) < 6:
        return 0, [], []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 6:
        return len(good), good, (kps1, kps2)
    src_pts = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if mask is None:
        return 0, good, (kps1, kps2)
    inliers = int(mask.sum())
    return inliers, good, (kps1, kps2)

# ----------------- Main detection pipeline -----------------
def detect_center(ref_img, search_img, expected_target_size=100, scale_candidates=None, topk_per_scale=5):
    ref_gray = to_gray(ref_img)
    search_gray = to_gray(search_img)
    # If scale candidates not given, search around 8..12 (downscale factors)
    if scale_candidates is None:
        scale_candidates = np.linspace(6, 14, 9)  # downscale reference by these factors to produce templates near 100px
    candidates = []
    for s in scale_candidates:
        sw = max(8, int(round(ref_gray.shape[1] / s)))
        sh = max(8, int(round(ref_gray.shape[0] / s)))
        template = cv2.resize(ref_gray, (sw, sh), interpolation=cv2.INTER_AREA)
        # normalized cross-correlation
        method = cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(search_gray, template, method)
        peaks = top_k_peaks(res, k=topk_per_scale, min_dist=sw//2, threshold=0.4)
        for (px, py, score) in peaks:
            # center coordinates of matched region
            cx = px + sw//2
            cy = py + sh//2
            # extract patch from search for verification (safe crop)
            x0 = max(0, px)
            y0 = max(0, py)
            x1 = min(search_gray.shape[1], px + sw)
            y1 = min(search_gray.shape[0], py + sh)
            patch = search_gray[y0:y1, x0:x1]
            # resize template and patch to same size for SSIM and ORB
            t_resized = cv2.resize(template, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_AREA)
            # edge SSIM on Canny edges
            t_edges = cv2.Canny(t_resized, 50, 150)
            p_edges = cv2.Canny(patch, 50, 150)
            try:
                edge_ssim = ssim(t_edges, p_edges, data_range=255)
            except Exception:
                edge_ssim = 0.0
            # ORB+RANSAC inlier count
            inliers, _, _ = orb_ransac_inliers(t_resized, patch)
            # composite score
            score_combined = 0.5 * score + 0.35 * (edge_ssim + 1)/2 + 0.15 * (inliers / (inliers + 10))
            candidates.append({"cx":cx, "cy":cy, "score":score_combined, "cc":score, "edge_ssim":edge_ssim, "inliers":inliers, "scale_s":s})
    if not candidates:
        # fallback: use center of search
        h,w = search_gray.shape
        return (w//2, h//2)
    # rank by score, but later break ties by proximity to center
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    # pick top plausible candidates, then choose closest to center among top plausible set
    top_scores = [c["score"] for c in candidates[:5]]
    threshold = max(top_scores[0]*0.85, np.median(top_scores))
    plausible = [c for c in candidates if c["score"] >= threshold]
    if len(plausible) == 1:
        best = plausible[0]
    else:
        # pick one closest to the center of the search image (hackathon rule)
        h,w = search_gray.shape
        sx, sy = w/2, h/2
        best = min(plausible, key=lambda c: (c["cx"]-sx)**2 + (c["cy"]-sy)**2)
    return (int(best["cx"]), int(best["cy"]))

# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--search", required=True)
    args = parser.parse_args()
    ref = cv2.imread(args.ref, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)
    cx, cy = detect_center(ref, search)
    print(f"({cx},{cy})")

if __name__ == "__main__":
    main()