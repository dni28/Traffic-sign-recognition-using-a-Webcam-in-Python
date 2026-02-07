# detect_and_crop.py
# Returns: (roi_bgr, (x1,y1,x2,y2)) or None

import cv2
import numpy as np


def _make_mask(hsv):
    """
    Stronger red + blue HSV masks.
    Tune S/V thresholds if your lighting is low.
    """
    # RED (two hue ranges)
    lower_red1 = np.array([0, 90, 60])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 90, 60])
    upper_red2 = np.array([180, 255, 255])

    # BLUE
    lower_blue = np.array([90, 80, 60])
    upper_blue = np.array([135, 255, 255])

    red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    blue = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = red | blue
    return mask


def _shape_and_quality(cnt):
    """
    Returns (shape_name, circularity, solidity, vertices_count)
    """
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri <= 1e-6:
        return "other", 0.0, 0.0, 0

    circularity = (4.0 * np.pi * area) / (peri * peri)  # 1.0 = perfect circle

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 1e-6 else 0.0  # close to 1 means solid clean shape

    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    v = len(approx)

    # classify by vertices (rough)
    if v == 3:
        shape = "triangle"
    elif 7 <= v <= 10:
        shape = "octagon"  # STOP often ends up 8-ish
    elif v >= 11 and circularity > 0.65:
        shape = "circle"
    else:
        shape = "other"

    return shape, float(circularity), float(solidity), int(v)


def detect_and_crop(frame_bgr):
    """
    Improved detector:
    - Color mask (red/blue)
    - Morphology clean-up
    - Contour filters + shape quality (circularity, solidity)
    Returns best ROI and bbox, or None if no good candidate.
    """
    h, w = frame_bgr.shape[:2]
    frame_area = h * w

    # Blur reduces noisy contours
    blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = _make_mask(hsv)

    # Clean mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1200:  # ignore tiny blobs
            continue

        # ignore huge regions (phone, shirt, etc.)
        if area > 0.45 * frame_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # aspect ratio (signs are roughly square-ish)
        ar = bw / float(bh)
        if ar < 0.65 or ar > 1.55:
            continue

        shape, circ, solid, v = _shape_and_quality(cnt)

        # Quality filters (these remove a LOT of false positives)
        if solid < 0.80:
            continue

        # Accept shapes we care about
        if shape not in ("triangle", "octagon", "circle"):
            continue

        # Scoring: prefer larger, cleaner, more circular/solid shapes
        # (Triangle is less circular, so we weigh solidity+area more)
        norm_area = area / frame_area
        score = (2.0 * solid) + (2.0 * norm_area) + (0.8 * circ)

        # Slightly prefer STOP-like octagons when available
        if shape == "octagon":
            score += 0.2

        if score > best_score:
            best_score = score
            best = (x, y, bw, bh)

    if best is None:
        return None

    x, y, bw, bh = best

    # Padding so the sign border isn't cut
    pad = int(0.12 * max(bw, bh))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)

    roi = frame_bgr[y1:y2, x1:x2].copy()
    return roi, (x1, y1, x2, y2)
