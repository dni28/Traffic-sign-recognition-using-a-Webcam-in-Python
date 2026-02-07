import cv2
import numpy as np

def red_blue_mask(hsv):
    # Red wraps around HSV hue range -> two intervals
    lower_red1 = np.array([0, 80, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 80, 50])
    upper_red2 = np.array([180, 255, 255])

    # Blue range (tweak if needed)
    lower_blue = np.array([90, 80, 50])
    upper_blue = np.array([130, 255, 255])

    red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    blue = cv2.inRange(hsv, lower_blue, upper_blue)

    return red | blue

def shape_type(cnt):
    """Classify contour as triangle/rectangle/circle-like based on polygon approximation."""
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
    v = len(approx)

    if v == 3:
        return "triangle"
    if v == 4:
        return "rectangle"
    if v >= 8:
        return "circle"
    return "other"

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    kernel = np.ones((5, 5), np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read frame.")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1) color mask
        mask = red_blue_mask(hsv)

        # 2) denoise mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 3) contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = frame.shape[0] * frame.shape[1]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1500:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # square-ish only
            ar = w / float(h)
            if ar < 0.75 or ar > 1.35:
                continue

            # shape filter
            st = shape_type(cnt)
            if st not in ("circle", "triangle"):
                continue

            # ignore very large detections (like whole phone)
            if area > 0.40 * frame_area:
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, st, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Mask", mask)
        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
