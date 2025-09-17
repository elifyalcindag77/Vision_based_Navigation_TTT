
#!/usr/bin/env python3
import cv2
import numpy as np

def bezier_point(p0, p1, p2, t):
    return (
        int((1-t)**2 * p0[0] + 2*(1-t)*t*p1[0] + t**2 * p2[0]),
        int((1-t)**2 * p0[1] + 2*(1-t)*t*p1[1] + t**2 * p2[1])
    )

def draw_curved_dash_line(overlay, h, w, base_x, base_y, vp_x, vp_y, sway,
                          dash_len=0.05, spacing=0.15, phase=0, color=(0,255, 0)):
    ctrl_x = base_x + sway
    ctrl_y = (base_y + vp_y) // 2
    p0, p1, p2 = (base_x, base_y), (ctrl_x, ctrl_y), (vp_x, vp_y)

    cycle = dash_len + spacing
    num_dashes = int(1 / cycle) + 2

    for i in range(num_dashes):
        t0 = (i * cycle + phase) % 1.0
        t1 = (t0 + dash_len) % 1.0

        if t1 < t0:
            continue

        x0, y0 = bezier_point(p0, p1, p2, t0)
        x1, y1 = bezier_point(p0, p1, p2, t1)

        thickness = max(1, int(20 * (1 - t0)))
        cv2.line(overlay, (x0, y0), (x1, y1), color, thickness)

def main():
    cap = cv2.VideoCapture(0)  # <-- /dev/video5
    if not cap.isOpened():
        print("Error: Could not open camera at /dev/video5")
        return

    lk_params = dict(winSize=(15,15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    sway_smooth = 0
    phase = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        h, w, _ = frame.shape
        vp_x, vp_y = w // 2, int(h * 0.3)

        forward_motion = 0
        dx = 0

        if p1 is not None and st.sum() > 0:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Horizontal turn (yaw proxy)
            dx = np.mean(good_new[:,0] - good_old[:,0])

            # Forward motion proxy: dot product with radial vector
            vals = []
            for (x0, y0), (x1, y1) in zip(good_old, good_new):
                v = np.array([x1-x0, y1-y0])
                r = np.array([x0-vp_x, y0-vp_y])
                if np.linalg.norm(r) > 1e-5:
                    vals.append(np.dot(v, r) / np.linalg.norm(r))
            if len(vals) > 0:
                forward_motion = np.mean(vals)

        sway = int(dx * 5)
        sway_smooth = int(0.8 * sway_smooth + 0.2 * sway)

        # Increment phase only if forward motion detected
        if forward_motion > 0.5:  # threshold for moving forward
            phase = (phase - 0.02) % 1.0

        overlay = frame.copy()
        base_x, base_y = w//2, h

        draw_curved_dash_line(overlay, h, w, base_x, base_y, vp_x, vp_y, sway_smooth, phase=phase)

        frame = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)
        cv2.imshow("Lane reacts to Forward Motion", frame)

        old_gray = frame_gray.copy()
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

