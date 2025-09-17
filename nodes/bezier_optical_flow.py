#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

def bezier_point(p0, p1, p2, t):
    return (
        int((1-t)**2 * p0[0] + 2*(1-t)*t*p1[0] + t**2 * p2[0]),
        int((1-t)**2 * p0[1] + 2*(1-t)*t*p1[1] + t**2 * p2[1])
    )

def draw_curved_dash_line(overlay, h, w, base_x, base_y, vp_x, vp_y, sway,
                          dash_len=0.05, spacing=0.15, phase=0, color=(0,255,0)):
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


class LaneVizNode(Node):
    def __init__(self):
        super().__init__('bezier_optical_flow')

        # Bridge for ROS <-> OpenCV
        self.bridge = CvBridge()

        # Subscribe to camera feed
        self.subscription = self.create_subscription(
            Image, '/camera/image', self.image_callback, 10)

        # Lucas–Kanade params
        self.lk_params = dict(winSize=(15,15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3,
                                   minDistance=7, blockSize=7)

        # State
        self.old_gray = None
        self.p0 = None
        self.sway_smooth = 0
        self.phase = 0

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w, _ = frame.shape
        vp_x, vp_y = w // 2, int(h * 0.3)

        if self.old_gray is None:
            # First frame → initialize features
            self.old_gray = frame_gray
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
            return

        # Optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)

        forward_motion = 0
        dx = 0
        if p1 is not None and st.sum() > 0:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]

            dx = np.mean(good_new[:,0] - good_old[:,0])  # yaw proxy

            vals = []
            for (x0, y0), (x1, y1) in zip(good_old, good_new):
                v = np.array([x1-x0, y1-y0])
                r = np.array([x0-vp_x, y0-vp_y])
                if np.linalg.norm(r) > 1e-5:
                    vals.append(np.dot(v, r) / np.linalg.norm(r))
            if len(vals) > 0:
                forward_motion = np.mean(vals)

        sway = int(dx * 5)
        self.sway_smooth = int(0.8 * self.sway_smooth + 0.2 * sway)

        # Increment phase only if forward motion
        if forward_motion > 0.5:
            self.phase = (self.phase - 0.02) % 1.0

        overlay = frame.copy()
        base_x, base_y = w//2, h
        draw_curved_dash_line(overlay, h, w, base_x, base_y, vp_x, vp_y,
                              self.sway_smooth, phase=self.phase)

        # Blend dashes
        out_frame = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)

        # Show debug window
        cv2.imshow("Lane reacts to Forward Motion", out_frame)
        cv2.waitKey(1)

        # Update for next step
        self.old_gray = frame_gray.copy()
        self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)


def main(args=None):
    rclpy.init(args=args)
    node = LaneVizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
