#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTT (Time-To-Transit) tabanlı dinamik engel kaçınma denetleyicisi
Tek kamera görüntüsünden optik akış (LK) hesaplar, ROI bazlı TTT ve
"dinamik risk" (yaklaşan nesneler) üretir; hız ve yön komutu yayınlar.

ROS 2 (rclpy) + OpenCV gerektirir. Stereo/depth gerekmez.

Kullanım (örnek):
  # Paket adın: mobilenettt_visionnavigation
  ros2 run mobilenettt_visionnavigation ttt_dynamic_avoid_controller.py --ros-args \
    -p image_topic:=/camera/image_raw -p v_max:=0.6 -p tau_stop:=0.6

Not: Bu düğüm kendi başına OF ve TTT hesaplar; orijinal repodaki
/OpticalFlow veya /TauComputation düğümlerine bağımlı değildir.
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

try:
    from cv_bridge import CvBridge
    import cv2
except Exception as e:
    raise RuntimeError("OpenCV ve cv_bridge gereklidir: sudo apt install ros-${ROS_DISTRO}-cv-bridge python3-opencv")


def robust_median(vec):
    if len(vec) == 0:
        return 0.0
    return float(np.median(np.asarray(vec)))


class TTTDynamicAvoid(Node):
    def __init__(self):
        # Parametre override'larını otomatik kabul et; yeniden ilan hatalarını önler
        super().__init__(
            'ttt_dynamic_avoid_controller',
            automatically_declare_parameters_from_overrides=True
        )

        # --- Parametreleri güvenli ilan et (varsa yeniden ilan etme) ---
        self._declare_if_absent('image_topic', '/camera/image_raw')
        self._declare_if_absent('cmd_vel_topic', '/cmd_vel')
        # use_sim_time BİLEREK ilan edilmiyor (sistem genelinde mevcut olabilir)
        self._declare_if_absent('width', 640)
        self._declare_if_absent('height', 480)
        # LK + özellik
        self._declare_if_absent('max_corners', 600)
        self._declare_if_absent('quality_level', 0.01)
        self._declare_if_absent('min_distance', 7.0)
        self._declare_if_absent('lk_win', 15)
        self._declare_if_absent('lk_max_level', 2)
        self._declare_if_absent('lk_iters', 20)
        self._declare_if_absent('feature_refresh_hz', 3.0)
        # ROI (sol-orta-sağ)
        self._declare_if_absent('roi_split', [0.0, 0.33, 0.66, 1.0])
        # Hız parametreleri
        self._declare_if_absent('v_max', 0.5)
        self._declare_if_absent('v_min', 0.0)
        self._declare_if_absent('accel_limit', 0.8)
        self._declare_if_absent('omega_max', 1.8)
        # TTT eşikleri (saniye)
        self._declare_if_absent('tau_stop', 0.6)
        self._declare_if_absent('tau_slow', 2.0)
        self._declare_if_absent('tau_clip_min', 0.05)
        self._declare_if_absent('tau_clip_max', 6.0)
        # Kazançlar
        self._declare_if_absent('k_avoid', 1.2)
        self._declare_if_absent('k_balance', 0.8)
        # Gürültü/filtre
        self._declare_if_absent('min_points_per_roi', 30)
        self._declare_if_absent('min_forward_points', 80)
        self._declare_if_absent('flow_mag_min', 0.2)
        # Debug yayınları
        self._declare_if_absent('publish_debug', True)

        self.image_topic = self.get_parameter('image_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)

        qos = QoSProfile(depth=5)
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos.history = QoSHistoryPolicy.KEEP_LAST

        self.sub_img = self.create_subscription(Image, self.image_topic, self.on_image, qos)
        self.pub_cmd = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.pub_tau = self.create_publisher(Float32MultiArray, 'ttt_debug/tau_rois', 10)
        self.pub_dyn = self.create_publisher(Float32MultiArray, 'ttt_debug/tau_dyn_rois', 10)

        self.bridge = CvBridge()
        self.prev_gray = None
        self.prev_pts = None
        self.last_feature_refresh = self.get_clock().now()

        # Hedef hızlar
        self.v_cmd = 0.0
        self.omega_cmd = 0.0

        # 30 Hz komut yayın
        self.create_timer(1.0/30.0, self.publish_cmd)

        self.get_logger().info(f"TTT Dynamic Avoid çalışıyor. image:= {self.image_topic} -> {self.cmd_vel_topic}")

    # Var olmayan parametreyi ilan et
    def _declare_if_absent(self, name, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    # --- Görüntü akışı ---
    def on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        if frame is None:
            return
        if frame.shape[:2][::-1] != (self.width, self.height):
            gray = cv2.resize(frame, (self.width, self.height))
        else:
            gray = frame

        # Başlangıçta ya da belirli aralıklarla özellik seç
        now = self.get_clock().now()
        need_refresh = (
            self.prev_pts is None or
            ((now - self.last_feature_refresh).nanoseconds * 1e-9 > (1.0 / max(1e-3, self.get_parameter('feature_refresh_hz').value)))
        )

        if self.prev_gray is None or need_refresh:
            self.prev_gray = gray.copy()
            self.prev_pts = cv2.goodFeaturesToTrack(
                self.prev_gray,
                maxCorners=int(self.get_parameter('max_corners').value),
                qualityLevel=float(self.get_parameter('quality_level').value),
                minDistance=float(self.get_parameter('min_distance').value),
                blockSize=7
            )
            self.last_feature_refresh = now
            return  # Akış için bir sonraki kareyi bekle

        # LK optik akış
        lk_params = dict(
            winSize=(int(self.get_parameter('lk_win').value), int(self.get_parameter('lk_win').value)),
            maxLevel=int(self.get_parameter('lk_max_level').value),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(self.get_parameter('lk_iters').value), 0.03)
        )

        next_pts, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None, **lk_params)
        if next_pts is None:
            self.prev_gray = gray.copy()
            self.prev_pts = None
            return

        good_new = next_pts[st == 1]
        good_old = self.prev_pts[st == 1]

        if len(good_new) < 20:
            self.prev_gray = gray.copy()
            self.prev_pts = None
            return

        # Vektörler
        flows = (good_new - good_old).reshape(-1, 2)
        pts = good_old.reshape(-1, 2)

        # Küçük akışları ele
        mag = np.linalg.norm(flows, axis=1)
        sel = mag > float(self.get_parameter('flow_mag_min').value)
        flows = flows[sel]
        pts = pts[sel]
        mag = mag[sel]

        if len(pts) < 20:
            self.prev_gray = gray.copy()
            self.prev_pts = None
            return

        # Ego hareket yaklaşık çıkarımı: medyan akışı global olarak çıkar (basit model)
        global_flow = np.median(flows, axis=0)  # [ux, uy]
        flows_resid = flows - global_flow  # hareketli nesneler ~ artıklarda

        # ROI bölünmesi
        splits = np.array(self.get_parameter('roi_split').value, dtype=float)
        xs = pts[:, 0]
        rois = np.digitize(xs / self.width, splits) - 1  # 0..len-2
        n_rois = len(splits) - 1

        # TTT tahmini: tau = ||p|| / (v_r) ; v_r = flow'un radyal bileşeni (p yönünde), yaklaşma için v_r>0
        cx, cy = self.width / 2.0, self.height / 2.0
        p = np.stack([pts[:, 0] - cx, pts[:, 1] - cy], axis=1)
        p_norm = np.linalg.norm(p, axis=1) + 1e-6
        p_hat = p / p_norm[:, None]
        v_r = np.sum(flows * p_hat, axis=1)  # radyal bileşen (piksel/frame)

        with np.errstate(divide='ignore', invalid='ignore'):
            tau_all = np.where(v_r > 1e-6, p_norm / v_r, np.inf)

        # ROI bazlı TTT (dengeleme için): medyanı kullan
        tau_rois = [np.median(tau_all[rois == i]) if np.any(rois == i) else np.inf for i in range(n_rois)]

        # Dinamik risk: artık (residual) akıştan hesapla (yaklaşan nesne, kamera hareketine uymayan)
        v_r_resid = np.sum(flows_resid * p_hat, axis=1)
        tau_dyn = np.where(v_r_resid > 1e-6, p_norm / v_r_resid, np.inf)
        tau_dyn_rois = [
            np.min(tau_dyn[(rois == i) & (tau_dyn > 0)]) if np.any(rois == i) else np.inf
            for i in range(n_rois)
        ]

        # Güvenilirlik filtreleri
        counts = [int(np.sum(rois == i)) for i in range(n_rois)]
        min_pts = int(self.get_parameter('min_points_per_roi').value)
        for i in range(n_rois):
            if counts[i] < min_pts:
                tau_rois[i] = np.inf
                tau_dyn_rois[i] = np.inf

        # Kırpma (stabilite)
        tmin = float(self.get_parameter('tau_clip_min').value)
        tmax = float(self.get_parameter('tau_clip_max').value)
        tau_rois = [float(np.clip(t, tmin, tmax)) for t in tau_rois]
        tau_dyn_rois = [float(np.clip(t, tmin, tmax)) for t in tau_dyn_rois]

        # --- Denetim ---
        v_max = float(self.get_parameter('v_max').value)
        tau_stop = float(self.get_parameter('tau_stop').value)
        tau_slow = float(self.get_parameter('tau_slow').value)
        k_avoid = float(self.get_parameter('k_avoid').value)
        k_balance = float(self.get_parameter('k_balance').value)
        omega_max = float(self.get_parameter('omega_max').value)

        # İleri hız: en tehlikeli (en küçük) dinamik tau'ya göre ölçekle
        tau_dyn_min = float(np.min(tau_dyn_rois)) if len(tau_dyn_rois) else np.inf
        if tau_dyn_min < tau_stop:
            v_des = 0.0
        elif tau_dyn_min < tau_slow:
            v_des = v_max * (tau_dyn_min - tau_stop) / max(1e-3, (tau_slow - tau_stop))
        else:
            v_des = v_max

        # Yön: (1) TTT dengeleme (sol-sağ), (2) dinamik risk iticisi
        # Basit 3 ROI varsayıyoruz: [sol, orta, sağ]
        if n_rois >= 3:
            left, center, right = tau_rois[0], tau_rois[1], tau_rois[2]
            # dengeleme: sağ daha uzak (tau büyük) ise sağa dönme isteği azalır
            balance = k_balance * (1.0 / max(1e-3, left) - 1.0 / max(1e-3, right))
            # dinamik kaçınma: risk sağdaysa sola dön (sağ ROI'deki tau_dyn küçükse +omega)
            dyn_left, dyn_center, dyn_right = tau_dyn_rois[0], tau_dyn_rois[1], tau_dyn_rois[2]
            avoid = k_avoid * (1.0 / max(1e-3, dyn_right) - 1.0 / max(1e-3, dyn_left))
            omega_des = balance + avoid
        else:
            omega_des = 0.0

        # Doyumlar ve hız değişim sınırı
        omega_des = float(np.clip(omega_des, -omega_max, omega_max))
        a_lim = float(self.get_parameter('accel_limit').value)
        dt = 1.0 / 30.0
        self.v_cmd += np.clip(v_des - self.v_cmd, -a_lim * dt, a_lim * dt)
        self.omega_cmd = omega_des

        # Debug yayınları
        if self.get_parameter('publish_debug').value:
            self.pub_tau.publish(Float32MultiArray(data=tau_rois))
            self.pub_dyn.publish(Float32MultiArray(data=tau_dyn_rois))

        # Durum güncelle
        self.prev_gray = gray.copy()
        self.prev_pts = good_new.reshape(-1, 1, 2)

    def publish_cmd(self):
        msg = Twist()
        msg.linear.x = float(np.clip(self.v_cmd, self.get_parameter('v_min').value, self.get_parameter('v_max').value))
        msg.angular.z = self.omega_cmd
        self.pub_cmd.publish(msg)


def main():
    rclpy.init()
    node = TTTDynamicAvoid()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
