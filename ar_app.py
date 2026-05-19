import os
import cv2
import numpy as np
import threading
import queue
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.core.window import Window
from kivy.core.camera import Camera
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
from kivy.logger import Logger

# ==========================================
# 1. AUTO-SETUP (ينشئ الملفات المطلوبة تلقائياً)
# ==========================================
def setup_environment():
    folders = ['assets/markers', 'assets/images', 'assets/videos', 'assets/models']
    for f in folders:
        os.makedirs(f, exist_ok=True)
        
    # إنشاء علامة التتبع (Marker) إذا لم تكن موجودة
    marker_path = 'assets/markers/marker.jpg'
    if not os.path.exists(marker_path):
        img = np.zeros((500, 500), dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (499, 499), 255, 30)
        cv2.circle(img, (125, 125), 60, 255, -1)
        cv2.rectangle(img, (250, 250), (450, 450), 255, -1)
        cv2.line(img, (125, 450), (450, 125), 255, 20)
        noise = np.random.randint(0, 80, (500, 500), dtype=np.uint8)
        img = cv2.add(img, noise)
        cv2.imwrite(marker_path, img)
        print("[INFO] Created marker at assets/markers/marker.jpg -> اطبعها على ورق!")

    # إنشاء Buildozer.spec تلقائياً
    if not os.path.exists('buildozer.spec'):
        spec_content = """[app]
title = PyAR Pro
package.name = pyarpro
package.domain = org.pyar
source.dir = .
source.include_exts = py,png,jpg,jpeg,mp4,glb
version = 1.0.0
requirements = python3==3.11.1,kivy[base]==2.3.0,opencv-contrib-python-headless==4.8.1.78,numpy==1.26.1
android.permissions = CAMERA
android.api = 33
android.minapi = 24
orientation = portrait
fullscreen = 1
android.arch = arm64-v8a
"""
        with open('buildozer.spec', 'w') as f:
            f.write(spec_content)
        print("[INFO] Created buildozer.spec")

setup_environment()

# ==========================================
# 2. KALMAN FILTER (لتنعيم الحركة وإزالة الاهتزاز)
# ==========================================
class PoseKalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 6, 0)
        self.kf.measurementMatrix = np.eye(6, dtype=np.float32)
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(6, dtype=np.float32) * 0.1
        
    def update(self, measurement):
        m = measurement.reshape(6, 1).astype(np.float32)
        self.kf.correct(m)
        prediction = self.kf.predict()
        return prediction.flatten()

# ==========================================
# 3. IMAGE TRACKER (ORB + FLANN)
# ==========================================
class ImageTracker:
    def __init__(self):
        self.detector = cv2.ORB_create(nfeatures=1000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.ref_kp = []
        self.ref_des = []
        self.marker_size = 0.1

    def add_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: return
        kp, des = self.detector.detectAndCompute(img, None)
        self.ref_kp.append(kp)
        self.ref_des.append(des)
        self.marker_size = max(img.shape) * 0.001

    def track(self, frame_gray):
        kp_frame, des_frame = self.detector.detectAndCompute(frame_gray, None)
        if des_frame is None: return None
        
        for i, des_ref in enumerate(self.ref_des):
            if des_ref is None: continue
            matches = self.flann.knnMatch(des_ref, des_frame, k=2)
            good = [m for m, n in matches if m.distance < 0.7 * n.distance]
            
            if len(good) > 10:
                src_pts = np.float32([self.ref_kp[i][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is not None: return H
        return None

# ==========================================
# 4. POSE ESTIMATOR (تحديد الإحداثيات 3D)
# ==========================================
class PoseEstimator:
    def __init__(self, w, h):
        self.kf = PoseKalmanFilter()
        fx = fy = w * 1.1
        cx, cy = w / 2, h / 2
        self.cam_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)

    def estimate(self, H):
        if H is None: return None
        obj_pts = np.array([[0,0,0], [0.1,0,0], [0.1,0.1,0], [0,0.1,0]], dtype=np.float32)
        img_pts = cv2.perspectiveTransform(obj_pts.reshape(-1,1,2), H)
        _, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.cam_matrix, self.dist_coeffs)
        
        measurement = np.hstack((tvec.flatten(), rvec.flatten()))
        smoothed = self.kf.update(measurement)
        s_tvec = smoothed[:3]
        
        # حساب موقع النقطة على الشاشة (2D Projection)
        x_2d = int(self.cam_matrix[0,0] * s_tvec[0] / s_tvec[2] + self.cam_matrix[0,2])
        y_2d = int(self.cam_matrix[1,1] * s_tvec[1] / s_tvec[2] + self.cam_matrix[1,2])
        
        # قلب محور Y لأن كيفي يبدأ من الأسفل بينما OpenCV من الأعلى
        y_2d_kivy = Window.height - y_2d
        
        return (x_2d, y_2d_kivy)

# ==========================================
# 5. PROCESSING THREAD (خيط بدائي لرفع الأداء)
# ==========================================
class ARThread(threading.Thread):
    def __init__(self, tracker, pose):
        super().__init__(daemon=True)
        self.tracker = tracker
        self.pose = pose
        self.queue = queue.Queue(maxsize=1)
        self.result = None

    def run(self):
        while True:
            frame = self.queue.get()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            H = self.tracker.track(gray)
            self.result = self.pose.estimate(H)

# ==========================================
# 6. KIVY APPLICATION (الواجهة والكاميرا)
# ==========================================
class ARLayout(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # تهيئة محرك التتبع
        self.tracker = ImageTracker()
        self.tracker.add_image('assets/markers/marker.jpg')
        self.pose = PoseEstimator(Window.width, Window.height)
        self.thread = ARThread(self.tracker, self.pose)
        self.thread.start()
        
        # تشغيل الكاميرا
        self.cam = Camera(index=0, resolution=(1280, 720), stopped=True)
        self.cam.bind(on_texture=self.on_cam_frame)
        self.add_widget(self.cam)
        self.cam.play()
        
        # إنشاء مستطيل شفاف سيتبع العلامة
        with self.canvas.after:
            Color(0, 1, 0.5, 0.6) # أخضر شفاف
            self.ar_rect = Rectangle(size=(150, 150), pos=(0, 0))
            
        # تحديث الواجهة 60 مرة بالثانية
        Clock.schedule_interval(self.update_ar, 1.0/60.0)

    def on_cam_frame(self, instance):
        texture = instance.texture
        if not texture: return
        frame = np.frombuffer(texture.pixels, dtype=np.uint8).reshape(texture.size[1], texture.size[0], 4)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        
        # إرسال الفريم لخيط المعالجة إذا كان فارغاً
        if self.thread.queue.empty():
            self.thread.queue.put(frame)

    def update_ar(self, dt):
        pos = self.thread.result
        if pos:
            # تحريك المستطيل الأخضر ليتتبع العلامة المطبوعة
            self.ar_rect.pos = (pos[0] - 75, pos[1] - 75)
            self.ar_rect.size = (150, 150)
        else:
            # إخفاء المستطيل إذا لم يتم رؤية العلامة
            self.ar_rect.pos = (-200, -200)

class ARApp(App):
    def build(self):
        Window.fullscreen = True
        return ARLayout()

if __name__ == '__main__':
    ARApp().run()