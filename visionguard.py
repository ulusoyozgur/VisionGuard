import cv2
import logging
import sys
import time
import json
import threading
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# DeepFace — duygu, yaş, cinsiyet, kişi tanıma
try:
    from deepface import DeepFace
    # Model warm-up: ilk analizin yavaş olmaması için
    _warmup_img = np.zeros((64, 64, 3), dtype=np.uint8)
    try:
        DeepFace.analyze(_warmup_img, actions=["emotion"],
                         enforce_detection=False, silent=True)
    except Exception:
        pass
    DEEPFACE_OK = True
except ImportError:
    DEEPFACE_OK = False
    logging.warning("DeepFace bulunamadı. 'pip install deepface' ile kurabilirsin.")


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

CONFIG_FILE = "visionguard_config.json"

DEFAULT_CONFIG = {
    "camera_id": 0,
    "window_title": "VisionGuard AI v3.1",
    "detection": {
        "scaleFactor": 1.1,
        "minNeighbors": 5,
        "minSize": [30, 30]
    },
    "display": {
        "show_fps": True,
        "show_face_count": True,
        "show_timestamp": True,
        "box_color": [0, 255, 0],
        "eye_box_color": [255, 165, 0],
        "unknown_color": [0, 0, 255],
        "known_color": [255, 215, 0]
    },
    "ai": {
        "emotion": True,
        "age_gender": True,
        "analyze_every_n_frames": 15,
        "recognition": True,
        "recognition_db": "known_faces",
        "recognition_model": "VGG-Face",
        # DÜZELTME #5: VGG-Face cosine için güvenli threshold 0.40
        "recognition_threshold": 0.40
    },
    "tracking": {
        "enabled": True,
        "max_disappeared": 30
    },
    "recording": {
        "enabled": False,
        "output_dir": "recordings",
        "fps": 20.0,
        "fourcc": "mp4v"
    },
    "screenshot": {
        "output_dir": "screenshots"
    }
}


def load_config(path: str = CONFIG_FILE) -> dict:
    if not Path(path).exists():
        with open(path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        logging.info(f"Config oluşturuldu: {path}")
        return DEFAULT_CONFIG
    with open(path) as f:
        cfg = json.load(f)
    logging.info(f"Config yüklendi: {path}")
    return cfg


# ─────────────────────────────────────────────
#  LOGLAMA
# ─────────────────────────────────────────────

def setup_logging() -> None:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"visionguard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


# ─────────────────────────────────────────────
#  PERFORMANS SAYACI
# ─────────────────────────────────────────────

@dataclass
class PerformanceMonitor:
    fps: float = 0.0
    # DÜZELTME #4: frame_count ve total birbirinden ayrıldı
    _window_frames: int = field(default=0, repr=False)
    total_frames: int = 0
    total_faces_detected: int = 0
    _prev_time: float = field(default_factory=time.time, repr=False)

    def update(self, face_count: int) -> None:
        self._window_frames += 1
        self.total_frames += 1
        self.total_faces_detected += face_count
        now = time.time()
        elapsed = now - self._prev_time
        if elapsed >= 0.5:
            self.fps = self._window_frames / elapsed
            self._window_frames = 0      # sadece pencere sıfırlanır
            self._prev_time = now

    def summary(self) -> str:
        return (
            f"Toplam Frame: {self.total_frames} | "
            f"Toplam Yüz Tespiti: {self.total_faces_detected}"
        )


# ─────────────────────────────────────────────
#  ÇOK NESNE TAKİPÇİSİ (Centroid Tracker)
# ─────────────────────────────────────────────

class CentroidTracker:
    """Her yüze benzersiz ID atar, kaybolunca max_disappeared kadar hafızada tutar."""

    def __init__(self, max_disappeared: int = 30):
        self.next_id = 0
        self.objects: dict[int, np.ndarray] = {}
        self.disappeared: dict[int, int] = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid: np.ndarray) -> int:
        obj_id = self.next_id
        self.objects[obj_id] = centroid
        self.disappeared[obj_id] = 0
        self.next_id += 1
        return obj_id

    def deregister(self, obj_id: int) -> None:
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    def update(self, rects) -> dict[int, np.ndarray]:
        if len(rects) == 0:
            for obj_id in list(self.disappeared):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        input_centroids = np.array(
            [(x + w // 2, y + h // 2) for (x, y, w, h) in rects], dtype="float"
        )

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
        else:
            obj_ids = list(self.objects.keys())
            obj_centroids = list(self.objects.values())

            D = np.linalg.norm(
                np.array(obj_centroids)[:, np.newaxis] - input_centroids[np.newaxis, :],
                axis=2
            )

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                obj_id = obj_ids[row]
                self.objects[obj_id] = input_centroids[col]
                self.disappeared[obj_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for row in unused_rows:
                obj_id = obj_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects


# ─────────────────────────────────────────────
#  DEEPFACE ANALİZ SONUCU
# ─────────────────────────────────────────────

@dataclass
class FaceInfo:
    obj_id: int
    emotion: str = "?"
    emotion_score: float = 0.0
    age: int = 0
    gender: str = "?"
    name: str = "Bilinmiyor"
    last_updated: float = field(default_factory=time.time)


# ─────────────────────────────────────────────
#  ANA SINIF
# ─────────────────────────────────────────────

class VisionGuard:
    """
    Gerçek zamanlı yüz tespiti + duygu/yaş/cinsiyet analizi
    + kişi tanıma + çok nesne takibi.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.camera_id: int = config["camera_id"]
        self.cap: cv2.VideoCapture | None = None
        self.recorder: cv2.VideoWriter | None = None
        self.perf = PerformanceMonitor()

        Path(config["screenshot"]["output_dir"]).mkdir(exist_ok=True)
        if config["recording"]["enabled"]:
            Path(config["recording"]["output_dir"]).mkdir(exist_ok=True)
        Path(config["ai"]["recognition_db"]).mkdir(exist_ok=True)

        self.face_cascade = self._load_cascade("haarcascade_frontalface_default.xml")
        self.eye_cascade  = self._load_cascade("haarcascade_eye.xml")

        self.tracker = CentroidTracker(
            max_disappeared=config["tracking"]["max_disappeared"]
        ) if config["tracking"]["enabled"] else None

        self.face_infos: dict[int, FaceInfo] = {}

        # DÜZELTME #1: Hem flag hem veri için aynı lock kullanılıyor
        self._analysis_lock = threading.Lock()
        self._analysis_running = False  # sadece lock altında okunup yazılacak

        self._frame_counter = 0

        # DÜZELTME #2: Yüz kayıt isteği ana döngüyü bloklamıyor
        self._register_requested = False
        self._register_lock = threading.Lock()

    # ── Yardımcı ──────────────────────────────

    @staticmethod
    def _load_cascade(filename: str) -> cv2.CascadeClassifier:
        path = cv2.data.haarcascades + filename
        clf  = cv2.CascadeClassifier(path)
        if clf.empty():
            logging.error(f"Model yüklenemedi: {filename}")
            sys.exit(1)
        return clf

    def _init_recorder(self, frame) -> None:
        rec = self.cfg["recording"]
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*rec["fourcc"])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(rec["output_dir"]) / f"record_{ts}.mp4"
        self.recorder = cv2.VideoWriter(str(path), fourcc, rec["fps"], (w, h))
        logging.info(f"Kayıt başladı: {path}")

    # ── Ana Akış ──────────────────────────────

    def start(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            logging.error(f"Kamera açılamadı (ID: {self.camera_id})")
            return
        logging.info(
            "VisionGuard v3.1 başlatıldı. "
            "[q] çıkış | [s] ekran görüntüsü | [r] kişi kaydet"
        )
        try:
            self._run_loop()
        except Exception as e:
            logging.exception(f"Hata: {e}")
        finally:
            self._cleanup()

    def _run_loop(self) -> None:
        rec_enabled   = self.cfg["recording"]["enabled"]
        ai_cfg        = self.cfg["ai"]
        analyze_every = ai_cfg["analyze_every_n_frames"]

        while True:
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("Frame alınamıyor.")
                break

            frame = cv2.flip(frame, 1)

            if rec_enabled and self.recorder is None:
                self._init_recorder(frame)

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._detect_faces(gray)

            objects = {}
            if self.tracker:
                objects = self.tracker.update(
                    [(x, y, w, h) for (x, y, w, h) in faces]
                )

            self._frame_counter += 1

            # DÜZELTME #1: _analysis_running flag lock altında okunuyor
            with self._analysis_lock:
                already_running = self._analysis_running

            if (DEEPFACE_OK
                    and self._frame_counter % analyze_every == 0
                    and not already_running
                    and len(faces) > 0):
                self._schedule_analysis(frame.copy(), list(faces), dict(objects))

            # DÜZELTME #2: Kayıt isteği varsa thread'de işle
            with self._register_lock:
                reg_req = self._register_requested
                if reg_req:
                    self._register_requested = False

            if reg_req:
                self._register_face_threaded(frame.copy(), list(faces))

            self._draw_detections(frame, gray, faces, objects)
            self._draw_overlay(frame, len(faces))

            if rec_enabled and self.recorder:
                self.recorder.write(frame)

            cv2.imshow(self.cfg["window_title"], frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logging.info("Çıkış.")
                break
            elif key == ord("s"):
                self._take_screenshot(frame)
            elif key == ord("r"):
                # DÜZELTME #2: Sadece flag set ediliyor, input() burada çağrılmıyor
                with self._register_lock:
                    self._register_requested = True

            self.perf.update(len(faces))

    # ── Tespit ────────────────────────────────

    def _detect_faces(self, gray):
        det = self.cfg["detection"]
        return self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=det["scaleFactor"],
            minNeighbors=det["minNeighbors"],
            minSize=tuple(det["minSize"])
        )

    def _detect_eyes(self, face_gray):
        return self.eye_cascade.detectMultiScale(
            face_gray, scaleFactor=1.1, minNeighbors=10
        )

    # ── DeepFace Arka Plan Analizi ─────────────

    def _schedule_analysis(self, frame, faces, objects) -> None:
        """DeepFace analizini ayrı thread'de çalıştır — ana döngüyü bloklamaz."""

        # DÜZELTME #1: Flag lock altında set ediliyor
        with self._analysis_lock:
            self._analysis_running = True

        def _analyze():
            ai      = self.cfg["ai"]
            actions = []
            if ai["emotion"]:
                actions.append("emotion")
            if ai["age_gender"]:
                actions.extend(["age", "gender"])

            updated_infos: dict[int, FaceInfo] = {}

            for i, (x, y, w, h) in enumerate(faces):
                face_img = frame[y:y+h, x:x+w]
                if face_img.size == 0:
                    continue

                # En yakın centroid'e göre obj_id bul
                obj_id = i
                if objects:
                    cx, cy = x + w // 2, y + h // 2
                    obj_id = min(
                        objects,
                        key=lambda oid: np.linalg.norm(
                            np.array(objects[oid]) - [cx, cy]
                        )
                    )

                # Mevcut bilgiyi al (lock altında kopyala)
                with self._analysis_lock:
                    info = self.face_infos.get(obj_id, FaceInfo(obj_id=obj_id))

                # Duygu / yaş / cinsiyet
                if actions:
                    try:
                        result = DeepFace.analyze(
                            face_img, actions=actions,
                            enforce_detection=False, silent=True
                        )
                        r = result[0] if isinstance(result, list) else result
                        if ai["emotion"]:
                            info.emotion = r.get("dominant_emotion", "?")
                            scores = r.get("emotion", {})
                            info.emotion_score = scores.get(info.emotion, 0.0)
                        if ai["age_gender"]:
                            info.age    = int(r.get("age", 0))
                            info.gender = r.get("dominant_gender", "?")
                    except Exception as e:
                        logging.debug(f"DeepFace analiz hatası: {e}")

                # Kişi tanıma
                if ai["recognition"]:
                    db = ai["recognition_db"]
                    db_path = Path(db)
                    if db_path.exists() and any(db_path.iterdir()):
                        try:
                            matches = DeepFace.find(
                                face_img, db_path=db,
                                model_name=ai["recognition_model"],
                                enforce_detection=False, silent=True
                            )
                            if matches and len(matches[0]) > 0:
                                best = matches[0].iloc[0]
                                # DÜZELTME #5: threshold düşürüldü (0.40)
                                dist_key = f"{ai['recognition_model']}_cosine"
                                dist = best.get(dist_key, 1.0)
                                if dist < ai["recognition_threshold"]:
                                    identity   = Path(best["identity"]).stem
                                    info.name  = identity.replace("_", " ").title()
                                else:
                                    info.name = "Bilinmiyor"
                        except Exception as e:
                            logging.debug(f"Kişi tanıma hatası: {e}")

                info.last_updated = time.time()
                updated_infos[obj_id] = info

            # DÜZELTME #1: Tüm güncellemeler tek seferde lock altında yazılıyor
            with self._analysis_lock:
                self.face_infos.update(updated_infos)
                self._analysis_running = False

        t = threading.Thread(target=_analyze, daemon=True)
        t.start()

    # ── Çizim ─────────────────────────────────

    EMOTION_LABEL = {
        "happy":    "MUTLU  :)",
        "sad":      "UZGUN  :(",
        "angry":    "SINIRLI >:(",
        "surprise": "SASKIN  :O",
        "fear":     "KORKMUS :|",
        "disgust":  "TIKSINMIS",
        "neutral":  "NOTR  :-|",
    }

    def _draw_detections(self, frame, gray, faces, objects) -> None:
        disp    = self.cfg["display"]
        f_color = tuple(disp["box_color"])
        e_color = tuple(disp["eye_box_color"])
        k_color = tuple(disp["known_color"])

        # DÜZELTME #1: face_infos'u lock altında snapshot al
        with self._analysis_lock:
            face_infos_snapshot = dict(self.face_infos)

        for i, (x, y, w, h) in enumerate(faces):
            obj_id = i
            if objects:
                cx, cy = x + w // 2, y + h // 2
                obj_id = min(
                    objects,
                    key=lambda oid: np.linalg.norm(
                        np.array(objects[oid]) - [cx, cy]
                    )
                )

            info: FaceInfo | None = face_infos_snapshot.get(obj_id)
            is_known  = info and info.name != "Bilinmiyor"
            box_color = k_color if is_known else f_color

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            lines = []
            if info:
                lines.append(f"[#{obj_id}] {info.name}")
                if self.cfg["ai"]["emotion"] and info.emotion != "?":
                    emo = self.EMOTION_LABEL.get(info.emotion, info.emotion.upper())
                    lines.append(f"{emo} ({info.emotion_score:.0f}%)")
                if self.cfg["ai"]["age_gender"] and info.age > 0:
                    lines.append(f"~{info.age} yas | {info.gender}")
            else:
                lines.append(f"[#{obj_id}] Analiz ediliyor...")

            for j, line in enumerate(lines):
                ty = y - 10 - (len(lines) - 1 - j) * 18
                if ty < 14:
                    ty = y + h + 14 + j * 18
                cv2.putText(frame, line, (x, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, box_color, 1)

            roi_gray  = gray[y:y+h, x:x+w]
            roi_frame = frame[y:y+h, x:x+w]
            for (ex, ey, ew, eh) in self._detect_eyes(roi_gray):
                cv2.rectangle(roi_frame, (ex, ey), (ex+ew, ey+eh), e_color, 1)

        if objects:
            for obj_id, centroid in objects.items():
                cx, cy = int(centroid[0]), int(centroid[1])
                cv2.circle(frame, (cx, cy), 4, (255, 255, 0), -1)
                cv2.putText(frame, str(obj_id), (cx - 8, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    def _draw_overlay(self, frame, face_count: int) -> None:
        disp    = self.cfg["display"]
        h, w    = frame.shape[:2]
        lines   = []

        if disp["show_fps"]:
            lines.append(f"FPS: {self.perf.fps:.1f}")
        if disp["show_face_count"]:
            lines.append(f"Kisi: {face_count}")
        if disp["show_timestamp"]:
            lines.append(datetime.now().strftime("%H:%M:%S"))
        lines.append("[s] goruntu  [r] kaydet  [q] cikis")

        for i, text in enumerate(lines):
            cv2.putText(frame, text, (10, 22 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # DÜZELTME #1: Flag lock altında okunuyor
        with self._analysis_lock:
            running = self._analysis_running

        if running:
            cv2.putText(frame, "AI analiz ediyor...", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)

        if self.cfg["recording"]["enabled"]:
            cv2.circle(frame, (w - 20, 20), 8, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 55, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # ── Kişi Kaydetme (thread-safe, non-blocking) ──

    def _register_face_threaded(self, frame, faces) -> None:
        """
        DÜZELTME #2: input() ayrı thread'de çalışır,
        ana döngü (kamera) bloklanmaz.
        """
        if len(faces) == 0:
            logging.warning("Kayıt için kameraya bakın.")
            return

        def _do_register():
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]
            try:
                name = input(
                    "\nKişi adı girin (boşluk yerine _ kullanın): "
                ).strip()
            except EOFError:
                return
            if not name:
                return
            person_dir = Path(self.cfg["ai"]["recognition_db"]) / name
            person_dir.mkdir(exist_ok=True)
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = person_dir / f"{name}_{ts}.jpg"
            cv2.imwrite(str(path), face_img)
            logging.info(f"Yüz kaydedildi: {path}")

        t = threading.Thread(target=_do_register, daemon=True)
        t.start()

    # ── Ekran Görüntüsü ───────────────────────

    def _take_screenshot(self, frame) -> None:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(self.cfg["screenshot"]["output_dir"]) / f"screenshot_{ts}.png"
        cv2.imwrite(str(path), frame)
        logging.info(f"Ekran görüntüsü: {path}")

    # ── Temizlik ──────────────────────────────

    def _cleanup(self) -> None:
        if self.cap:
            self.cap.release()
        if self.recorder:
            self.recorder.release()
        cv2.destroyAllWindows()
        logging.info(f"Kapatıldı. {self.perf.summary()}")


# ─────────────────────────────────────────────
#  GİRİŞ NOKTASI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging()
    config = load_config()
    guard  = VisionGuard(config)
    guard.start()
