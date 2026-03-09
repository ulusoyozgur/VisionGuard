import cv2
import logging
import sys

# 1. Profesyonel Loglama Yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class VisionGuard:
    """
    Gerçek zamanlı yüz tespiti ve takibi yapan sınıf.
    OpenCV Haar Cascade sınıflandırıcılarını kullanır.
    """
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        
        # Model dosyasının yolunu belirliyoruz
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Model yüklenemezse sistemi güvenli bir şekilde durdur
        if self.face_cascade.empty():
            logging.error("Yüz tanıma modeli yüklenemedi! Dosya yolunu kontrol edin.")
            sys.exit(1)

    def start(self) -> None:
        """Kamerayı başlatır ve yüz takibi ana döngüsüne girer."""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        # Kamera erişimi kontrolü
        if not self.cap.isOpened():
            logging.error(f"Kamera (ID: {self.camera_id}) açılamadı. Donanımı kontrol edin.")
            return

        logging.info("VisionGuard başlatıldı. Çıkmak için video penceresindeyken 'q' tuşuna basın.")

        # Hata yönetimi (Kablo çıkması vb. durumlara karşı)
        try:
            self._run_loop()
        except Exception as e:
            logging.error(f"Görüntü işleme sırasında beklenmeyen bir hata oluştu: {e}")
        finally:
            self._cleanup()

    def _run_loop(self) -> None:
        """Kameradan görüntü alıp işleyen kapalı (private) döngü metodu."""
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                logging.warning("Kameradan kare (frame) alınamıyor. Bağlantı kopmuş olabilir.")
                break

            # İşlemciyi yormamak için görüntüyü gri tonlamaya çeviriyoruz
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Yüzleri tespit et (Parametreler hassasiyeti belirler)
            faces = self.face_cascade.detectMultiScale(
                gray_frame, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Bulunan yüzleri çiz
            self._draw_faces(frame, faces)

            # Ekrana yansıt
            cv2.imshow('VisionGuard AI v1.0', frame)

            # Çıkış kontrolü
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Kullanıcı tarafından çıkış komutu verildi.")
                break

    def _draw_faces(self, frame, faces) -> None:
        """Bulunan yüzlerin etrafına dikdörtgen ve bilgi metni çizer."""
        for (x, y, w, h) in faces:
            # Yeşil dikdörtgen (BGR formatında: 0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Üstüne metin yazma
            cv2.putText(
                frame, 
                "HEDEF KILITLENDI", 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 255, 0), 
                2
            )

    def _cleanup(self) -> None:
        """Bellek sızıntılarını önlemek için donanım kaynaklarını serbest bırakır."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        logging.info("Kaynaklar temizlendi, sistem güvenli bir şekilde kapatıldı.")

# Programın başlangıç noktası
if __name__ == "__main__":
    tracker = VisionGuard(camera_id=0)

    tracker.start()
