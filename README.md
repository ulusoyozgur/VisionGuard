# VisionGuard AI 👁️

Bu proje, Python ve OpenCV kullanılarak geliştirilmiş gerçek zamanlı bir bilgisayarlı görü (Computer Vision) uygulamasıdır. Kamera üzerinden alınan anlık görüntüleri işleyerek Haar Cascade algoritmalarıyla yüz tespiti ve takibi yapar.

## 🚀 Özellikler

- **Gerçek Zamanlı Takip:** Kameradan alınan her kare anlık olarak analiz edilir.
- **Düşük İşlemci Tüketimi:** Görüntüler işlenmeden önce gri tonlamaya (grayscale) çevrilerek performans optimize edilmiştir.
- **Hata Yönetimi ve Loglama:** Sistem, kamera bağlantısı koptuğunda veya model yüklenemediğinde çökmek yerine profesyonel log kayıtları tutarak güvenli çıkış yapar.
- **Nesne Yönelimli Tasarım (OOP):** Kod, temiz ve ölçeklenebilir olması için sınıf (`class`) yapısında yazılmıştır.

## 🛠️ Kurulum ve Kullanım

1. Gerekli kütüphaneyi bilgisayarınıza kurun:
```bash
pip install opencv-python
