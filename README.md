# VisionGuard AI 👁️

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

VisionGuard AI, Python ve OpenCV kütüphanesi kullanılarak geliştirilmiş, gerçek zamanlı bir bilgisayarlı görü (Computer Vision) projesidir. Haar Cascade algoritmalarını kullanarak insan yüzünü anlık olarak tespit eder ve takip altına alır.

---

## ✨ Öne Çıkan Özellikler

- **Nesne Yönelimli Mimari (OOP):** Kod yapısı tamamen sınıflar (classes) üzerine kuruludur, bu sayede kolayca genişletilebilir.
- **Yüksek Performans:** Görüntüler işlenmeden önce gri tonlamaya çevrilerek CPU kullanımı minimize edilmiştir.
- **Profesyonel Loglama:** `logging` kütüphanesi ile sistemin her adımı (başlatma, hata, kapanış) terminal üzerinden takip edilebilir.
- **Hata Yönetimi:** Kamera bağlantısı kesilmesi veya model dosyası eksikliği gibi durumlarda sistem güvenli bir şekilde kapanır.

## 🛠️ Teknik Detaylar

Sistem, yüz tespiti için OpenCV'nin sunduğu **Haar Cascade Classifier** modelini kullanır. Tespit edilen koordinatlar üzerine dinamik olarak yeşil bir takip çerçevesi ve durum bilgisi ekler.

## 🚀 Kurulum ve Çalıştırma

### 1. Ön Gereksinimler
Bilgisayarınızda Python yüklü olmalıdır. Ardından gerekli kütüphaneyi şu komutla yükleyin:

```bash
pip install opencv-python
