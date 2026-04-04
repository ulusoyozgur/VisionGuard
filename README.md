# VisionGuard AI 👁️‍🗨️

<div align="center">

### 🔍 Gerçek zamanlı yüz tespiti · 🧠 Duygu analizi · 🎯 Kişi tanıma

<br>

![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![DeepFace](https://img.shields.io/badge/DeepFace-FF4B4B?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

</div>

---

## 💡 Hakkında

VisionGuard AI, Python ve OpenCV üzerine inşa edilmiş gerçek zamanlı bir **Computer Vision** sistemidir. Haar Cascade algoritması ile anlık yüz tespiti yaparken, DeepFace derin öğrenme modeli ile duygu durumu, yaş ve cinsiyet analizi gerçekleştirir. Her yüze benzersiz bir takip kimliği atanarak birden fazla kişi aynı anda izlenebilir.

Proje, v1'deki temel yüz tespitinden; çok katmanlı AI analizi, kişi tanıma ve video kaydına uzanan bir yolculukla v3.0'a evrildi.

---

## ✨ Özellikler

### 🏗️ Temel Sistem
- 🎥 Haar Cascade ile gerçek zamanlı yüz ve göz tespiti
- 🧩 Nesne Yönelimli mimari — kolayca genişletilebilir sınıf yapısı
- ⚡ Gri tonlama ön işleme ile minimize edilmiş CPU kullanımı
- 🛡️ Kapsamlı hata yönetimi ve profesyonel loglama sistemi
- 🪞 Ayna modu — görüntü yatayda çevrilerek doğal kullanım sağlanır

### 🤖 AI Katmanı
- 😄 **Duygu analizi** — mutlu, üzgün, sinirli, şaşkın, nötr ve daha fazlası
- 🎂 **Yaş & cinsiyet tahmini** — DeepFace modeli ile anlık tahmin
- 🪪 **Kişi tanıma** — kayıtlı kişileri otomatik olarak tespit eder
- 🔢 **Çok nesne takibi** — Centroid Tracker ile her yüze benzersiz ID

### 🎮 Kullanıcı Özellikleri
- 📸 `s` tuşu ile ekran görüntüsü alma
- 💾 `r` tuşu ile yüz kaydetme ve kişi tanıma sistemi
- ⚙️ Config dosyası ile kod değiştirmeden ayar yapma
- 🎬 İsteğe bağlı mp4 video kaydı

---

## 🏛️ Mimari

DeepFace analizi, ana kamera döngüsünü bloke etmemek için **ayrı bir thread**'de çalışır. Bu tasarım sayesinde görüntü akışı akıcı kalırken arka planda AI modelleri sessizce çalışmaya devam eder.

```
📷 Kamera Akışı
    │
    ▼
🔧 Ön İşleme (gri tonlama + ayna efekti)
    │
    ▼
🔍 Haar Cascade ──► Yüz & Göz Tespiti
    │
    ▼
📍 Centroid Tracker ──► Benzersiz ID Atama
    │
    ├──► 🖥️  Ana Thread: Ekrana çizim + klavye kontrolü
    │
    └──► 🧠 Arka Plan Thread (her 15 frame'de bir)
              │
              ├── 😄 Duygu analizi
              ├── 🎂 Yaş & cinsiyet tahmini
              └── 🪪 Kişi tanıma
```

---

## 🚀 Kurulum

### 📋 Gereksinimler

- Python 3.10 veya üzeri
- Web kamerası

### 📦 Bağımlılıkları Yükle

```bash
pip install deepface opencv-python numpy tf-keras
```

> ⚠️ İlk çalıştırmada DeepFace, AI modellerini otomatik olarak indirir (~500MB). Bu işlem yalnızca bir kez gerçekleşir.

---

## 🎯 Kullanım

```bash
python visionguard.py
```

### ⌨️ Klavye Kısayolları

| Tuş | İşlev |
|-----|-------|
| `s` | 📸 Ekran görüntüsü al |
| `r` | 💾 Kameradaki yüzü kaydet |
| `q` | 🚪 Programı kapat |

### 🪪 Kişi Tanıma

`r` tuşuna bastıktan sonra terminale isim girin:

```
Kişi adı girin (boşluk yerine _ kullanın): Ozgur_Ulusoy
```

Yüz `known_faces/` klasörüne kaydedilir. Sistem bir sonraki açılışta bu kişiyi otomatik olarak tanır ve kutunun rengi 🟡 sarıya döner.

---

## ⚙️ Yapılandırma

Tüm ayarlar `visionguard_config.json` dosyasında tutulur. Kod değiştirmeden sistemi özelleştirebilirsiniz:

```json
{
  "ai": {
    "emotion": true,
    "age_gender": true,
    "analyze_every_n_frames": 15,
    "recognition": true
  },
  "recording": {
    "enabled": false
  }
}
```

> 💡 `analyze_every_n_frames` değerini artırarak (örn. `30`) performansı iyileştirebilirsiniz.

---

## 📁 Proje Yapısı

```
visionguard-ai/
├── 🐍 visionguard.py              # Ana uygulama
├── ⚙️  visionguard_config.json     # Yapılandırma (otomatik oluşur)
├── 🪪 known_faces/                # Kayıtlı yüzler (otomatik oluşur)
├── 📸 screenshots/                # Ekran görüntüleri (otomatik oluşur)
├── 🎬 recordings/                 # Video kayıtları (otomatik oluşur)
├── 📋 logs/                       # Log dosyaları (otomatik oluşur)
└── 📄 README.md
```

---

## 🛠️ Teknoloji Yığını

| Teknoloji | Kullanım Amacı |
|-----------|----------------|
| 🐍 Python 3 | Ana dil |
| 📷 OpenCV | Kamera akışı ve görüntü işleme |
| 🤖 DeepFace | Duygu, yaş, cinsiyet, kişi tanıma |
| 🔶 TensorFlow / Keras | DeepFace altyapısı |
| 🔢 NumPy | Centroid hesaplamaları |
| 🧵 threading | Arka plan AI analizi |

---

## 📌 Sürüm Geçmişi

| Sürüm | Özellikler |
|-------|------------|
| ⚡ v1.0 | Temel yüz tespiti, Haar Cascade, loglama |
| 🚀 v2.0 | Göz tespiti, FPS sayacı, ekran görüntüsü, video kaydı, config sistemi |
| 🧠 v3.0 | DeepFace AI entegrasyonu, duygu/yaş/cinsiyet, kişi tanıma, çok nesne takibi |

---

<div align="center">

**Özgür Ulusoy** · Software Engineering Student 

[![GitHub](https://img.shields.io/badge/GitHub-ulusoyozgur-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ulusoyozgur)
[![Website](https://img.shields.io/badge/Website-ulusoyozgur.me-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://ulusoyozgur.vercel.app)

<br>

⭐ Projeyi beğendiyseniz star atmayı unutmayın!

</div>
