# 🛣️ Deteksi Kerusakan Jalan — Streamlit App

Aplikasi web **Computer Vision (YOLO)** untuk mendeteksi kerusakan permukaan
jalan: **lubang (pothole)**, **retakan (crack)**, dan **manhole**.
Mendukung **SDG 9: Industri, Inovasi & Infrastruktur** untuk infrastruktur pintar.

Dibuat oleh **VINIX7 — Divisi AI**.

---

## 📦 Struktur Folder

```
road-damage-app/
├── app.py                  # aplikasi Streamlit utama
├── requirements.txt        # dependensi Python
├── README.md               # dokumen ini
├── .gitignore
├── .streamlit/
│   └── config.toml         # tema warna (navy-teal minimalis)
├── models/
│   ├── README.md
│   └── best.pt             # ⬅️ letakkan bobot model di sini (dari notebook)
└── sample_images/          # gambar contoh untuk demo
    ├── sample1.jpg
    ├── sample2.jpg
    └── sample3.jpg
```

---

## 🚀 Cara Menjalankan (Lokal)

**1. Latih model** menggunakan notebook `road_damage_yolo_training.ipynb`
(di Kaggle / Google Colab dengan GPU). Setelah selesai, unduh file
`models/best.pt` yang dihasilkan.

**2. Letakkan** `best.pt` ke dalam folder `models/`.

**3. Install & jalankan:**

```bash
cd road-damage-app
python -m venv venv && source venv/bin/activate   # opsional
pip install -r requirements.txt
streamlit run app.py
```

Buka browser ke `http://localhost:8501`.

---

## ☁️ Deploy ke Streamlit Community Cloud (Gratis)

1. Push folder ini ke sebuah repositori **GitHub**.
2. Buka [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pilih repo, branch, dan set **Main file path** ke `app.py`.
4. Karena `best.pt` besar & di-*ignore* git, unggah lewat salah satu cara:
   - **Git LFS**, atau
   - simpan sebagai **GitHub Release** lalu unduh saat runtime, atau
   - commit langsung bila ukurannya kecil (< 100 MB).
5. Klik **Deploy**.

---

## 🎛️ Fitur

- Unggah gambar sendiri **atau** pakai gambar contoh sekali klik.
- Slider **confidence** & **IoU** yang bisa diatur.
- **Filter kelas** (tampilkan/sembunyikan lubang, retakan, manhole).
- Perbandingan **gambar asli vs hasil deteksi** berdampingan.
- Kartu ringkasan jumlah per kelas + tabel detail confidence.
- Tombol **unduh** gambar hasil deteksi.

---

## 🧠 Kelas Model

| ID | Kelas | Keterangan |
|----|-------|------------|
| 0 | Pothole | Lubang jalan |
| 1 | Crack | Retakan |
| 2 | Manhole | Tutup gorong-gorong |

**Dataset:** [Road Damage Dataset — Kaggle](https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes)
(2009 gambar, 3 kelas).
