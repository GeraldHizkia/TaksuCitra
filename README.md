# TaksuCitra

Platform demo Image Processing Programming (Semester 5) — Bab 2 (Operasi Piksel,
Konversi Gambar, Operasi Histogram), AI Lab, Advanced Ops, Batch/Webcam/Report,
dan Feature Extraction Lab (27 fitur Sesi 3 + Matching Engine + Concat Playground).

## Menjalankan

```bash
pip install -r requirements.txt
python app.py
```

Buka `http://localhost:5000`.

## Model AI Lab (opsional)

Fitur **Deteksi Wajah**, **Super Resolution**, dan **Deteksi Objek** butuh file
model yang TIDAK disertakan (ukurannya besar). Tanpa file ini, ketiga fitur
tersebut akan menampilkan pesan error yang jelas ("Model AI belum tersedia"),
bukan crash — semua fitur lain tetap berjalan normal.

Taruh file berikut untuk mengaktifkannya:

```
models/face/deploy.prototxt
models/face/res10_300x300_ssd_iter_140000.caffemodel
models/superres/ESPCN_x2.pb
models/superres/ESPCN_x3.pb
models/superres/ESPCN_x4.pb
models/detection/MobileNetSSD_deploy.prototxt
models/detection/MobileNetSSD_deploy.caffemodel
```

Semuanya adalah model publik standar (SSD ResNet10 face detector, ESPCN
super-resolution, MobileNet-SSD Pascal VOC) yang mudah dicari di repo OpenCV
atau model zoo masing-masing.

## Sample gambar

`static/samples/` berisi 4 gambar sintetis (bentuk geometris, wajah sederhana,
pola garis, dan tekstur noise) untuk demo cepat. Ganti/tambah sesuai kebutuhan.

## Struktur

- `processing/` — seluruh logic pemrosesan citra (tidak ada logic di `app.py`,
  murni routing + encode/decode base64).
- `static/js/panels.js` — definisi tiap panel (field, endpoint, render).
- `static/js/main.js` — mesin generik render panel + beberapa custom UI
  (mask-draw, multi-upload, dual-upload wajib, concat playground, dst).
- `templates/index.html` — sidebar navigasi (data-panel harus cocok persis
  dengan key di `panels.js` dan `FE_PIPELINE_MAP`/route di `app.py`).

## Status pengujian

Seluruh route (piksel, konversi, histogram, advanced ops, batch, webcam,
report, 27+5 Feature Extraction Lab) sudah diuji otomatis lewat Flask test
client dan lolos. Seluruh 64 panel (kecuali live webcam, yang butuh kamera
nyata) sudah disimulasikan lewat jsdom — render, isi parameter, klik proses,
hasil tampil — tanpa error JavaScript. Tiga fitur AI Lab yang butuh model
eksternal (face-blur, super-resolution, object-detection) sudah diverifikasi
gagal-dengan-anggun (pesan jelas, bukan crash) karena file model memang belum
disertakan di paket ini.
