# NASKAH PRESENTASI PROGRESS: PENCOCOKAN CITRA 1-TO-N & SUARA DTW
**Mata Kuliah:** Image Processing Programming — Semester 5 (TaksuCitra Platform)  
**Estimasi Durasi:** 15 Menit  
**Gaya Penyampaian:** Percakapan natural, santai, percaya diri, lugas, dan kaya pemahaman konsep (*concept-driven*).

---

## PANDUAN PENGGUNAAN SKRIP
- Teks dalam tanda kurung siku tebal **`[AKSI / DEMO]`** adalah instruksi tindakan di layar/laptop.
- Teks dalam tanda kurung miring *(bicara santai, tersenyum, jeda)* adalah petunjuk intonasi dan bahasa tubuh.
- Durasi waktu di setiap bab dirancang pas untuk total 15 menit presentasi + transisi.

---

```
TIMELINE PRESENTASI (TOTAL ~15 MENIT)
┌────────────────────────────────────────────────────────────────────────┐
│ 00:00 - 02:00 (2') │ Pembuka, Konteks & Problem Statement               │
│ 02:00 - 05:30 (3.5')│ Konsep Citra 1-to-N, Ekstraksi Fitur & Threshold │
│ 05:30 - 08:00 (2.5')│ Live Demo: Pencocokan Citra 1-vs-Database          │
│ 08:00 - 11:30 (3.5')│ Konsep Audio DTW: MFCC, Warping Path & Objektif    │
│ 11:30 - 13:30 (2') │ Live Demo: Audio DTW 1-vs-1 & 1-to-N Database      │
│ 13:30 - 15:00 (1.5')│ Kesimpulan, Insight Konseptual & Tanya Jawab       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## BAB 1: PEMBUKA & KONTEKS (00:00 – 02:00)

*(Berdiri rileks, tatap audiens/dosen penguji dengan hangat, jangan terburu-buru)*

"Selamat pagi/siang Bapak/Ibu dosen dan rekan-rekan semua.  
Terima kasih banyak atas waktu dan kesempatannya hari ini.

Di pertemuan atau progres sebelumnya, sistem **TaksuCitra** kita sudah berhasil membedah berbagai macam operasi dasar hingga pencocokan satu-lawan-satu (*one-to-one image matching*). Kita sudah melihat bagaimana dua gambar dijejerkan dan dihitung kesamaannya.

Tapi jujur saja, kalau kita bicara kebutuhan industri atau implementasi nyata di lapangan—seperti sistem *Google Lens*, pencarian produk di e-commerce, atau absensi wajah biometrik—kasusnya hampir tidak pernah sesederhana membandingkan Gambar A dengan Gambar B saja.  
Tantangan aslinya adalah: **bagaimana kalau kita punya satu gambar uji (*query*), lalu sistem harus mencari kandidat yang paling mirip di dalam sebuah gudang database berisi puluhan, ratusan, bahkan ribuan gambar?**

Nah, di progres terbaru ini, saya berfokus mengembangkan dua modul utama yang menjawab tantangan skalabilitas dan perluasan modalitas sinyal:

1. **Pertama, Pencocokan Citra 1-ke-Banyak (*1-to-N Image Database Matching*)**: lengkap dengan arsitektur ekstraksi luring (*offline*), pencarian daring (*online*), penentuan nilai ambang (*thresholding*), serta kemampuan fleksibel bagi pengguna untuk mengunggah folder databasenya sendiri.
2. **Kedua, Pencocokan Suara berbasis Dynamic Time Warping (DTW)**: memperluas pemahaman kita dari sekuens spasial di citra ke sekuens temporal pada audio menggunakan koefisien MFCC.

Mari kita bahas konsep fundamental di balik modul citra terlebih dahulu."

---

## BAB 2: FUNDAMENTAL KONSEP CITRA 1-TO-N & THRESHOLDING (02:00 – 05:30)

*(Gunakan nada edukatif, artikulasi jelas saat menyebut istilah teknis)*

"Bapak/Ibu, ketika kita memindahkan paradigma dari 1-vs-1 ke 1-to-N, ada kesalahan arsitektur klasik yang sering terjadi: yaitu mengekstrak fitur dari seluruh gambar di database tepat saat pengguna menekan tombol 'Cari'.  
Kalau databasenya cuma 3 gambar, mungkin aman. Tapi kalau ada 100 gambar, sistem pasti *freeze* karena kompleksitas komputasinya membengkak drastis.

Oleh karena itu, di TaksuCitra saya membagi alur kerja backend ke dalam dua tahapan disiplin:

### 1. Tahap Luring (*Offline Feature Extraction & Indexing*)
Koleksi gambar di database diekstraksi representasi numeriknya terlebih dahulu di latar belakang. Baik itu deskriptor biner seperti **ORB**, deskriptor gradien float 128D seperti **SIFT**, distribusi warna 3D HSV, bentuk momen invarian Hu, sampai *deep embedding*.  
Seluruh representasi ini disimpan ke dalam struktur indeks atau *feature cache*. Jadi saat ada kueri baru masuk, database sudah dalam status 'siap tanding' tanpa perlu ekstraksi ulang.

### 2. Tahap Daring (*Online Query Matching*)
Saat pengguna mengunggah citra uji (*query image*), sistem hanya mengekstrak fitur dari satu citra uji tersebut secara *real-time*.  
Setelah vektor fiturnya didapat, sistem langsung menghitung jarak atau skor kemiripannya terhadap vektor-vektor yang sudah ada di indeks menggunakan metrik yang relevan:
- Jarak **Hamming** untuk deskriptor biner ORB,
- Jarak **Euclidean / $L_2$** untuk SIFT,
- Serta **Cosine Distance** untuk *deep embeddings*.

### Mengapa Nilai Ambang (*Thresholding*) itu Penentu Segalanya?
*(Beri penekanan di bagian ini)*  
Pencocokan fitur tidak akan ada artinya tanpa **Thresholding**.  
Dalam praktiknya, kita selalu berhadapan dengan *trade-off*:
- Kalau nilai ambang kita pasang terlalu ketat, sistem kita menderita **False Rejection**—gambar yang sebenarnya sama persis tapi cuma beda sedikit pencahayaan atau sudut foto akan ditolak mentah-mentah.
- Sebaliknya, kalau nilai ambang terlalu longgar, kita kena **False Match**—gambar yang sama sekali berbeda bisa dianggap cocok karena banyak *outliers* yang lolos.

Untuk mengatasi ini, di backend saya menerapkan kombinasi:
1. **Lowe’s Ratio Test**: Membandingkan jarak tetangga terdekat pertama ($d_1$) dan kedua ($d_2$). Jika rasio $\frac{d_1}{d_2} < 0.75$, baru kita anggap fitur itu unik dan tidak ambigu.
2. **Global Similarity Threshold**: Parameter slider di antarmuka yang memungkinkan pengguna menentukan batas ambang keputusan akhir: apakah kandidat tersebut sah dinyatakan sebagai **MATCH** atau **NON-MATCH**.

Biar tidak hanya teori, mari kita langsung lihat implementasinya di antarmuka."

---

## BAB 3: DEMONSTRASI LANGSUNG CITRA 1-TO-N (05:30 – 08:00)

**`[AKSI DEMO]`**:  
*(Buka browser di menu **Citra 1-vs-Database (§18)**)*

"Bisa kita lihat di layar, ini adalah tampilan antarmuka yang baru saja saya sempurnakan.

*(Arahkan kursor ke kartu Citra Uji)*  
Pertama, untuk slot **Citra Uji (Query)** di sebelah kiri: tata letaknya sekarang sudah terkunci rapi dengan rasio yang proporsional. Tidak ada lagi masalah gambar resolusi tinggi meluber atau merusak tombol di bawahnya seperti pada versi sebelumnya. Di sini kita bisa langsung *drag-and-drop* satu citra uji, misalnya logo atau objek tertentu.

**`[AKSI DEMO]`**:  
*(Pilih/unggah satu gambar uji, misalnya `db_taksu_logo.jpg`)*

"Kedua, ini fitur penting yang baru ditambahkan: **fleksibilitas database target**.  
Di bagian bawah, pengguna tidak dikunci hanya memakai database bawaan sistem. Pengguna punya tombol **'Pilih Folder Citra'** yang memanfaatkan kemampuan `webkitdirectory`.  
Artinya, kita cukup menunjuk satu folder di laptop kita yang berisi sekumpulan citra—misal file a.jpg, b.jpg, c.jpg, dan seterusnya. Sistem akan langsung membaca seluruh folder tersebut, menampilkan pratinjau strip thumbnail, dan mengindeksnya secara otomatis.

**`[AKSI DEMO]`**:  
*(Pilih metode fitur misal 'ORB', atur slider threshold di 50%, pilih Top-5, lalu klik tombol **'Cari di Database (1-to-N)'**)*

*(Tunggu ~0.5 detik hingga hasil muncul)*

"Bisa kita perhatikan:
1. Dalam waktu hanya beberapa puluh milidetik, sistem menampilkan **Galeri Top-K Hasil Database** di sebelah kanan.
2. Setiap kartu langsung diberi peringkat nomor 1, 2, 3, lengkap dengan nilai kemiripan persentase, jarak numerik, dan status badge hijau **`✓ MATCH`** atau merah **`✗ NON-MATCH`** berdasarkan apakah ia melampaui nilai ambang yang kita atur.
3. Dan kalau kita gulir ke bawah, sistem menyediakan **visualisasi korespondensi titik fitur (keypoints)** antara citra uji kita dengan kandidat peringkat pertama (Top-1), sehingga dosen atau pengguna bisa memverifikasi *kenapa* kedua gambar tersebut dinyatakan cocok secara geometris."

---

## BAB 4: PERPINDAHAN KE AUDIO & KONSEP DYNAMIC TIME WARPING (08:00 – 11:30)

*(Tersenyum santai, lakukan transisi konsep yang mulus)*

"Sekarang, mari kita melangkah ke modul kedua yang tidak kalah menarik: **Pencocokan Suara berbasis Dynamic Time Warping (DTW)**.

Bapak/Ibu mungkin bertanya: *'Lho, dari pemrosesan citra kok melompat ke audio?'*  
Secara konseptual, keduanya sebenarnya saudara kembar. Bedanya:
- Di citra, kita mencocokkan sekuens **spasial** (garis kontur, piksel, atau koordinat bentuk 2D).
- Di suara, kita mencocokkan sekuens **temporal** (sinyal gelombang suara terhadap dimensi waktu).

### Masalah Fundamental Sinyal Suara
Kalau kita membandingkan dua orang mengucapkan kata yang sama, misalnya kata **'Taksu'**:
- Orang pertama mengucapkannya dengan tempo cepat, durasinya hanya **0.4 detik**.
- Orang kedua mengucapkannya lebih santai dan berayun, durasinya **0.8 detik**.

Kalau kita menggunakan pengukuran jarak kaku seperti **Euclidean Distance** dengan teknik *zero-padding* atau *resampling* linier, skor kemiripannya pasti hancur total. Kenapa? Karena gelombang suara mengalami pergeseran fase dan peregangan waktu non-linier. Huruf 'k' di audio A tidak sejajar di detik yang sama dengan huruf 'k' di audio B.

### Di sinilah DTW Menjadi Pahlawan
Dynamic Time Warping bekerja dengan algoritma *dynamic programming*.  
DTW mencari **Optimal Warping Path** (jalur pelengkungan waktu optimal) pada matriks akumulasi biaya kumulatif:

$$D(i, j) = c(i, j) + \min \big( D(i-1, j), D(i, j-1), D(i-1, j-1) \big)$$

Artinya, waktu diperbolehkan untuk 'melar' atau 'memadat' secara lokal agar fitur akustik yang bersesuaian bisa saling bertemu dengan biaya penalti minimum!

### Ekstraksi Fitur Akustik: Kenapa MFCC?
Sinyal audio mentah tidak kita bandingkan per sampel amplitudo, melainkan kita pecah menjadi bingkai (*framing*) 25 milidetik dengan lompatan (*hop*) 10 milidetik.  
Lalu kita ekstraksi **13 koefisien MFCC (Mel-Frequency Cepstral Coefficients)**.  
MFCC ini memetakan spektrum daya audio ke skala **Mel**, yaitu skala frekuensi non-linier yang meniru cara telinga manusia membedakan nada vokal dan konsonan.

### Dua Objektif Utama di Antarmuka
Di aplikasi ini, saya merancang 2 skenario objektif:
1. **Keyword Spotting**: Mendeteksi kemiripan kata kunci (misal mencocokkan lafal kata 'buka', 'tutup', 'taksucitra' meski diucapkan dengan tempo cepat vs lambat).
2. **Speaker Verification (Biometrik Suara)**: Memvalidasi apakah yang berbicara adalah pengguna terdaftar berdasarkan sidik suara akustiknya (*voiceprint*)."

---

## BAB 5: DEMONSTRASI AUDIO DTW 1-VS-1 & 1-TO-N (11:30 – 13:30)

**`[AKSI DEMO]`**:  
*(Buka tab **Suara DTW Lab (§19)**)*

"Mari kita lihat implementasi antarmukanya. Sesuai revisi terbaru, antarmuka ini menyediakan fleksibilitas penuh baik untuk pengujian 1-vs-1 maupun pencarian database 1-ke-banyak.

### Demo 1: Mode A (Pencocokan 1-vs-1 Audio A vs Audio B)
**`[AKSI DEMO]`**:  
*(Klik tombol 'Mode A: 1-vs-1')*

"Di Mode A ini, tata letaknya dirancang simetris:
- Kotak Audio A di kiri dan Kotak Audio B di kanan.
- Pengguna bisa mengunggah file `.wav` sendiri untuk masing-masing slot, atau **bisa langsung merekam menggunakan mikrofon live**. Ada visualisasi gelombang suara (*live waveform canvas*) saat kita berbicara.

**`[AKSI DEMO]`**:  
*(Jalankan pencocokan dengan dua sampel yang temponya berbeda, misal `kw_taksucitra_normal.wav` vs `kw_taksucitra_cepat.wav`)*

"Lihat hasilnya:
1. Di layar muncul visualisasi **2D Heatmap Matriks Akumulasi DTW**.  
   Garis cyan yang meliuk diagonal dari kiri bawah ke kanan atas itu adalah **Optimal Warping Path**. Garis ini membuktikan bagaimana frame dari Audio A yang lambat dilengkungkan agar pas dengan Audio B yang cepat.
2. Di bawahnya ada grafik komparasi: terlihat jelas jarak DTW sangat rendah (artinya dinyatakan **MATCH**), sementara metode Euclidean linier menghasilkan nilai error yang jauh lebih tinggi."

### Demo 2: Mode B (Pencarian Audio Uji ke Folder Database)
**`[AKSI DEMO]`**:  
*(Klik kembali ke 'Mode B: Audio Uji vs Database Suara')*

"Sama persis seperti konsep citra tadi, di Mode B kita bisa merekam suara uji atau mengunggah 1 file suara, lalu pada bagian database target pengguna dapat mengklik **'Pilih Folder Audio'** untuk memuat folder berisi kumpulan rekaman kata atau suara pengguna lain.

Sistem akan mengekstrak matriks MFCC dari setiap audio di folder tersebut, menghitung jarak DTW ternormalisasi ke kueri uji, lalu memfilter dengan batas ambang threshold untuk mengeluarkan status keputusan: apakah kata kunci atau penutur tersebut terverifikasi di database."

---

## BAB 6: PENUTUP & SESI DISKUSI (13:30 – 15:00)

*(Tutup laptop/demo sejenak, berdiri tegak, tatap audiens dengan senyum percaya diri)*

"Bapak/Ibu dosen dan rekan-rekan,  
Sebagai kesimpulan dari progres yang telah saya kerjakan:

Benang merah dari seluruh sistem ini adalah pemahaman utuh mengenai siklus pengenalan pola modern:
$$\text{Input Sinyal} \longrightarrow \text{Ekstraksi Representasi} \longrightarrow \text{Pengukuran Jarak/Skor} \longrightarrow \text{Keputusan Nilai Ambang}$$

Baik di domain **spasial (citra)** maupun di domain **temporal (suara)**:
- Kita berhasil mengoptimalkan pencarian berskala besar melalui **arsitektur luring & daring**.
- Kita memberikan fleksibilitas nyata di mana **pengguna bebas membawa databasenya sendiri dalam bentuk folder**.
- Dan yang paling penting, kita menerapkan **penentuan nilai ambang (*thresholding*) yang terukur**, sehingga sistem tidak sekadar mengeluarkan angka abstrak, melainkan mampu mengambil keputusan biner yang objektif: apakah data uji tersebut valid sebagai **MATCH** atau **NON-MATCH**.

Demikian presentasi progres dari saya. Saya sangat terbuka untuk menerima saran, masukan, maupun pertanyaan dari Bapak/Ibu dosen sekalian.  
Terima kasih banyak!"

---

## TIPS MENGHADAPI TANYA JAWAB DOSEN (FAQ CHEAT SHEET)

Jika dosen mengajukan pertanyaan teknis mendalam, berikut contekan jawaban taktisnya:

1. **Tanya:** *"Kenapa kamu pakai MFCC bukan Spectrogram mentah saja untuk DTW audio?"*  
   **Jawab:** *"Spektrogram mentah masih membawa informasi redundan dan sangat sensitif terhadap derau amplitudo absolut. MFCC mereduksi dimensi tersebut menjadi 13 koefisien representatif yang menonjolkan selubung spektral vokal (formant) serta mensimulasikan kepekaan logaritmik koklea telinga manusia pada skala Mel."*

2. **Tanya:** *"Bagaimana kompleksitas komputasi DTW kalau databasenya ada 10.000 audio?"*  
   **Jawab:** *"Kompleksitas DTW standar adalah $O(N \times M)$ untuk setiap pasangan. Untuk skala ribuan audio, kita bisa menambahkan optimasi Sakoe-Chiba Band untuk membatasi jendela pencarian jalur secara lokal, atau melakukan coarse-to-fine filtering dengan FastDTW / embedding clustering terlebih dahulu."*

3. **Tanya:** *"Apa bedanya Lowe Ratio Test dengan Global Threshold pada pencocokan citra?"*  
   **Jawab:** *"Lowe Ratio Test bekerja di level lokal per-titik fitur untuk mengeliminasi titik ambigu yang mirip di banyak tempat (tekstur berulang). Sedangkan Global Threshold bekerja di level keputusan akhir citra untuk menentukan apakah jumlah inliers atau skor kesamaan total sudah cukup untuk menyatakan kedua gambar tersebut identik."*
