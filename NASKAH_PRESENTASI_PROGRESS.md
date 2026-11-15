# Naskah Presentasi Progress: Penambahan Fitur Morfologi & Peningkatan Citra di TaksuCitra

> **Panduan Penggunaan Naskah:**
> - **Estimasi Durasi:** ± 15 Menit.
> - **Gaya Bahasa:** Santai, natural, mengalir (bukan hafalan kata-demi-kata), namun berbobot secara konsep dan terminologi ilmiah.
> - **Tanda Baca & Petunjuk Stage:**
>   - `[Klik menu / Tunjukkan layar]`: Instruksi gestur atau aksi demo di laptop/proyektor.
>   - `(jeda sejenak)`: Momen berhenti 1–2 detik untuk memberi penekanan atau membiarkan audiens mencerna konsep.
>   - *Teks miring dalam tanda kurung*: Petunjuk intonasi suara.

---

## 00:00 – 01:30 | 1. Pembuka & Hook (Santai, Hangat, Mengundang Rasa Ingin Tahu)

"Halo semuanya, selamat pagi / siang. Terima kasih sudah menyempatkan waktu.

Hari ini saya ingin membagikan update progress yang cukup signifikan untuk aplikasi web kita, **TaksuCitra**.

Sebelum masuk ke teknis, saya ingin cerita sedikit tentang latar belakangnya. Kita tahu di mata kuliah Pengolahan Citra, teori matematika itu kadang terlihat sangat abstrak di atas kertas—bicara soal konvolusi, transformasi himpunan Minkowski, hysteresis thresholding, atau adaptasi matriks kovarians. Nah, di TaksuCitra, fokus saya adalah bagaimana teori-teori fundamental ini nggak cuma berhenti jadi rumus di slide kuliah, tapi hidup dalam sebuah platform interaktif yang intuitif, responsif, dan punya nilai estetika.

Di sprint kali ini, saya baru saja menuntaskan implementasi **11 fitur utama** yang terbagi dalam dua pilar besar: 
1. **Morfologi Matematika Citra Biner**, dan
2. **Peningkatan serta Analisis Citra (Image Enhancement & Analysis)**, lengkap dengan pipeline terpadu dan toolkit evaluasi kualitas.

Semua fitur ini sudah terintegrasi utuh ke aplikasi web yang ada, tanpa merusak satu pun fitur lama, dan tentunya sudah lulus pengujian unit test otomatis."

---

## 01:30 – 03:00 | 2. Arsitektur & Keputusan Desain (Clean & Non-Intrusive)

*(Intonasi: Percaya diri, menunjukkan pemahaman arsitektur sistem)*

"Hal pertama yang saya lakukan sebelum menulis kode adalah audit sistem. Aplikasi kita berbasis **Flask di backend** dan **Vanilla JS modern di frontend** dengan tema gelap elegan bernuansa *Bali Modern*. 

Ada tiga prinsip utama yang saya pegang selama integrasi:

1. **Prinsip Separation of Concerns:**
   Semua core processing murni diisolasi di folder `processing/`—yaitu `morphology_service.py` dan `enhancement_service.py` yang mengadaptasi toolkit acuan kita. `app.py` murni bertugas sebagai router API, validasi payload, dan encoding/decoding Base64. Tidak ada logika matematika yang tercecer di file route.
2. **Reuse Design System & Zero Dependency Bloat:**
   Saya tidak menambahkan CSS framework baru atau library frontend yang berat. Semua menggunakan komponen yang sudah ada: *Before-After Interactive Slider*, kontrol slider dan dropdown dinamis, serta sistem warna berbasis variabel CSS `--accent` (emas) dan `--purple` (terracotta).
3. **Pemberdayaan User Experience (UX):**
   Setiap fitur baru sekarang dilengkapi **Blok Panduan Konsep Terpusat** di file `help_texts.js`. Polanya konsisten: menjelaskan *apa fungsinya*, *kapan cocok dipakai*, dan *apa peringatan atau batasannya*. Ditambah lagi, ada tombol **Reset Parameter** serta **Bar Rangkuman Parameter** di dekat hasil agar kita tahu persis nilai parameter apa yang menghasilkan gambar tersebut."

---

## 03:00 – 05:30 | 3. Pilar I: Morfologi Citra (Dilasi, Erosi, Opening, Closing)

`[Aksi: Klik accordion 'Morfologi Citra' di sidebar, buka panel Dilasi & Erosi]`

"Sekarang, mari kita bedah pilar pertama: **Morfologi Matematika**. 

Morfologi ini bekerja pada citra biner—jadi hitam dan putih. Tapi kita tahu, citra dari dunia nyata itu berwarna atau grayscale. Maka, langkah pertama yang krusial adalah **praproses biner**. Di backend, citra kita ubah ke grayscale, kita beri sedikit Gaussian blur untuk meredam noise frekuensi tinggi, lalu disegmentasi menggunakan **Otsu Automatic Thresholding**. 

Menariknya, ada fitur cerdas **Penebakan Polaritas Otomatis**: sistem secara matematis menghitung proporsi piksel putih. Jika piksel putih lebih dari 50%, sistem otomatis membalikkan polaritas (invert), karena secara konvensi morfologi, objek *foreground* harus berwarna putih dan latar belakang hitam.

`[Aksi: Tunjukkan slider Before-After Dilasi, lalu beralih ke Erosi]`

Dari situ, kita masuk ke empat operasi dasar:
- **Dilasi:** Secara matematis ini adalah *Minkowski Addition* antara objek citra $A$ dengan *Structuring Element* (SE) $B$. Efeknya: objek putih membesar, celah kecil tersambung. Cocok untuk menyambung garis putus atau menebalkan teks yang pudar. Kelemahannya? Objek yang berdekatan bisa saling menyatu.
- **Erosi:** Kebalikan dari dilasi, yaitu *Minkowski Subtraction*—piksel hanya bertahan jika seluruh bentuk SE masuk ke dalam objek. Fungsinya mengikis batas luar, memisahkan objek yang menempel tipis, dan membuang bintik derau kecil. Tapi ingat, seluruh ukuran objek pasti ikut menyusut.

`[Aksi: Buka panel Opening dan Closing]`

- **Opening (Buka):** Ini adalah erosi yang langsung diikuti dilasi dengan SE yang sama: $(A \ominus B) \oplus B$. Keistimewaannya luar biasa: ia membuang tonjolan kontur dan noise bintik putih (*salt noise*) **tanpa mengubah ukuran global objek**. Di UI, kita bahkan menampilkan citra selisih (*difference map*) untuk melihat berapa banyak piksel derau yang berhasil dieliminasi.
- **Closing (Tutup):** Kebalikannya, dilasi dulu baru erosi: $(A \oplus B) \ominus B$. Fungsinya menutup lubang hitam di dalam objek (*pepper noise*) dan menyambung retakan tipis tanpa membuat objek menggelembung permanen.

Pengguna bisa memilih tiga variasi bentuk SE: **Ellipse**, **Rectangle**, atau **Cross**, dengan ukuran ganjil dari 3 sampai 31 piksel."

---

## 05:30 – 08:00 | 4. Morfologi Tingkat Lanjut: Hit-or-Miss Transform & Skeletonization

`[Aksi: Buka panel Hit-or-Miss Transform]`

"Nah, ini dua fitur morfologi favorit saya karena konsep matematikanya sangat elegan.

Pertama, **Hit-or-Miss Transform (HMT)**.
Kalau dilasi dan erosi hanya melihat objek, HMT adalah pencocokan template biner eksak. HMT butuh kernel ternary:
- Elemen bernilai **+1 (Hit)**: piksel wajib berada di objek putih.
- Elemen bernilai **-1 (Miss)**: piksel wajib berada di latar belakang hitam.
- Elemen bernilai **0 (Don't Care)**: kita abaikan.

Formulanya: $\text{HMT}(A) = (A \ominus B_{hit}) \cap (A^c \ominus B_{miss})$.
Di aplikasi, saya menyediakan preset pencarian:
1. Mendeteksi **piksel terisolasi** (noise 1 piksel).
2. Mendeteksi **titik ujung garis (endpoints)** menggunakan 8 template orientasi rotasi.
3. Mencari **persegi solid berukuran tepat $n \times n$**.
4. Dan yang paling interaktif: **Custom 3x3 Kernel Editor**. Pengguna bisa langsung mengklik grid 3x3 di layar untuk mengatur status *Hit*, *Miss*, atau *Don't Care*, lalu melihat hasilnya secara instan ditandai dengan marker emas di atas citra asli!

`[Aksi: Pindah ke panel Skeletonization]`

Kedua, **Skeletonization & Analisis Topologi**.
Tujuannya adalah mereduksi objek biner padat menjadi kerangka setebal **1 piksel** tanpa memutus konektivitas aslinya. Sangat penting untuk analisis pembuluh darah medis, sidik jari, atau jaringan jalan.

Di sini saya menyediakan **4 algoritma berbeda**:
1. **Lantuejoul (Morfologis):** Berbasis erosi dan opening bersyarat. Sifatnya *lossless*—bisa direkonstruksi kembali 100% menjadi citra asli!
2. **Zhang-Suen Thinning:** Implementasi algoritma thinning manual berbasis kondisi tetangga 8-konektivitas.
3. **Lee Algorithm:** Menghasilkan garis tengah yang sangat mulus dan terhubung rapi.
4. **Medial Axis Transform:** Skeleton yang sekaligus membawa informasi jarak (*distance transform*).

Tantangan skeletonization adalah batas objek yang kasar sering menimbulkan 'cabang palsu' atau *spurs*. Solusinya, saya implementasikan **Pruning Algorithm**: cabang-cabang pendek di bawah panjang tertentu dipangkas, lalu kerangka utama ditumbuhkan kembali secara bersyarat.

Di hasil akhir, aplikasi langsung menandai secara visual: **titik ujung (endpoints) diberi warna merah**, dan **titik persimpangan (branches/junctions) diberi warna biru**. Jumlah masing-masing dihitung otomatis menggunakan *connected components*."

---

## 08:00 – 10:30 | 5. Pilar II: Kecerahan, Kontras, Geometri & Deteksi Tepi

`[Aksi: Beralih ke grup menu 'Peningkatan & Analisis', buka 'Kecerahan & Kontras']`

"Sekarang kita masuk ke pilar kedua: **Peningkatan dan Analisis Citra (Image Enhancement)**.

`[Tunjukkan Before-After Slider dan Histogram di layar]`

Fitur pertama adalah **Brightness & Contrast**. Di sini bukan sekadar slider biasa, tapi mencakup ragam teknik komputasi:
- **Linear Transformation:** Formula klasik $g(x,y) = \alpha \cdot f(x,y) + \beta$, di mana $\alpha$ mengatur kontras dan $\beta$ mengatur kecerahan.
- **Gamma Correction:** Koreksi non-linear $g = 255 \cdot (f/255)^\gamma$. Nilai $\gamma < 1$ mencerahkan area gelap (*shadow*) tanpa membuat area terang meledak (*burn out*).
- **Auto-Gamma:** Sistem menghitung rata-rata luminans citra, lalu secara adaptif memilih nilai $\gamma$ agar rata-rata luminans bergeser ke target yang diinginkan.
- **Contrast Stretching:** Merentangkan persentil bawah (1%) dan atas (99%) ke rentang penuh [0, 255] untuk mengeliminasi outlier.
- **Histogram Equalization & CLAHE:** Equalization global sering kali membuat noise latar belakang ikut meledak. Karena itu kita sertakan **CLAHE (Contrast Limited Adaptive Histogram Equalization)** yang membatasi penguatan kontras lokal per grid ubin (tile).

Setiap kali diproses, aplikasi menampilkan **Kartu Metrik Kualitas**: nilai rata-rata *Brightness*, *Contrast (RMS Standard Deviation)*, dan *Shannon Entropy*, disertai **Visualisasi Histogram Sebelum dan Sesudah** secara berdampingan.

`[Aksi: Buka panel Flip dan Rotasi]`

Selanjutnya untuk manipulasi geometri:
- **Flip / Mirror:** Mendukung Horizontal, Vertikal, dan Keduanya. Ini murni pengindeksan array NumPy (`[::-1]`), jadi sifatnya 100% *lossless*.
- **Rotasi Citra:** Ada opsi **Rot90** (lossless) dan **Sudut Bebas (-180° s.d. +180°)**. Untuk sudut bebas, saya menyertakan implementasi **Manual Inverse Mapping dengan Interpolasi Bilinear** untuk membuktikan bahwa kita tidak sekadar memanggil wrapper library, tapi memahami bagaimana intensitas piksel dipetakan balik dari koordinat tujuan ke koordinat sumber. Dilengkapi juga opsi **Perbesar Kanvas (Expand)** agar keempat sudut citra tidak terpotong keluar bingkai."

---

## 10:30 – 12:45 | 6. Deteksi Tepi, Pereduksian Derau (Denoise), dan Pipeline Terpadu

`[Aksi: Buka panel Deteksi Tepi]`

"Untuk **Deteksi Tepi (Edge Detection)**, aplikasi mencakup perbandingan turunan pertama dan kedua:
- **Operator Gradien Turunan Pertama:** Sobel, Prewitt, dan Scharr. Menggunakan konvolusi filter 2D untuk menghitung magnitudo $|G| = \sqrt{G_x^2 + G_y^2}$ dan arah sudut $\theta$.
- **Laplacian of Gaussian (LoG):** Turunan kedua untuk mendeteksi *zero-crossing*.
- **Canny Edge Detector Manual:** Dibangun lengkap dari nol sesuai teori murni: mulai dari Gaussian blur $\rightarrow$ gradien Sobel $\rightarrow$ **Non-Maximum Suppression (NMS)** dengan kuantisasi sudut ke 4 sektor (0°, 45°, 90°, 135°) untuk menipiskan tepi jadi 1 piksel $\rightarrow$ **Double Thresholding** $\rightarrow$ dan pelacakan tepi lemah lewat **Hysteresis 8-konektivitas**.
- Kita juga bandingkan dengan **Auto-Canny OpenCV** yang menghitung ambang batas adaptif dari nilai median citra, dan kita ukur kesesuaiannya dengan metrik **F1-Agreement Score**.

`[Aksi: Buka panel Reduksi Derau (Denoise)]`

Lalu bagaimana jika citra penuh noise? Masuklah modul **Pereduksian Derau (Noise Reduction)**.
Di sini kita sediakan 6 filter:
1. **Mean & Gaussian Filter:** Untuk derau halus berdistribusi normal.
2. **Median & Adaptive Median Filter:** Sangat sakti untuk mengatasi *salt-and-pepper noise*. Pada Adaptive Median, ukuran jendela membesar secara dinamis (3, 5, 7) hanya jika piksel terbukti merupakan impuls derau. Piksel yang bukan derau dipertahankan utuh, sehingga detail tekstur tidak hilang!
3. **Bilateral & Non-Local Means (NLM):** Mereduksi derau Gaussian dengan tetap menjaga ketajaman garis tepi (*edge-preserving*).

Fitur unggulan di panel ini:
- **Estimasi Derau Otomatis (Metode Immerkaer):** Menggunakan respon filter Laplacian untuk menebak standar deviasi derau $\sigma$, sehingga parameter filter bisa disetel otomatis tanpa tebak-tebakan.
- **Generator Noise Pengujian:** Pengguna bisa menginjeksi derau buatan (*Gaussian, Salt & Pepper, Speckle, Poisson*) untuk menguji ketangguhan filter secara langsung, lengkap dengan penghitungan metrik objektif **PSNR (Peak Signal-to-Noise Ratio)** dan **SSIM (Structural Similarity Index)**!

`[Aksi: Buka panel Pipeline Terpadu]`

Puncaknya ada di **Pipeline Peningkatan Terpadu**:
Pengguna bisa menjalankan rantai proses multi-tahap sekaligus: 
$$\text{Input Bernoise} \longrightarrow \text{Denoise (NLM)} \longrightarrow \text{Koreksi Kontras (CLAHE)} \longrightarrow \text{Deteksi Tepi (Canny)}$$
Aplikasi langsung membandingkan hasilnya dengan pendekatan *naive* (CLAHE langsung tanpa denoise), yang memperlihatkan dengan sangat gamblang mengapa pembersihan noise di tahap awal wajib dilakukan sebelum kontras dinaikkan."

---

## 12:45 – 14:00 | 7. Verifikasi Kualitas Kode, Testing, dan Aspek UI/UX

*(Intonasi: Serius, profesional, menegaskan reliability sistem)*

"Sebagai seorang software engineer, fitur yang canggih tidak ada artinya kalau sistemnya rapuh atau sulit digunakan.

Untuk memastikan kualitasnya:
1. **Automated Unit Testing:**
   Saya membuat test suite lengkap di `tests/test_morphology_and_enhancement.py` dengan **27 kasus uji unit**. Seluruh test berhasil lolos (100% OK). Ini memvalidasi:
   - Kesetaraan matematis fungsi manual vs OpenCV (erosi, dilasi, opening, closing, hit-or-miss, flip, rotasi, sobel).
   - Bukti rekonstruksi lossless pada skeleton Lantuejoul.
   - Peningkatan metrik PSNR pada adaptive median filter.
   - Ketahanan API terhadap *edge cases* seperti citra berukuran super kecil (3x3 piksel) dan penanganan request tanpa gambar agar tidak terjadi crash unhandled exception di server.
   - Dan yang terpenting: *regression testing* pada modul lama (seperti Matching Lab) tetap berjalan 100% aman.

2. **Konsistensi UI/UX:**
   - Navigasi menu tersusun rapi di sidebar accordion dengan ikon Material Symbols.
   - Ada fitur pencarian cepat lewat **Command Palette (Ctrl + K)**.
   - Setiap panel terhubung ke sistem **Riwayat Lokal (History Drawer)** sehingga hasil olah citra bisa langsung disimpan dan diekspor ke **Laporan PDF** resmi."

---

## 14:00 – 15:00 | 8. Kesimpulan & Penutup (Membuka Sesi Diskusi)

"Secara kesimpulan, penambahan fitur ini telah melengkapi platform **TaksuCitra** dengan kemampuan morfologi tingkat lanjut dan peningkatan citra yang komprehensif—baik dari sisi ketepatan formula matematis, kecepatan performa, hingga kemudahan eksplorasi bagi mahasiswa dan pengguna.

Tentu saja masih ada ruang untuk pengembangan lebih lanjut, misalnya menambahkan pemrosesan citra berukuran ultra-tinggi secara asynchronous menggunakan Web Worker atau background task queue jika nantinya di-deploy ke server multi-user.

Sekian pemaparan progress dari saya. Saya sangat terbuka untuk masukan, saran, atau pertanyaan dari rekan-rekan dan dosen pengampu. 

Terima kasih banyak atas perhatiannya!"

---

## Lampiran: Pertanyaan yang Sering Muncul (Q&A Cheat Sheet)

Jika dalam sesi tanya jawab muncul pertanyaan teknis mendalam, berikut contekan konsep kuncinya:

1. **Kenapa ukuran Structuring Element (SE) harus ganjil?**
   *Jawaban:* Agar SE memiliki piksel pusat (*origin / anchor point*) integer yang simetris dan definitif pada $( \lfloor k/2 \rfloor, \lfloor k/2 \rfloor )$. Jika genap, pusatnya berada di antara dua piksel sehingga terjadi pergeseran spasial (*spatial bias*) saat operasi morfologi dilakukan.
2. **Apa beda mendasar antara Histogram Equalization biasa dengan CLAHE?**
   *Jawaban:* Equalization biasa bersifat global—ia meratakan frekuensi kumulatif seluruh citra, sehingga area yang homogen (misal langit atau background gelap) akan over-amplified dan noise-nya terlihat sangat kasar. CLAHE membagi citra menjadi blok-blok lokal (tiles), menerapkan *clip limit* untuk memotong puncak histogram lokal sebelum redistribusi, sehingga kontras meningkat tanpa mengamplifikasi derau.
3. **Mengapa pada rotasi manual kita menggunakan *Inverse Mapping*, bukan *Forward Mapping*?**
   *Jawaban:* Jika menggunakan *Forward Mapping* (menghitung koordinat tujuan dari koordinat sumber), pembulatan bilangan pecahan sering menghasilkan 'lubang' (*holes / moiré artifacts*) di mana ada piksel di kanvas tujuan yang tidak kebagian nilai. Dengan *Inverse Mapping*, kita menelusuri setiap piksel di kanvas output lalu mencari posisinya di citra input menggunakan interpolasi bilinear, sehingga seluruh kanvas output dijamin terisi rapat dan mulus.
4. **Apa beda Canny manual dengan Sobel biasa?**
   *Jawaban:* Sobel hanya menghasilkan peta kekuatan gradien di mana tepinya masih tebal dan berbayang sesuai lebar objek. Canny menyempurnakannya dengan *Non-Maximum Suppression* untuk mencari puncak lokal tegak lurus arah gradien (menghasilkan tepi setipis 1 piksel) dan *Hysteresis Thresholding* untuk memastikan garis tepi yang samar tetap tersambung jika terhubung dengan tepi yang kuat.
