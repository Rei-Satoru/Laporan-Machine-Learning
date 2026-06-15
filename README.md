# Laporan Proyek Machine Learning

## Judul: Sistem Rekomendasi Destinasi Wisata Jawa Barat

### Nama : Rayhan Muhammad Syawal
### NIM : 2330511063
### Kelas : 6B

## Project Overview
Pariwisata merupakan salah satu sektor penting yang berkontribusi terhadap pertumbuhan ekonomi daerah. Provinsi Jawa Barat memiliki beragam destinasi wisata yang mencakup wisata alam, wisata budaya, dan wisata rekreasi. Banyaknya pilihan destinasi wisata sering kali membuat wisatawan kesulitan menemukan tempat yang sesuai dengan minat mereka.
Untuk mengatasi permasalahan tersebut, dikembangkan sebuah Sistem Rekomendasi Destinasi Wisata Jawa Barat berbasis Machine Learning. Sistem ini menggunakan pendekatan **Content-Based Filtering** dengan memanfaatkan informasi kategori dan deskripsi destinasi wisata. Melalui sistem rekomendasi ini, pengguna dapat memperoleh rekomendasi tempat wisata yang memiliki karakteristik serupa dengan destinasi yang disukai sebelumnya.

### Manfaat Proyek:
1. Membantu wisatawan menemukan destinasi wisata di Jawa Barat yang sesuai dengan minatnya.
2. Mempermudah proses pencarian tempat wisata tanpa harus menelusuri seluruh daftar destinasi yang tersedia.
3. Memberikan rekomendasi destinasi wisata yang relevan berdasarkan karakteristik wisata yang dipilih pengguna.
4. Menjadi contoh implementasi Machine Learning khususnya sistem rekomendasi berbasis Content-Based Filtering pada sektor pariwisata.
5. Mendukung promosi destinasi wisata Jawa Barat melalui pemanfaatan teknologi rekomendasi.

## Business Understanding
### Problem Statements:
1. Bagaimana membantu wisatawan menemukan destinasi wisata yang sesuai dengan preferensinya?
2. Bagaimana memanfaatkan informasi kategori dan deskripsi wisata untuk menghasilkan rekomendasi yang relevan?
3. Bagaimana membangun sistem rekomendasi wisata menggunakan pendekatan Machine Learning yang sederhana namun efektif?
### Goals:
1. Membangun sistem rekomendasi destinasi wisata Jawa Barat.
2. Memberikan rekomendasi wisata berdasarkan kemiripan karakteristik destinasi.
3. Mengevaluasi kualitas rekomendasi menggunakan metrik Precision@10.
### Solution Statements:
Pendekatan yang digunakan pada proyek ini adalah:
1. Menggunakan **TF-IDF Vectorizer** untuk mengubah data teks menjadi representasi numerik.
2. Menggunakan **Cosine Similarity** untuk mengukur tingkat kemiripan antar destinasi wisata.
3. Mengimplementasikan metode **Content-Based Filtering** untuk menghasilkan rekomendasi wisata yang relevan.

## Data Understanding
### Informasi Dataset
Dataset yang digunakan: **wisata_indonesia_clean.csv**
Dataset berisi informasi destinasi wisata di Indonesia yang kemudian difilter sehingga hanya menyisakan data destinasi wisata yang berada di Provinsi Jawa Barat.
Variabel Dataset
|  Variabel	        |  Keterangan                    |
|-------------------|--------------------------------|
|  nama_wisata      |  Nama destinasi wisata         |
|  kategori	        |  Kategori wisata               |
|  deskripsi_bersih	|  Deskripsi destinasi wisata    |
|  provinsi	        |  Provinsi lokasi wisata        |
|  kota_kabupaten	|  Kota/Kabupaten lokasi wisata  |
|  alamat	        |  Alamat destinasi wisata       |
### Dataset Understanding Process
Tahapan yang dilakukan:
1. Memeriksa jumlah data.
2. Memeriksa tipe data menggunakan `info()`.
3. Memeriksa missing value menggunakan `isnull().sum()`.
4. Memfilter data berdasarkan provinsi Jawa Barat.
### Hasil Filtering
Dataset awal berisi data wisata dari seluruh Indonesia. Setelah proses filtering, hanya data wisata yang berasal dari Provinsi Jawa Barat yang digunakan dalam proses pemodelan.

## Data Preparation
Tahapan persiapan data yang dilakukan sebagai berikut.
### 1. Filtering Data Jawa Barat
Data difilter menggunakan kolom `provinsi` sehingga hanya data wisata dari Jawa Barat yang digunakan.
```
jabar = df[
    df['provinsi'].str.contains(
        'Jawa Barat',
        case=False,
        na=False
    )
].copy()
```
### 2. Kategori Wisata
Kategori wisata dikelompokkan menjadi tiga kategori utama.
|  Kategori Awal	                                    |  Kategori Baru    |
|-------------------------------------------------------|-------------------|
|  gunung, curug, pantai, danau, bukit, hutan, geopark	|  Wisata Alam      |
|  museum, sejarah, budaya	                            |  Wisata Budaya    |
|  lainnya	                                            |  Wisata Rekreasi  |
### 3. Feature Engineering
Membuat fitur baru bernama `content` yang merupakan gabungan antara kategori dan deskripsi wisata.
```
jabar['content'] = (
    jabar['kategori'].fillna('')
    + ' ' +
    jabar['deskripsi_bersih'].fillna('')
)
```
### 4. Text Cleaning
Tahapan preprocessing teks meliputi:
A. Mengubah teks menjadi lowercase.
B. Menghapus karakter selain huruf.
C. Menghapus spasi berlebih.
Contoh:
`Pantai Indah!!! 2025`
Menjadi:
`pantai indah`

## Modeling
### Content-Based Filtering
Sistem rekomendasi dibangun menggunakan metode **Content-Based Filtering** yang memanfaatkan informasi deskripsi dan kategori wisata.
### TF-IDF Vectorizer
**TF-IDF** digunakan untuk mengubah data teks menjadi vektor numerik.
Parameter yang digunakan:
```
TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)
```
Keunggulan TF-IDF:
1. Memberikan bobot lebih tinggi pada kata yang penting.
2. Mengurangi pengaruh kata yang terlalu sering muncul.
3. Cocok digunakan pada sistem rekomendasi berbasis teks.
### Cosine Similarity
Setelah data direpresentasikan dalam bentuk vektor TF-IDF, dilakukan perhitungan tingkat kemiripan menggunakan Cosine Similarity.
Rumus:
```
[
Similarity(A,B)=\frac{A \cdot B}{||A|| \times ||B||}
]
```
Nilai similarity:
| Nilai	| Interpretasi |
|-------|--------------|
| 0	    | Tidak mirip  |
| 1	    | Sangat mirip |

Proses Rekomendasi:
1. Pengguna memilih destinasi wisata.
2. Sistem mengambil indeks wisata tersebut.
3. Menghitung nilai cosine similarity terhadap seluruh wisata lain.
4. Mengurutkan nilai similarity dari terbesar ke terkecil.
5. Menampilkan Top-10 destinasi wisata yang paling mirip.

## Evaluation


## Deployment

