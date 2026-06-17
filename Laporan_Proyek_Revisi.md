# Laporan Proyek Machine Learning
## Judul: Sistem Rekomendasi Destinasi Wisata Jawa Barat Menggunakan TF-IDF dan Cosine Similarity

## Project Overview
Pariwisata merupakan salah satu sektor yang berkontribusi terhadap pertumbuhan ekonomi daerah. Jawa Barat memiliki banyak destinasi wisata dengan karakteristik yang berbeda-beda, mulai dari wisata alam, wisata budaya, hingga wisata rekreasi. Banyaknya pilihan tersebut sering membuat wisatawan kesulitan menentukan destinasi yang sesuai dengan minatnya.

Untuk mengatasi permasalahan tersebut, dikembangkan sistem rekomendasi destinasi wisata berbasis Machine Learning menggunakan pendekatan Content-Based Filtering. Sistem memanfaatkan informasi kategori dan deskripsi wisata untuk mencari destinasi yang memiliki karakteristik serupa dengan wisata yang dipilih pengguna.

## Business Understanding

### Problem Statements
1. Bagaimana membantu wisatawan menemukan destinasi wisata yang sesuai dengan preferensinya?
2. Bagaimana memanfaatkan kategori dan deskripsi wisata untuk menghasilkan rekomendasi yang relevan?
3. Bagaimana membangun sistem rekomendasi yang sederhana namun efektif?

### Goals
1. Membangun sistem rekomendasi wisata Jawa Barat.
2. Menghasilkan rekomendasi berdasarkan kemiripan karakteristik destinasi.
3. Mengevaluasi kualitas rekomendasi yang dihasilkan.

### Solution Statements
1. Menggunakan TF-IDF untuk mengubah data teks menjadi representasi numerik.
2. Menggunakan Cosine Similarity untuk mengukur kemiripan antar destinasi.
3. Mengimplementasikan Content-Based Filtering sebagai metode rekomendasi.

## Data Understanding

### Dataset
Dataset yang digunakan adalah **wisata_indonesia_clean.csv** yang berisi informasi destinasi wisata di Indonesia.

### Variabel Dataset
| Variabel | Keterangan |
|-----------|------------|
| nama_wisata | Nama destinasi wisata |
| kategori | Kategori wisata |
| deskripsi_bersih | Deskripsi wisata |
| provinsi | Lokasi provinsi |
| kota_kabupaten | Lokasi kota/kabupaten |
| alamat | Alamat wisata |

### Tahapan Data Understanding
1. Memuat dataset menggunakan Pandas.
2. Melihat jumlah data dan contoh data awal.
3. Memeriksa struktur data menggunakan `info()`.
4. Memeriksa missing value menggunakan `isnull().sum()`.
5. Memfilter data agar hanya menggunakan destinasi wisata Jawa Barat.

## Data Preparation

### 1. Filtering Data Jawa Barat
Data difilter berdasarkan kolom `provinsi` sehingga hanya destinasi wisata Jawa Barat yang digunakan.

### 2. Kategorisasi Wisata
Kategori wisata dikelompokkan menjadi:
- Wisata Alam
- Wisata Budaya
- Wisata Rekreasi

Tujuannya agar proses evaluasi dan rekomendasi lebih terstruktur.

### 3. Feature Engineering
Dibuat fitur baru bernama `content` yang menggabungkan kategori dan deskripsi wisata.

Contoh:

```
Gunung wisata pendakian dengan pemandangan alam indah
```

### 4. Text Cleaning
Proses pembersihan teks meliputi:
- Mengubah huruf menjadi lowercase.
- Menghapus angka.
- Menghapus karakter khusus.
- Menghapus spasi berlebih.

Contoh:

Input:
```
Pantai Indah!!! 2025
```

Output:
```
pantai indah
```

## Modeling

### Content-Based Filtering
Metode rekomendasi yang digunakan adalah Content-Based Filtering. Sistem akan mencari wisata yang memiliki karakteristik serupa berdasarkan informasi tekstual.

### TF-IDF Vectorizer
TF-IDF digunakan untuk mengubah teks menjadi vektor numerik sehingga dapat diproses oleh algoritma machine learning.

Parameter yang digunakan:

```python
TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)
```

Output yang dihasilkan berupa matriks TF-IDF yang merepresentasikan setiap destinasi wisata.

### Cosine Similarity
Cosine Similarity digunakan untuk menghitung tingkat kemiripan antar destinasi wisata berdasarkan matriks TF-IDF.

Interpretasi nilai:

| Nilai | Keterangan |
|---------|-----------|
| 0 | Tidak mirip |
| 1 | Sangat mirip |

### Sistem Rekomendasi
Tahapan rekomendasi:

1. Pengguna memasukkan nama wisata.
2. Sistem mencari indeks wisata.
3. Sistem menghitung kemiripan terhadap seluruh wisata lain.
4. Sistem mengurutkan nilai similarity.
5. Sistem menampilkan Top-N rekomendasi wisata yang paling mirip.

Contoh:

Input:
```
Gunung Gede
```

Output:
```
1. Gunung Papandayan
2. Gunung Ciremai
3. Gunung Galunggung
...
```

## Evaluation

### Precision@10
Evaluasi dilakukan menggunakan Precision@10.

Rumus:

```
Precision@10 =
Jumlah Rekomendasi Relevan / 10
```

Rekomendasi dianggap relevan apabila memiliki kategori yang sama dengan wisata acuan.

Semakin tinggi nilai Precision@10, semakin baik kualitas rekomendasi yang dihasilkan.

## Deployment

Model yang telah dibangun disimpan menggunakan Pickle:

```python
pickle.dump(tfidf, open('tfidf_model.pkl', 'wb'))
pickle.dump(cosine_sim, open('cosine_similarity.pkl', 'wb'))
```

File yang dihasilkan:

1. tfidf_model.pkl
2. cosine_similarity.pkl

## Hasil Visualisasi

### Distribusi Kategori Wisata
Visualisasi digunakan untuk melihat persebaran kategori wisata yang digunakan dalam penelitian.

### Distribusi Hasil Rekomendasi
Visualisasi digunakan untuk melihat kecenderungan kategori wisata yang direkomendasikan oleh sistem.

## Kesimpulan
Berhasil dibangun sistem rekomendasi destinasi wisata Jawa Barat menggunakan metode Content-Based Filtering dengan TF-IDF dan Cosine Similarity.

TF-IDF digunakan untuk merepresentasikan data teks dalam bentuk numerik, sedangkan Cosine Similarity digunakan untuk mengukur tingkat kemiripan antar destinasi wisata. Sistem mampu menghasilkan rekomendasi wisata yang relevan berdasarkan karakteristik wisata yang dipilih pengguna dan dapat digunakan sebagai solusi sederhana dalam membantu wisatawan menemukan destinasi wisata yang sesuai.
