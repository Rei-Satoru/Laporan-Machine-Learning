# Laporan Proyek Machine Learning
## Judul: Sistem Rekomendasi Destinasi Wisata Jawa Barat Menggunakan TF-IDF dan Cosine Similarity

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
Dataset yang digunakan: [**wisata_indonesia_clean.csv**](https://www.kaggle.com/datasets/sagitasantia/wisata-indonesia])
Dataset berisi informasi destinasi wisata di Indonesia yang kemudian difilter sehingga hanya menyisakan data destinasi wisata yang berada di Provinsi Jawa Barat.

<img width="342" height="278" alt="Screenshot 2026-06-23 103812" src="https://github.com/user-attachments/assets/8dc9964b-c3a2-4cf3-87d0-9e2376134da7" />

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

<img width="338" height="24" alt="Screenshot 2026-06-23 113137" src="https://github.com/user-attachments/assets/058400c8-2225-450b-b1e6-bba54ba78822" />

### 2. Kategori Wisata
Kategori wisata dikelompokkan menjadi tiga kategori utama.

<img width="173" height="69" alt="Screenshot 2026-06-23 112238" src="https://github.com/user-attachments/assets/a48ff733-bd7e-45bd-b168-e1805cf7c001" />

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
Tahapan preprocessing teks meliput Mengubah teks menjadi lowercase, Menghapus karakter selain huruf dan Menghapus spasi berlebih.

<img width="716" height="185" alt="image" src="https://github.com/user-attachments/assets/c318569e-45e2-4c56-90ff-a746861ca414" />

## Modeling
### Content-Based Filtering
Sistem rekomendasi dibangun menggunakan metode **Content-Based Filtering** yang memanfaatkan informasi deskripsi dan kategori wisata.

### TF-IDF Vectorizer
**TF-IDF** digunakan untuk mengubah data teks menjadi vektor numerik.

<img width="189" height="26" alt="image" src="https://github.com/user-attachments/assets/33df36f1-764f-4e63-9079-c52292e0888d" />

### Cosine Similarity
Setelah data direpresentasikan dalam bentuk vektor TF-IDF, dilakukan perhitungan tingkat kemiripan menggunakan Cosine Similarity.

<img width="269" height="24" alt="Screenshot 2026-06-23 113426" src="https://github.com/user-attachments/assets/cb5fcc95-8e4e-4642-9e1b-fb1cefd01f3b" />

Proses Rekomendasi:
1. Pengguna memilih destinasi wisata.
2. Sistem mengambil indeks wisata tersebut.
3. Menghitung nilai cosine similarity terhadap seluruh wisata lain.
4. Mengurutkan nilai similarity dari terbesar ke terkecil.
5. Menampilkan Top-10 destinasi wisata yang paling mirip.

## Evaluation
### Metrik Evaluasi
Evaluasi dilakukan menggunakan **Precision@10**.

### Definisi Relevan
Rekomendasi dianggap relevan apabila memiliki nilai **kategori_baru** yang sama dengan wisata acuan.

Implementasi Evaluasi
```
def precision_at_k(
    rekomendasi,
    kategori_target,
    k=10
):

    relevan = rekomendasi[
        rekomendasi['kategori_baru']
        == kategori_target
    ]
    return len(relevan) / k
```
### Hasil Evaluasi
Model dievaluasi menggunakan Precision@10 terhadap hasil rekomendasi yang dihasilkan.
Semakin tinggi nilai Precision@10, semakin baik kemampuan sistem dalam memberikan rekomendasi yang relevan.

<img width="877" height="423" alt="Screenshot 2026-06-23 115750" src="https://github.com/user-attachments/assets/b5d7d116-c927-4668-986f-33b16d9c1eee" />

## Deployment
Model yang telah dibangun disimpan menggunakan library Pickle.
```
pickle.dump(
    tfidf,
    open(
        'tfidf_model.pkl',
        'wb'
    )
)
pickle.dump(
    cosine_sim,
    open(
        'cosine_similarity.pkl',
        'wb'
    )
)
```
File model yang dihasilkan:
1. `tfidf_model.pkl`
2. `cosine_similarity.pkl`

Model dapat digunakan kembali tanpa perlu melakukan proses training ulang.

## Hasil Visualisasi
### Distribusi Kategori Wisata Jawa Barat
Tambahkan gambar hasil visualisasi pada bagian ini.

<img width="691" height="467" alt="Screenshot 2026-06-23 113947" src="https://github.com/user-attachments/assets/dde73a81-a216-4287-8c8c-507f9a1efa83" />

### Distribusi Kategori Hasil Rekomendasi
Tambahkan gambar hasil visualisasi rekomendasi pada bagian ini.

<img width="709" height="469" alt="Screenshot 2026-06-23 114045" src="https://github.com/user-attachments/assets/8d145907-fa30-499b-89df-edb4606f74a1" />

## Kesimpulan
Pada proyek ini berhasil dibangun sistem rekomendasi destinasi wisata Jawa Barat menggunakan pendekatan Content-Based Filtering.

Metode TF-IDF digunakan untuk mengubah informasi tekstual menjadi representasi numerik, sedangkan Cosine Similarity digunakan untuk mengukur tingkat kemiripan antar destinasi wisata.

Sistem mampu menghasilkan rekomendasi wisata yang relevan berdasarkan karakteristik wisata yang dipilih pengguna. Evaluasi menggunakan Precision@10 menunjukkan bahwa pendekatan ini dapat digunakan sebagai solusi sederhana dan efektif dalam membantu wisatawan menemukan destinasi wisata yang sesuai dengan preferensinya.
