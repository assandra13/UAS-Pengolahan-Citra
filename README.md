# UAS-Pengolahan-Citra

# Segmentasi Gambar Menggunakan K-means Clustering

Proyek ini bertujuan untuk melakukan segmentasi gambar menggunakan algoritma K-means clustering. Segmentasi gambar adalah proses membagi gambar digital menjadi beberapa segmen (kumpulan piksel) untuk menyederhanakan representasi gambar menjadi sesuatu yang lebih bermakna dan lebih mudah dianalisis.

## Prasyarat

Pastikan Anda telah menginstal scikit-learn, opencv-python, matplotlib, dan numpy.
Anda bisa menginstalnya menggunakan pip:

```sh
pip install scikit-learn opencv-python matplotlib numpy
```

## Langkah - langkah

**1. Instalasi scikit-learn di Jupyter Notebook**
ss
Jika Anda menggunakan Jupyter Notebook, Anda bisa menginstal scikit-learn dengan menjalankan perintah berikut di sel notebook:

```sh
 !pip install scikit-learn
```

**2. Impor Pustaka yang Diperlukan**

Setelah menginstal pustaka, impor pustaka yang diperlukan untuk proses segmentasi:

```sh
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
```

**3. Mendefinisikan Fungsi Segmentasi**

Definisikan fungsi segment_and_display yang akan membaca gambar, melakukan segmentasi menggunakan K-means clustering, dan menampilkan hasilnya:

```sh
def segment_and_display(image_path, scale_percent=50, k=3):
    # Load gambar
    image = cv2.imread(image_path)
    if image is None:
        print(f"Gambar {image_path} tidak dapat dibaca. Periksa kembali path dan integritas file.")
        return

    # Mengurangi ukuran gambar untuk mempercepat komputasi
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Mengubah warna ke RGB (dari BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Membentuk ulang gambar menjadi susunan piksel 2D dan 3 nilai warna (RGB)
    pixel_vals = image.reshape((-1, 3))

    # Mengkonversikan ke tipe float
    pixel_vals = np.float32(pixel_vals)

    # Melakukan K-means clustering dengan jumlah cluster yang ditetapkan sebagai 3
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(pixel_vals)
    centers = kmeans.cluster_centers_.astype(np.uint8)
    labels = kmeans.labels_

    # Membentuk ulang data menjadi dimensi gambar asli
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    # Menampilkan gambar asli dan hasil segmentasi
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Gambar Asli: ' + image_path)
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title('Hasil Segmentasi')
    plt.show()
```

**4. Menjalankan Segmentasi untuk Setiap Gambar**

Buat daftar path gambar yang ingin disegmentasi dan jalankan fungsi segment_and_display untuk setiap gambar:

```sh
# Daftar path gambar
images = ['1.png', '2.png', '3.png']

# Proses setiap gambar
for image_path in images:
    segment_and_display(image_path)
```

## Hasil

## Gambar Asli

![Gambar Asli](image/asli%201.png)
![Gambar Asli](image/asli%202.png)
![Gambar Asli](image/asli%203.png)

## Hasil Segmentasi dengan K-means clustering

![Hasil Segmentasi](image/segmen%201.png)
![Hasil Segmentasi](image/segmen%202.png)
![Hasil Segmentasi](image/segmen%203.png)

## Struktur Proyek

Pastikan struktur direktori Anda sebagai berikut:

```sh
project-directory/
│
├── 1.png
├── 2.png
├── 3.png
└── segmentasi_gambar.ipynb  # Jupyter Notebook yang berisi kode di atas
```

## Kesimpulan

Proyek ini menunjukkan bagaimana menggunakan K-means clustering untuk segmentasi gambar dalam Python menggunakan pustaka scikit-learn. Dengan menyesuaikan parameter dan meningkatkan preprocessing, hasil segmentasi dapat disesuaikan untuk berbagai aplikasi dalam pengolahan citra.
