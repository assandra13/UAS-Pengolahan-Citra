import numpy as np
import cv2
import matplotlib.pyplot as plt

def segment_and_display(image_path):
    # Load gambar
    image = cv2.imread(image_path)
    if image is None:
        print(f"Gambar {image_path} tidak dapat dibaca. Periksa kembali path dan integritas file.")
        return

    # Mengubah warna ke RGB (dari BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Membentuk ulang gambar menjadi susunan piksel 2D dan 3 nilai warna (RGB)
    pixel_vals = image.reshape((-1,3))

    # Mengkonversikan ke tipe float
    pixel_vals = np.float32(pixel_vals)

    # Menentukan kriteria berhenti (100 iterasi atau akurasi mencapai 85%)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

    # Melakukan K-means clustering dengan jumlah cluster yang ditetapkan sebagai 3
    k = 3
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Mengonversi data menjadi nilai 8-bit dan membentuk ulang data menjadi dimensi gambar asli
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((image.shape))

    # Menampilkan gambar asli dan hasil segmentasi
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Gambar Asli: ' + image_path)
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title('Hasil Segmentasi')
    plt.show()

# Daftar path gambar
images = ['1.png', '2.png', '3.png']

# Proses setiap gambar
for image_path in images:
    segment_and_display(image_path)
