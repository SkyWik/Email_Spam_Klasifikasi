# Studi Kasus Deep Learning: Klasifikasi Pesan Spam (NLP)

Studi kasus ini disusun berdasarkan materi **Resume Sesi 4: Introduction to Machine Learning** (Pemateri: Taniza Tajuddin, UiTM Kedah).

## Deskripsi Masalah
Dalam era digital, pesan spam (iklan ilegal, penipuan, phishing) sangat mengganggu pengguna. 
Sesuai dengan pembahasan *Supervised Learning (Classification)* dan perkembangan *Deep Learning (Neural Network)* pada materi, project ini mengimplementasikan model Deep Learning (Artificial Neural Network / Multi-Layer Perceptron) menggunakan Keras/TensorFlow untuk mengklasifikasikan pesan ke dalam kategori **Spam** atau **Ham** (Bukan Spam).

## Alur Kerja (Workflow) sesuai Materi ML:
1. **Pengumpulan & Pembersihan Data**: Membersihkan teks dari karakter khusus dan mengubah ke huruf kecil.
2. **Ekstraksi Fitur / Pembentukan Input**: Menggunakan Tokenization dan Text Vectorization/TF-IDF.
3. **Pembagian Data (Train/Test Split)**: Membagi data 80% untuk training dan 20% untuk testing.
4. **Pelatihan Model Deep Learning**: Membangun Neural Network dengan Dense layer, Dropout (mencegah overfitting), dan fungsi aktivasi ReLU/Sigmoid.
5. **Evaluasi Model**: Menghitung **Accuracy**, **Precision**, **Recall**, **F1-Score**, dan menampilkan **Confusion Matrix**.
6. **Inference / Prediksi Data Baru**: Menguji model dengan kalimat pesan baru.

## Cara Menjalankan Kode:
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
python main.py
```
