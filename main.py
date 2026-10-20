import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------------
# 1. Pengumpulan Data & Synthetic Dataset
# ---------------------------------------------------------
print("=== 1. Pengumpulan & Penyiapan Data ===")
data = {
    'text': [
        'Selamat Anda menang hadiah uang tunai 100 juta rupiah, klik link ini sekarang!',
        'Halo bro, nanti malam jadi kumpul di cafe biasa tidak?',
        'Pinjaman online tanpa agunan cair cepat, hubungi WA 08123456789',
        'Rapat koordinasi proyek akan dilaksanakan besok jam 9 pagi di ruang 302',
        'PROMO DAHSYAT! Beli 1 gratis 1 paket kuota internet khusus hari ini',
        'Ibu minta tolong belikan obat di apotek ya nak, pulangnya hati-hati',
        'Dapatkan diskon 70% untuk semua produk pakaian hanya minggu ini!',
        'Tugas kuliah pengantar machine learning dikumpulkan kapan ya?',
        'Transfer sekarang ke rekening ini untuk memenangkan mobil mewah',
        'Selamat pagi Pak, berikut lampiran laporan mingguan yang sudah direvisi',
        'Info loker gaji 10 juta per bulan tanpa ijazah, daftar di link berikut',
        'Jangan lupa besok kita ada jadwal presentasi studi kasus deep learning',
        'Bonus deposit 100% untuk member baru game online, klaim sekarang!',
        'Catatan kuliah sesi 4 tentang supervised dan unsupervised learning',
        'Anda terpilih mendapatkan voucher belanja gratis senilai 500rb',
        'Terima kasih atas bantuan Anda dalam menyelesaikan proyek kemarin',
        'Kredit cepat langsung cair tanpa syarat berbelit-belit!',
        'Apakah besok ada perkuliahan tatap muka di kampus?',
        'Pemberitahuan resmi: akun Anda diblokir, klik tautan untuk verifikasi',
        'Ayo makan siang bareng di kantin pusat jam 12 nanti'
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # 1: Spam, 0: Ham
}

df = pd.DataFrame(data)
print(f"Jumlah sampel data: {len(df)}")
print(df.head())

# ---------------------------------------------------------
# 2. Pembersihan & Pra-pemrosesan Data (Feature Extraction)
# ---------------------------------------------------------
print("\n=== 2. Pembersihan & Vektoriasi Teks ===")
# Mengubah teks menjadi angka menggunakan TF-IDF
vectorizer = TfidfVectorizer(lowercase=True, stop_words=None)
X = vectorizer.fit_transform(df['text']).toarray()
y = np.array(df['label'])

print(f"Bentuk matriks fitur (X): {X.shape}")

# ---------------------------------------------------------
# 3. Pembagian Data Pelatihan & Pengujian (80:20)
# ---------------------------------------------------------
print("\n=== 3. Pembagian Data (80% Train, 20% Test) ===")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data Pelatihan: {X_train.shape[0]} sampel")
print(f"Data Pengujian: {X_test.shape[0]} sampel")

# ---------------------------------------------------------
# 4. Membangun Model Deep Learning (Neural Network)
# ---------------------------------------------------------
print("\n=== 4. Membangun & Melatih Model Deep Learning ===")
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),  # Mencegah overfitting (sesuai bahasan di resume)
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')  # Output binary (Spam vs Ham)
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Pelatihan Model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=2,
    validation_split=0.2,
    verbose=0
)

print("Pelatihan model selesai.")

# ---------------------------------------------------------
# 5. Evaluasi Kinerja Model
# ---------------------------------------------------------
print("\n=== 5. Evaluasi Kinerja Model ===")
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# ---------------------------------------------------------
# 6. Pengujian Data Baru (Inference)
# ---------------------------------------------------------
print("\n=== 6. Pengujian Data Baru (Inference) ===")
pesan_baru = [
    "Selamat! Anda mendapatkan hadiah voucher gratis 1 juta, hubungi kami segera",
    "Halo Pak, kumpul tugas pengantar AI dikirim lewat email atau LMS?"
]

pesan_vec = vectorizer.transform(pesan_baru).toarray()
pred_prob = model.predict(pesan_vec)

for i, msg in enumerate(pesan_baru):
    label = "SPAM" if pred_prob[i] > 0.5 else "HAM (Bukan Spam)"
    print(f"Pesan: '{msg}'")
    print(f"Prediksi: {label} (Probabilitas Spam: {pred_prob[i][0]:.4f})\n")
