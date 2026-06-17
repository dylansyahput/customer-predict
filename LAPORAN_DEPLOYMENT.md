# Laporan Deployment Customer Churn Prediction

## 1. Tujuan Deployment

Deployment dilakukan untuk menyediakan aplikasi prediksi customer churn berbasis Streamlit. Aplikasi ini membantu pengguna memasukkan data pelanggan, memperoleh prediksi churn, melihat probabilitas risiko, dan membaca rekomendasi tindak lanjut secara lebih mudah.

## 2. Model yang Digunakan

Model yang digunakan pada deployment adalah **Random Forest** dari skenario **Hyperparameter Tuning + Feature Selection**.

Hasil evaluasi model terbaik:

| Metric | Score |
|---|---:|
| Accuracy | 0.8553 |
| Precision | 0.5148 |
| Recall | 0.9826 |
| F1-score | 0.6756 |

Model dipilih berdasarkan F1-score karena target churn memiliki ketidakseimbangan kelas. Dengan F1-score, penilaian model tidak hanya melihat akurasi, tetapi juga mempertimbangkan precision dan recall.

## 3. File Deployment

File yang digunakan untuk deployment adalah:

1. `app.py` sebagai aplikasi utama Streamlit.
2. `model_utils.py` sebagai fungsi pendukung preprocessing dan feature engineering.
3. `best_model.joblib` sebagai model terbaik.
4. `model_metadata.joblib` sebagai metadata fitur, opsi input, metrik, dan informasi model.
5. `requirements.txt` sebagai daftar dependency.
6. `.streamlit/config.toml` sebagai konfigurasi tema aplikasi.

## 4. Fitur Aplikasi

Aplikasi Streamlit sudah dibuat lebih premium dan user friendly dengan fitur berikut:

1. **Single Customer Prediction**
   - Form input dibagi menjadi Profil, Engagement, Transaksi, dan Service.
   - Hasil prediksi ditampilkan dalam bentuk risk segment.
   - Probabilitas churn ditampilkan dalam persentase.
   - Tersedia rekomendasi aksi retensi.

2. **Batch Prediction**
   - Pengguna dapat upload file CSV.
   - Aplikasi menghasilkan kolom `churn_prediction`, `churn_probability`, dan `risk_segment`.
   - Hasil prediksi dapat diunduh kembali sebagai CSV.

3. **Model Insights**
   - Menampilkan evaluasi 9 model.
   - Menampilkan confusion matrix model terbaik.
   - Menampilkan feature importance.
   - Menampilkan distribusi target dan missing value.

4. **Deployment Guide**
   - Menyediakan langkah menjalankan lokal.
   - Menyediakan checklist deployment ke Streamlit Cloud.

## 5. Cara Menjalankan Lokal

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## 6. Cara Deploy ke Streamlit Cloud

1. Buat repository GitHub.
2. Upload semua file proyek.
3. Buka Streamlit Community Cloud.
4. Pilih repository.
5. Isi main file path dengan `app.py`.
6. Klik deploy.
7. Pastikan aplikasi dapat diakses publik.

## 7. Validasi Lokal

Training script telah dijalankan dan berhasil menghasilkan:

1. `best_model.joblib`
2. `model_metadata.joblib`
3. `artifacts/model_evaluation_results.csv`
4. `artifacts/feature_importance.csv`
5. `artifacts/tuning_results.csv`

Model juga sudah diuji untuk melakukan prediksi menggunakan input dari dataset.
