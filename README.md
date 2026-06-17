# Customer Churn Prediction - UAS Bengkel Koding Data Science

Proyek ini berisi notebook, training script, model terbaik, dan aplikasi Streamlit untuk memprediksi **customer churn** pada dataset Sales and Marketing Customer.

## Ringkasan Perbaikan

Kode sudah disesuaikan dengan aspek penilaian UAS:

1. **EDA lengkap**
   - Menampilkan 5 baris pertama.
   - Menampilkan informasi dataset.
   - Menampilkan statistik deskriptif.
   - Menghitung dan memvisualisasikan persentase missing value.
   - Menampilkan distribusi target `churn`.
   - Membuat heatmap korelasi fitur numerik.

2. **Direct Modeling**
   - Menggunakan target `churn` sebagai `y`.
   - Menggunakan seluruh kolom lain sebagai `X`.
   - Train-test split stratified.
   - Melatih 3 kategori model:
     - Logistic Regression sebagai model konvensional.
     - Random Forest sebagai ensemble bagging.
     - Voting Classifier sebagai gabungan model konvensional.
   - Evaluasi menggunakan accuracy, precision, recall, F1-score, dan confusion matrix.

3. **Modeling dengan Preprocessing**
   - Penanganan missing value.
   - Penghapusan duplikasi.
   - Deteksi dan clipping outlier berbasis IQR pada data latih.
   - Encoding fitur kategorikal.
   - Scaling fitur numerik setelah data splitting melalui pipeline.
   - Penghapusan fitur yang tidak relevan seperti `customer_id`, `signup_date`, dan `last_purchase_date` setelah dibuat fitur turunan.

4. **Hyperparameter Tuning dan Feature Selection**
   - Feature importance menggunakan Random Forest.
   - Feature selection berdasarkan importance.
   - Hyperparameter tuning menggunakan `RandomizedSearchCV`.
   - Evaluasi ulang best estimator.
   - Total evaluasi 9 model dari 3 model x 3 skenario.

5. **Deployment Premium Streamlit**
   - UI lebih modern dan user friendly.
   - Single customer prediction.
   - Batch CSV prediction.
   - Model insights.
   - Feature importance.
   - Data quality view.
   - Deployment guide.

## Model Terbaik

Berdasarkan hasil training terbaru, model terbaik adalah:

| Skenario | Model | Accuracy | Precision | Recall | F1-score |
|---|---|---:|---:|---:|---:|
| Hyperparameter Tuning + Feature Selection | Random Forest | 0.8553 | 0.5148 | 0.9826 | 0.6756 |

Model ini dipilih karena menghasilkan F1-score terbaik, sehingga lebih sesuai untuk kasus churn yang membutuhkan keseimbangan antara kemampuan menemukan pelanggan churn dan menjaga ketepatan prediksi.

## Struktur File

```text
customer-predict-main/
├── .streamlit/
│   └── config.toml
├── artifacts/
│   ├── feature_importance.csv
│   ├── missing_value_percentage.csv
│   ├── model_evaluation_results.csv
│   ├── model_metadata.json
│   ├── outlier_summary_train.csv
│   └── tuning_results.csv
├── Sales - Marketing customer dataset.csv
├── app.py
├── best_model.joblib
├── customer_predict.ipynb
├── model_metadata.joblib
├── model_utils.py
├── train_model.py
├── requirements.txt
├── README.md
└── LAPORAN_DEPLOYMENT.md
```

## Cara Menjalankan Lokal

1. Install dependency:

```bash
pip install -r requirements.txt
```

2. Train ulang model:

```bash
python train_model.py
```

3. Jalankan aplikasi Streamlit:

```bash
streamlit run app.py
```

## Cara Deploy ke Streamlit Community Cloud

1. Push semua file proyek ke GitHub.
2. Buka Streamlit Community Cloud.
3. Pilih **New app**.
4. Pilih repository GitHub.
5. Set **Main file path** ke:

```text
app.py
```

6. Klik **Deploy**.
7. Uji halaman:
   - Single Prediction
   - Batch Prediction
   - Model Insights
   - Deployment Guide

## Catatan Penting

- File `best_model.joblib` membutuhkan `model_utils.py` karena pipeline memakai transformer custom `IQRClipper`.
- Jangan menghapus `model_metadata.joblib` karena aplikasi memakai file tersebut untuk opsi input, daftar fitur, metrik, dan informasi model.
- Dataset mentah tidak wajib dibuka oleh aplikasi Streamlit, tetapi tetap disertakan supaya model dapat dilatih ulang.
