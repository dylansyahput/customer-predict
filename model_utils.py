"""Utility functions for customer churn modelling and Streamlit deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TARGET_COL = "churn"
DATE_COLUMNS = ["signup_date", "last_purchase_date"]
DROP_COLUMNS = ["customer_id", "signup_date", "last_purchase_date"]
RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURE_EXPLANATIONS = {
    "gender": "Jenis kelamin pelanggan.",
    "age": "Usia pelanggan.",
    "country": "Negara asal pelanggan.",
    "city": "Kota pelanggan.",
    "acquisition_channel": "Sumber akuisisi pelanggan, seperti email, organic, referral, atau ads.",
    "device_type": "Perangkat utama yang digunakan pelanggan saat mengakses layanan.",
    "subscription_type": "Jenis paket atau langganan pelanggan.",
    "is_premium_user": "Status pelanggan premium, 1 berarti premium dan 0 berarti nonpremium.",
    "total_visits": "Total kunjungan pelanggan pada layanan.",
    "avg_session_time": "Rata-rata durasi sesi pelanggan.",
    "pages_per_session": "Rata-rata jumlah halaman yang dibuka dalam satu sesi.",
    "email_open_rate": "Persentase email marketing yang dibuka oleh pelanggan.",
    "email_click_rate": "Persentase klik pelanggan pada email marketing.",
    "total_spent": "Total pengeluaran pelanggan selama menggunakan layanan.",
    "avg_order_value": "Rata-rata nilai transaksi pelanggan.",
    "discount_used": "Status penggunaan diskon, 1 berarti pernah memakai diskon dan 0 berarti tidak.",
    "coupon_code": "Kode kupon yang digunakan pelanggan.",
    "support_tickets": "Jumlah tiket bantuan atau keluhan yang dibuat pelanggan.",
    "refund_requested": "Status permintaan refund, 1 berarti pernah meminta refund dan 0 berarti tidak.",
    "delivery_delay_days": "Jumlah hari keterlambatan pengiriman yang pernah dialami pelanggan.",
    "payment_method": "Metode pembayaran utama pelanggan.",
    "satisfaction_score": "Skor kepuasan pelanggan.",
    "nps_score": "Net Promoter Score pelanggan.",
    "marketing_spend_per_user": "Biaya pemasaran yang dikeluarkan untuk pelanggan tersebut.",
    "lifetime_value": "Nilai pelanggan selama berhubungan dengan perusahaan.",
    "last_3_month_purchase_freq": "Frekuensi pembelian pelanggan dalam 3 bulan terakhir.",
    "customer_tenure_days": "Lama hubungan pelanggan sejak tanggal daftar sampai transaksi terakhir.",
    "recency_days": "Jarak hari antara tanggal referensi dengan transaksi terakhir pelanggan.",
    "signup_month": "Bulan pelanggan mendaftar.",
    "signup_year": "Tahun pelanggan mendaftar.",
    "last_purchase_month": "Bulan transaksi terakhir pelanggan.",
    "last_purchase_year": "Tahun transaksi terakhir pelanggan.",
    "last_purchase_dayofweek": "Hari dalam minggu ketika transaksi terakhir terjadi.",
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of dataframe with trimmed lowercase column names."""
    data = df.copy()
    data.columns = [str(column).strip().lower() for column in data.columns]
    return data


def engineer_customer_features(
    df: pd.DataFrame,
    reference_date: str | pd.Timestamp | None = None,
    drop_irrelevant: bool = True,
) -> pd.DataFrame:
    """Create date-based features and remove columns that are not useful for modelling."""
    data = normalize_column_names(df)

    for column in DATE_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_datetime(data[column], errors="coerce")

    if "signup_date" in data.columns and "last_purchase_date" in data.columns:
        if reference_date is None:
            reference = data["last_purchase_date"].max()
        else:
            reference = pd.to_datetime(reference_date, errors="coerce")

        data["customer_tenure_days"] = (
            data["last_purchase_date"] - data["signup_date"]
        ).dt.days
        data["recency_days"] = (reference - data["last_purchase_date"]).dt.days
        data["signup_month"] = data["signup_date"].dt.month.astype("Int64").astype("object")
        data["signup_year"] = data["signup_date"].dt.year.astype("Int64").astype("object")
        data["last_purchase_month"] = data["last_purchase_date"].dt.month.astype("Int64").astype("object")
        data["last_purchase_year"] = data["last_purchase_date"].dt.year.astype("Int64").astype("object")
        data["last_purchase_dayofweek"] = data["last_purchase_date"].dt.dayofweek.astype("Int64").astype("object")

    if drop_irrelevant:
        data = data.drop(columns=DROP_COLUMNS, errors="ignore")

    return data


def split_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separate numerical and categorical columns for ColumnTransformer."""
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32", "Int64"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]
    return numeric_features, categorical_features


def align_features(X: pd.DataFrame, selected_features: Iterable[str]) -> pd.DataFrame:
    """Ensure dataframe has every selected feature in the correct order."""
    data = X.copy()
    for feature in selected_features:
        if feature not in data.columns:
            data[feature] = pd.NA
    return data[list(selected_features)]


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip numerical outliers using IQR bounds learned from training data only."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=float)
        q1 = np.nanquantile(array, 0.25, axis=0)
        q3 = np.nanquantile(array, 0.75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.factor * iqr
        self.upper_bounds_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        array = np.asarray(X, dtype=float)
        return np.clip(array, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return np.asarray(input_features, dtype=object)
