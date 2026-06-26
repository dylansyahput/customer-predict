"""Simple and user-friendly Streamlit app for customer churn prediction.

Cara pakai:
1. Pastikan file ini berada satu folder dengan:
   - best_model.joblib
   - model_metadata.joblib
   - model_utils.py
2. Jalankan: streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from model_utils import (
    FEATURE_EXPLANATIONS,
    TARGET_COL,
    align_features,
    engineer_customer_features,
    normalize_column_names,
)

MODEL_PATH = Path("best_model.joblib")
METADATA_PATH = Path("model_metadata.joblib")

st.set_page_config(
    page_title="Cek Risiko Churn Pelanggan",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            max-width: 1050px;
        }
        .hero {
            padding: 26px 30px;
            border-radius: 22px;
            background: linear-gradient(135deg, #0f172a, #2563eb);
            color: white;
            margin-bottom: 18px;
        }
        .hero h1 {
            font-size: 2rem;
            margin-bottom: 6px;
        }
        .hero p {
            font-size: 1rem;
            line-height: 1.6;
            opacity: 0.94;
        }
        .simple-card {
            padding: 18px 20px;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            background: #ffffff;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
            margin-bottom: 14px;
        }
        .result-high {
            padding: 22px;
            border-radius: 20px;
            background: #fef2f2;
            border: 1px solid #fecaca;
        }
        .result-medium {
            padding: 22px;
            border-radius: 20px;
            background: #fffbeb;
            border: 1px solid #fde68a;
        }
        .result-low {
            padding: 22px;
            border-radius: 20px;
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
        }
        .small-text {
            color: #64748b;
            font-size: 0.92rem;
            line-height: 1.5;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.45rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_artifacts() -> tuple[Any | None, dict | None]:
    """Load trained model and metadata."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata


def get_numeric_default(metadata: dict, column: str, fallback: float = 0.0) -> float:
    """Get median value from training metadata."""
    info = metadata.get("feature_ranges", {}).get(column, {})
    return float(info.get("median", fallback))


def get_categorical_default(metadata: dict, column: str, fallback: str = "Unknown") -> str:
    """Get first category from training metadata."""
    options = metadata.get("categorical_options", {}).get(column, [])
    if options:
        return str(options[0])
    return fallback


def make_base_customer(metadata: dict) -> dict:
    """Create a safe default row so users only need to fill important fields."""
    required_columns = metadata.get("required_raw_columns", [])
    base: dict[str, Any] = {}

    for column in required_columns:
        if column == TARGET_COL:
            continue
        if column == "customer_id":
            base[column] = 0
        elif column == "signup_date":
            base[column] = "2023-01-01"
        elif column == "last_purchase_date":
            base[column] = metadata.get("reference_date", "2024-12-31")
        elif column in metadata.get("feature_ranges", {}):
            base[column] = get_numeric_default(metadata, column, 0.0)
        elif column in metadata.get("categorical_options", {}):
            base[column] = get_categorical_default(metadata, column)
        else:
            base[column] = pd.NA

    return base


def build_model_input(raw_df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Prepare raw user input into model-ready features."""
    selected_features = metadata.get("selected_features", [])
    model_df = engineer_customer_features(
        raw_df,
        reference_date=metadata.get("reference_date"),
        drop_irrelevant=True,
    )
    if selected_features:
        model_df = align_features(model_df, selected_features)
    return model_df


def predict_dataframe(model: Any, raw_df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Predict churn for one or many customers."""
    model_input = build_model_input(raw_df, metadata)
    threshold = float(metadata.get("best_threshold", 0.5))

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
    else:
        predictions = model.predict(model_input)
        probabilities = predictions.astype(float)

    result = raw_df.copy()
    result["probabilitas_churn"] = probabilities
    result["prediksi_churn"] = predictions
    result["status_prediksi"] = result["prediksi_churn"].map(
        {1: "Berisiko Churn", 0: "Tidak Churn"}
    )
    result["kategori_risiko"] = [risk_label(float(prob), threshold)[0] for prob in probabilities]
    return result


def risk_label(probability: float, threshold: float) -> tuple[str, str, str]:
    """Convert probability to plain-language risk label."""
    if probability >= max(0.65, threshold + 0.20):
        return (
            "Risiko Tinggi",
            "result-high",
            "Pelanggan perlu segera ditangani agar tidak berhenti menggunakan layanan.",
        )
    if probability >= threshold:
        return (
            "Risiko Sedang",
            "result-medium",
            "Pelanggan mulai menunjukkan sinyal churn dan perlu dipantau.",
        )
    return (
        "Risiko Rendah",
        "result-low",
        "Pelanggan relatif aman, tetapi hubungan tetap perlu dijaga.",
    )


def build_recommendations(input_row: dict, probability: float, threshold: float) -> list[str]:
    """Give simple action recommendations based on input and prediction."""
    recommendations: list[str] = []

    satisfaction = float(input_row.get("satisfaction_score", 0))
    nps = float(input_row.get("nps_score", 0))
    support_tickets = float(input_row.get("support_tickets", 0))
    refund = int(input_row.get("refund_requested", 0))
    visits = float(input_row.get("total_visits", 0))
    purchase_freq = float(input_row.get("last_3_month_purchase_freq", 0))

    if probability >= threshold:
        recommendations.append("Hubungi pelanggan melalui email, WhatsApp, atau channel yang paling sering digunakan.")
        recommendations.append("Berikan penawaran retensi yang relevan, misalnya benefit premium, voucher, atau bantuan khusus.")
    else:
        recommendations.append("Pertahankan komunikasi berkala agar pelanggan tetap aktif.")
        recommendations.append("Tawarkan edukasi fitur atau rekomendasi produk agar pelanggan makin sering bertransaksi.")

    if satisfaction <= 3:
        recommendations.append("Prioritaskan peningkatan kepuasan karena skor kepuasan pelanggan masih rendah.")
    if nps <= 5:
        recommendations.append("Tinjau pengalaman pelanggan karena NPS menunjukkan loyalitas yang belum kuat.")
    if support_tickets >= 3:
        recommendations.append("Cek riwayat keluhan pelanggan dan pastikan masalahnya sudah benar-benar selesai.")
    if refund == 1:
        recommendations.append("Periksa penyebab refund karena refund sering berkaitan dengan pengalaman negatif.")
    if visits <= 5 or purchase_freq <= 1:
        recommendations.append("Dorong aktivitas pelanggan dengan reminder, rekomendasi produk, atau campaign personal.")

    return recommendations[:5]


def month_to_date(year: int, month: int, day: int = 1) -> str:
    """Convert month input into date string."""
    return f"{year}-{month:02d}-{day:02d}"


def render_quick_prediction(model: Any, metadata: dict) -> None:
    """Render beginner-friendly single prediction page."""
    st.markdown("### Prediksi Cepat")

    preset = st.selectbox(
        "Pilih contoh kondisi pelanggan",
        ["Isi sendiri", "Pelanggan normal", "Pelanggan mulai berisiko", "Pelanggan sangat berisiko"],
    )

    defaults = {
        "total_visits": int(get_numeric_default(metadata, "total_visits", 15)),
        "avg_session_time": float(get_numeric_default(metadata, "avg_session_time", 8)),
        "pages_per_session": float(get_numeric_default(metadata, "pages_per_session", 4)),
        "total_spent": float(get_numeric_default(metadata, "total_spent", 500)),
        "avg_order_value": float(get_numeric_default(metadata, "avg_order_value", 50)),
        "lifetime_value": float(get_numeric_default(metadata, "lifetime_value", 1000)),
        "support_tickets": int(get_numeric_default(metadata, "support_tickets", 1)),
        "satisfaction_score": float(get_numeric_default(metadata, "satisfaction_score", 4)),
        "nps_score": int(get_numeric_default(metadata, "nps_score", 7)),
        "last_3_month_purchase_freq": int(get_numeric_default(metadata, "last_3_month_purchase_freq", 4)),
        "marketing_spend_per_user": float(get_numeric_default(metadata, "marketing_spend_per_user", 20)),
    }

    if preset == "Pelanggan normal":
        defaults.update(
            {
                "total_visits": 20,
                "avg_session_time": 10.0,
                "pages_per_session": 5.0,
                "support_tickets": 0,
                "satisfaction_score": 4.5,
                "nps_score": 8,
                "last_3_month_purchase_freq": 6,
            }
        )
    elif preset == "Pelanggan mulai berisiko":
        defaults.update(
            {
                "total_visits": 8,
                "avg_session_time": 5.0,
                "pages_per_session": 3.0,
                "support_tickets": 3,
                "satisfaction_score": 3.0,
                "nps_score": 5,
                "last_3_month_purchase_freq": 2,
            }
        )
    elif preset == "Pelanggan sangat berisiko":
        defaults.update(
            {
                "total_visits": 4,
                "avg_session_time": 2.5,
                "pages_per_session": 1.5,
                "support_tickets": 6,
                "satisfaction_score": 2.0,
                "nps_score": 2,
                "last_3_month_purchase_freq": 0,
            }
        )

    with st.form("quick_prediction_form"):
        st.markdown("#### 1. Kondisi pelanggan")
        col1, col2, col3 = st.columns(3)
        with col1:
            satisfaction_score = st.slider(
                "Skor kepuasan",
                1.0,
                5.0,
                float(defaults["satisfaction_score"]),
                0.5,
                help="1 berarti sangat tidak puas, 5 berarti sangat puas.",
            )
            nps_score = st.slider(
                "NPS atau loyalitas",
                0,
                10,
                int(defaults["nps_score"]),
                help="0 berarti tidak mau merekomendasikan, 10 berarti sangat mau merekomendasikan.",
            )
        with col2:
            support_tickets = st.number_input(
                "Jumlah keluhan atau tiket bantuan",
                min_value=0,
                max_value=50,
                value=int(defaults["support_tickets"]),
                step=1,
            )
            refund_requested = st.radio(
                "Pernah meminta refund?",
                [0, 1],
                format_func=lambda value: "Ya" if value == 1 else "Tidak",
                horizontal=True,
            )
        with col3:
            is_premium_user = st.radio(
                "Pelanggan premium?",
                [0, 1],
                format_func=lambda value: "Ya" if value == 1 else "Tidak",
                horizontal=True,
            )
            subscription_type = st.selectbox(
                "Jenis langganan",
                metadata.get("categorical_options", {}).get("subscription_type", ["Monthly", "Annual"]),
            )

        st.markdown("#### 2. Aktivitas penggunaan")
        col1, col2, col3 = st.columns(3)
        with col1:
            total_visits = st.number_input(
                "Total kunjungan",
                min_value=0,
                max_value=500,
                value=int(defaults["total_visits"]),
                step=1,
            )
            avg_session_time = st.number_input(
                "Rata-rata waktu sesi",
                min_value=0.0,
                max_value=120.0,
                value=float(defaults["avg_session_time"]),
                step=0.5,
                help="Semakin tinggi, biasanya pelanggan semakin aktif.",
            )
        with col2:
            pages_per_session = st.number_input(
                "Halaman per sesi",
                min_value=0.0,
                max_value=50.0,
                value=float(defaults["pages_per_session"]),
                step=0.5,
            )
            last_3_month_purchase_freq = st.number_input(
                "Frekuensi pembelian 3 bulan terakhir",
                min_value=0,
                max_value=100,
                value=int(defaults["last_3_month_purchase_freq"]),
                step=1,
            )
        with col3:
            signup_month = st.selectbox("Bulan daftar", list(range(1, 13)), index=0)
            last_purchase_month = st.selectbox("Bulan pembelian terakhir", list(range(1, 13)), index=11)

        st.markdown("#### 3. Nilai transaksi")
        col1, col2, col3 = st.columns(3)
        with col1:
            total_spent = st.number_input(
                "Total belanja pelanggan",
                min_value=0.0,
                max_value=100000.0,
                value=float(defaults["total_spent"]),
                step=50.0,
            )
            avg_order_value = st.number_input(
                "Rata-rata nilai transaksi",
                min_value=0.0,
                max_value=100000.0,
                value=float(defaults["avg_order_value"]),
                step=10.0,
            )
        with col2:
            lifetime_value = st.number_input(
                "Lifetime value",
                min_value=0.0,
                max_value=100000.0,
                value=float(defaults["lifetime_value"]),
                step=50.0,
                help="Perkiraan total nilai pelanggan selama memakai layanan.",
            )
            marketing_spend_per_user = st.number_input(
                "Biaya marketing untuk pelanggan",
                min_value=0.0,
                max_value=10000.0,
                value=float(defaults["marketing_spend_per_user"]),
                step=5.0,
            )
        with col3:
            discount_used = st.radio(
                "Pernah memakai diskon?",
                [0, 1],
                format_func=lambda value: "Ya" if value == 1 else "Tidak",
                horizontal=True,
            )
            delivery_delay_days = st.number_input(
                "Keterlambatan pengiriman atau layanan",
                min_value=0,
                max_value=100,
                value=int(get_numeric_default(metadata, "delivery_delay_days", 1)),
                step=1,
            )

        submitted = st.form_submit_button("Cek Risiko Churn", type="primary", use_container_width=True)

    if not submitted:
        return

    input_row = make_base_customer(metadata)
    input_row.update(
        {
            "signup_date": month_to_date(2023, int(signup_month), 1),
            "last_purchase_date": month_to_date(2024, int(last_purchase_month), 15),
            "subscription_type": subscription_type,
            "is_premium_user": is_premium_user,
            "total_visits": total_visits,
            "avg_session_time": avg_session_time,
            "pages_per_session": pages_per_session,
            "total_spent": total_spent,
            "avg_order_value": avg_order_value,
            "discount_used": discount_used,
            "support_tickets": support_tickets,
            "refund_requested": refund_requested,
            "delivery_delay_days": delivery_delay_days,
            "satisfaction_score": satisfaction_score,
            "nps_score": nps_score,
            "marketing_spend_per_user": marketing_spend_per_user,
            "lifetime_value": lifetime_value,
            "last_3_month_purchase_freq": last_3_month_purchase_freq,
        }
    )

    raw_df = pd.DataFrame([input_row])
    result = predict_dataframe(model, raw_df, metadata).iloc[0]
    probability = float(result["probabilitas_churn"])
    threshold = float(metadata.get("best_threshold", 0.5))
    risk_name, risk_class, risk_message = risk_label(probability, threshold)

    st.markdown("### Hasil Prediksi")
    col1, col2 = st.columns([1, 1.25])
    with col1:
        st.markdown(
            f"""
            <div class="{risk_class}">
                <h2>{risk_name}</h2>
                <p>{risk_message}</p>
                <h1>{probability:.1%}</h1>
                <p class="small-text">Probabilitas pelanggan akan churn.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(probability, 0.0), 1.0))

    with col2:
        if int(result["prediksi_churn"]) == 1:
            st.error("Kesimpulan: pelanggan berpotensi churn.")
        else:
            st.success("Kesimpulan: pelanggan diprediksi tidak churn.")

        st.markdown("#### Saran tindakan")
        for item in build_recommendations(input_row, probability, threshold):
            st.write(f"• {item}")

    with st.expander("Lihat detail input yang dikirim ke model"):
        st.dataframe(build_model_input(raw_df, metadata), use_container_width=True)


def fill_missing_batch_columns(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Fill missing columns in uploaded CSV with safe training defaults."""
    data = normalize_column_names(df)
    base = make_base_customer(metadata)

    for column, value in base.items():
        if column not in data.columns:
            data[column] = value

    return data


def render_batch_prediction(model: Any, metadata: dict) -> None:
    """Render batch prediction page."""
    st.markdown("### Upload Banyak Pelanggan")
    st.write(
        "Gunakan halaman ini kalau kamu punya file CSV. Aplikasi akan menambahkan kolom hasil prediksi di bagian akhir."
    )

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload file CSV terlebih dahulu.")
        return

    batch_df = pd.read_csv(uploaded_file)
    batch_df = fill_missing_batch_columns(batch_df, metadata)
    result_df = predict_dataframe(model, batch_df, metadata)

    high_risk = int((result_df["kategori_risiko"] == "Risiko Tinggi").sum())
    medium_risk = int((result_df["kategori_risiko"] == "Risiko Sedang").sum())
    low_risk = int((result_df["kategori_risiko"] == "Risiko Rendah").sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Risiko Tinggi", high_risk)
    col2.metric("Risiko Sedang", medium_risk)
    col3.metric("Risiko Rendah", low_risk)

    st.dataframe(result_df.head(100), use_container_width=True)

    csv_data = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download hasil prediksi",
        data=csv_data,
        file_name="hasil_prediksi_churn.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_explanation(metadata: dict) -> None:
    """Render short explanation page for non-technical users."""
    st.markdown("### Penjelasan Singkat")

    metrics = metadata.get("metrics", {})
    st.markdown(
        f"""
        <div class="simple-card">
            <h4>Model yang digunakan</h4>
            <p>
                Aplikasi ini memakai model <b>{metadata.get('best_model', 'machine learning')}</b>
                dari skenario <b>{metadata.get('best_scenario', 'model terbaik')}</b>.
                Model dipilih dari proses perbandingan beberapa model, lalu digunakan untuk memprediksi risiko churn pelanggan.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
    col2.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    col3.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    col4.metric("F1-score", f"{metrics.get('f1_score', 0):.3f}")

    st.markdown("#### Fitur yang paling dipakai model")
    feature_df = pd.DataFrame(
        [
            {
                "Fitur": feature,
                "Arti sederhana": metadata.get("feature_explanations", FEATURE_EXPLANATIONS).get(feature, "-"),
            }
            for feature in metadata.get("selected_features", [])
        ]
    )
    st.dataframe(feature_df, use_container_width=True)

    with st.expander("Apa arti hasil prediksi?"):
        st.write(
            "Risiko churn menunjukkan kemungkinan pelanggan berhenti menggunakan layanan. "
            "Hasil ini tidak boleh dianggap sebagai keputusan mutlak, tetapi sebagai alat bantu untuk menentukan pelanggan mana yang perlu diprioritaskan."
        )


model, metadata = load_artifacts()

st.markdown(
    """
    <div class="hero">
        <h1>Cek Risiko Churn Pelanggan</h1>
        
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None or metadata is None:
    st.error("Model belum ditemukan. Pastikan best_model.joblib dan model_metadata.joblib ada di folder yang sama dengan app.py.")
    st.info("Kalau file model belum ada, jalankan dulu: python train_model.py")
    st.stop()

with st.sidebar:
    st.markdown("## Menu")
    page = st.radio(
        "Pilih halaman",
        ["Prediksi Cepat", "Upload Banyak Data", "Penjelasan Model"],
        label_visibility="collapsed",
    )
    st.caption(f"Model aktif: {metadata.get('best_model', 'Model')}")

if page == "Prediksi Cepat":
    render_quick_prediction(model, metadata)
elif page == "Upload Banyak Data":
    render_batch_prediction(model, metadata)
else:
    render_explanation(metadata)