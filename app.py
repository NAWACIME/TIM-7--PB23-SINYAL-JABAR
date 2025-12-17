import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard Sinyal Jabar Kelompok 7",
    page_icon="📡",
    layout="wide"
)

# --- 2. STYLE TAMPILAN (CSS) ---
# BAGIAN INI YANG DIUBAH UNTUK MENGGANTI WARNA FONT
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    
    /* 1. Mengubah warna angka di dalam kotak Metrik (Value) */
    [data-testid="stMetricValue"] {
        color: #1E3A8A; /* Biru Gelap */
        font-weight: bold;
    }
    
    /* 2. Mengubah warna teks judul di dalam kotak Metrik (Label) */
    [data-testid="stMetricLabel"] {
        color: #333333; /* Abu-abu Tua agar terbaca jelas */
    }

    /* 3. Style Kotak Putih Metrik */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    
    /* 4. Mengubah warna font di dalam Tabel agar hitam pekat */
    .stDataFrame {
        color: #000000;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD DATA & LOGIC CLUSTERING (4 Variabel Sesuai .ipynb) ---
@st.cache_data
def get_clustered_data():
    df = pd.read_csv("Sinyal.csv")
    features = ['SINYAL KUAT', 'SINYAL LEMAH', 'TIDAK ADA SINYAL', '4G/LTE']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    return df

try:
    df_final = get_clustered_data()
except Exception as e:
    st.error(f"Gagal memuat file Sinyal.csv. Error: {e}")
    st.stop()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("📡 Navigasi")
    selected_cluster = st.selectbox("Pilih Opsi Cluster:", sorted(df_final['Cluster'].unique()))
    st.markdown("---")
    st.write("### 👥 Kelompok 7")
    st.info("1. Naura Afnandita\n2. Maura Azzahra\n3. Mimma desmaya\n4. Mustika Taulina")

# --- 5. JUDUL ---
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>ANALISIS KUALITAS SINYAL & 4G JAWA BARAT</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Berdasarkan Pemodelan K-Means Clustering</p>", unsafe_allow_html=True)
st.markdown("---")

# Filter data
filtered_df = df_final[df_final['Cluster'] == selected_cluster]

# --- 6. METRIK UTAMA ---
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Total BTS", f"{int(filtered_df['BTS'].sum())}")
with m2:
    st.metric("Avg Sinyal Kuat", f"{filtered_df['SINYAL KUAT'].mean():.1f}")
with m3:
    st.metric("Avg Sinyal Lemah", f"{filtered_df['SINYAL LEMAH'].mean():.1f}")
with m4:
    st.metric("Avg No Sinyal", f"{filtered_df['TIDAK ADA SINYAL'].mean():.1f}")
with m5:
    st.metric("Avg 4G/LTE", f"{filtered_df['4G/LTE'].mean():.1f}")

# --- 7. VISUALISASI GRAFIK ---
st.markdown("---")
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"📊 Perbandingan Variabel Sinyal di Cluster {selected_cluster}")
    fig_bar = px.bar(
        filtered_df,
        x='KABUPATEN JAWA BARAT',
        y=['SINYAL KUAT', 'SINYAL LEMAH', 'TIDAK ADA SINYAL', '4G/LTE'], 
        barmode='group',
        color_discrete_map={
            'SINYAL KUAT': '#2ecc71',
            'SINYAL LEMAH': '#f1c40f',
            'TIDAK ADA SINYAL': '#e74c3c',
            '4G/LTE': '#3498db'
        },
        template="plotly_white"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.subheader("💡 Interpretasi Cluster")
    avg_4g = filtered_df['4G/LTE'].mean()
    avg_no_sinyal = filtered_df['TIDAK ADA SINYAL'].mean()
    
    if avg_4g > 500:
        st.success("✅ **Cluster Unggul**: Wilayah ini memiliki jangkauan 4G dan sinyal kuat yang sangat luas.")
    elif avg_no_sinyal > 50:
        st.error("⚠️ **Cluster Prioritas**: Wilayah ini memiliki angka 'Tidak Ada Sinyal' yang cukup tinggi. Butuh perhatian infrastruktur.")
    else:
        st.info("ℹ️ **Cluster Berkembang**: Wilayah dengan kualitas sinyal rata-rata dan sedang dalam tahap pengembangan.")

# --- 8. TABEL DATA ---
st.markdown("---")
st.subheader("📋 Tabel Data Cluster")
st.dataframe(
    filtered_df[['KABUPATEN JAWA BARAT', 'BTS', 'SINYAL KUAT', 'SINYAL LEMAH', 'TIDAK ADA SINYAL', '4G/LTE']], 
    use_container_width=True
)

