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
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD DATA & LOGIC CLUSTERING (4 Variabel Sesuai .ipynb) ---
@st.cache_data
def get_clustered_data():
    df = pd.read_csv("Sinyal.csv")
    
    # VARIABEL MODELING: Sinyal Kuat, Lemah, Tidak Ada, dan 4G/LTE
    features = ['SINYAL KUAT', 'SINYAL LEMAH', 'TIDAK ADA SINYAL', '4G/LTE']
    X = df[features]
    
    # Standarisasi Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Menggunakan K=4 (Sesuai hasil Elbow Method di notebook Anda)
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
    # Memilih Cluster 0, 1, 2, atau 3
    selected_cluster = st.selectbox("Pilih Opsi Cluster:", sorted(df_final['Cluster'].unique()))
    st.markdown("---")
    st.write("### 👥 Kelompok 7")
    st.info("Naura, Maura, Mimma, Mustika")

# --- 5. JUDUL ---
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>ANALISIS KUALITAS SINYAL & 4G JAWA BARAT</h1>", unsafe_allow_html=True)
st.markdown("---")

# Filter data
filtered_df = df_final[df_final['Cluster'] == selected_cluster]

# --- 6. METRIK UTAMA (4 VARIABEL LENGKAP) ---
# Menampilkan rata-rata performa cluster untuk ke-4 variabel
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

# --- 7. VISUALISASI BAR CHART (4 VARIABEL) ---
st.markdown("---")
st.subheader(f"📊 Perbandingan 4 Variabel Modeling di Cluster {selected_cluster}")

# Menampilkan grafik batang untuk semua variabel modeling
fig_bar = px.bar(
    filtered_df,
    x='KABUPATEN JAWA BARAT',
    y=['SINYAL KUAT', 'SINYAL LEMAH', 'TIDAK ADA SINYAL', '4G/LTE'], 
    barmode='group',
    color_discrete_map={
        'SINYAL KUAT': '#2ecc71',      # Hijau
        'SINYAL LEMAH': '#f1c40f',     # Kuning
        'TIDAK ADA SINYAL': '#e74c3c', # Merah
        '4G/LTE': '#3498db'            # Biru
    },
    template="plotly_white"
)
fig_bar.update_layout(xaxis_title="Kabupaten/Kota", yaxis_title="Jumlah Desa")
st.plotly_chart(fig_bar, use_container_width=True)

# --- 8. TABEL DETAIL ---
st.markdown("---")
st.subheader("📋 Tabel Data Cluster")
st.dataframe(
    filtered_df[['KABUPATEN JAWA BARAT', 'BTS', 'SINYAL KUAT', 'SINYAL LEMAH', 'TIDAK ADA SINYAL', '4G/LTE']], 
    use_container_width=True
)

