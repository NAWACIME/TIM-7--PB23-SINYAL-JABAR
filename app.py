import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard Sinyal Jabar Kelompok 7",
    page_icon="📡",
    layout="wide"
)

# --- STYLE TAMPILAN (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #e0e0e0;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD DATA & LOGIC CLUSTERING ---
@st.cache_data
def get_clustered_data():
    # Memuat file
    df = pd.read_csv("Sinyal.csv")
    
    # Fitur Analisis
    features = ['BTS', 'SINYAL KUAT', 'SINYAL LEMAH', 'TIDAK ADA SINYAL', '4G/LTE']
    X = df[features]
    
    # Standarisasi & K-Means (K=5)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    return df

# Eksekusi Data
try:
    df_final = get_clustered_data()
except Exception as e:
    st.error(f"Gagal memuat file Sinyal.csv. Pastikan file ada di folder yang sama. Error: {e}")
    st.stop()

# --- SIDEBAR (NAVIGASI) ---
with st.sidebar:
    st.title("📡 Navigasi")
    # Fitur: Hanya Opsi 0, 1, 2, 3, 4
    selected_cluster = st.selectbox("Pilih Opsi Cluster:", [0, 1, 2, 3, 4])
    
    st.markdown("---")
    st.write("### 👥 Kelompok 7")
    st.info("1. Naura Afnandita\n2. Maura Azzahra\n3. Mimma desmaya\n4. Maustika Taulina")

# --- JUDUL UTAMA ---
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>KEKUATAN SINYAL DI KABUPATEN JAWA BARAT</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Analisis Clustering K-Means Infrastruktur Telekomunikasi</p>", unsafe_allow_html=True)
st.markdown("---")

# Filter data berdasarkan cluster terpilih
filtered_df = df_final[df_final['Cluster'] == selected_cluster]

# --- KOLOM METRIK UTAMA ---
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("Jumlah Kabupaten", len(filtered_df))
with col_m2:
    st.metric("Total BTS (Cluster Ini)", f"{int(filtered_df['BTS'].sum())}")
with col_m3:
    st.metric("Rata-rata Sinyal Kuat", f"{filtered_df['SINYAL KUAT'].mean():.1f}")

# --- NARASI INTERPRETASI (Poin Tinggi) ---
st.subheader(f"📍 Analisis Hasil Opsi Cluster {selected_cluster}")

interpretasi = {
    0: ("Wilayah Infrastruktur Sangat Tinggi (Hub Utama)", "success", 
        "Wilayah dengan jumlah BTS sangat masif (seperti Bogor/Cirebon). Memerlukan pemeliharaan rutin jaringan."),
    1: ("Wilayah Sinyal Kuat & Gangguan Menengah", "info", 
        "Sinyal kuat tersedia luas, namun sinyal lemah masih cukup banyak. Perlu optimasi kualitas antena."),
    2: ("Wilayah Berkembang (BTS Menengah)", "info", 
        "Memiliki jumlah infrastruktur yang sedang. Fokus pada peningkatan kapasitas jaringan secara bertahap."),
    3: ("Wilayah dengan Tantangan Sinyal", "warning", 
        "Sinyal lemah atau 'Tidak Ada Sinyal' lebih menonjol. Perlu pembangunan BTS penguat (Repeater)."),
    4: ("Wilayah Cakupan Minimal / Terpencil", "error", 
        "Wilayah dengan akses sinyal paling rendah. Prioritas utama pembangunan tower BTS baru.")
}

nama_cluster, status, deskripsi = interpretasi.get(selected_cluster)

if status == "success": st.success(f"**Karakteristik:** {nama_cluster}\n\n**Kesimpulan:** {deskripsi}")
elif status == "info": st.info(f"**Karakteristik:** {nama_cluster}\n\n**Kesimpulan:** {deskripsi}")
elif status == "warning": st.warning(f"**Karakteristik:** {nama_cluster}\n\n**Kesimpulan:** {deskripsi}")
else: st.error(f"**Karakteristik:** {nama_cluster}\n\n**Kesimpulan:** {deskripsi}")

# --- VISUALISASI PLOTLY ---
st.markdown("---")
col_chart, col_pie = st.columns([2, 1])

with col_chart:
    st.subheader("📊 Perbandingan Sinyal Kuat vs Lemah")
    fig = px.bar(
        filtered_df,
        x='KABUPATEN JAWA BARAT',
        y=['SINYAL KUAT', 'SINYAL LEMAH'],
        barmode='group',
        color_discrete_map={'SINYAL KUAT': '#2ecc71', 'SINYAL LEMAH': '#e74c3c'},
        template="plotly_white"
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with col_pie:
    st.subheader("📶 Distribusi 4G/LTE")
    fig_pie = px.pie(
        filtered_df,
        values='4G/LTE',
        names='KABUPATEN JAWA BARAT',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# --- TABEL DATA ---
st.markdown("---")
st.subheader("📋 Detail Data Cluster")
st.dataframe(filtered_df.drop(columns=['Cluster']), use_container_width=True)

st.caption("Dibuat oleh Kelompok 7 - Tugas Analisis Data K-Means")