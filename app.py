import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
    
    /* Warna angka metrik */
    [data-testid="stMetricValue"] {
        color: #1E3A8A;
        font-weight: bold;
    }
    
    /* Warna label metrik */
    [data-testid="stMetricLabel"] {
        color: #333333;
    }

    /* Kotak Metrik */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Tabel */
    .stDataFrame {
        color: #000000;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DICTIONARY KOORDINAT JAWA BARAT (MANUAL) ---
# Koordinat ini ditambahkan manual karena tidak ada di dataset asli
coords_jabar = {
    "KABUPATEN BOGOR": [-6.5518, 106.6291],
    "KABUPATEN SUKABUMI": [-7.0601, 106.7214],
    "KABUPATEN CIANJUR": [-6.8205, 107.1416],
    "KABUPATEN BANDUNG": [-7.0253, 107.5197],
    "KABUPATEN GARUT": [-7.2279, 107.9087],
    "KABUPATEN TASIKMALAYA": [-7.3506, 108.1065],
    "KABUPATEN CIAMIS": [-7.3274, 108.3542],
    "KABUPATEN KUNINGAN": [-6.9775, 108.4862],
    "KABUPATEN CIREBON": [-6.7372, 108.5507],
    "KABUPATEN MAJALENGKA": [-6.8371, 108.2274],
    "KABUPATEN SUMEDANG": [-6.8385, 107.9272],
    "KABUPATEN INDRAMAYU": [-6.4429, 108.1738],
    "KABUPATEN SUBANG": [-6.5716, 107.7587],
    "KABUPATEN PURWAKARTA": [-6.5561, 107.4426],
    "KABUPATEN KARAWANG": [-6.3195, 107.3060],
    "KABUPATEN BEKASI": [-6.2416, 107.1456],
    "KABUPATEN BANDUNG BARAT": [-6.8437, 107.5029],
    "KABUPATEN PANGANDARAN": [-7.6976, 108.4975],
    "KOTA BOGOR": [-6.5971, 106.7991],
    "KOTA SUKABUMI": [-6.9277, 106.9300],
    "KOTA BANDUNG": [-6.9175, 107.6191],
    "KOTA CIREBON": [-6.7320, 108.5523],
    "KOTA BEKASI": [-6.2383, 106.9756],
    "KOTA DEPOK": [-6.4025, 106.7942],
    "KOTA CIMAHI": [-6.8715, 107.5457],
    "KOTA TASIKMALAYA": [-7.3274, 108.2207],
    "KOTA BANJAR": [-7.3685, 108.5310]
}

# --- 4. LOAD DATA, CLUSTERING, & PCA ---
@st.cache_data
def get_clustered_data():
    df = pd.read_csv("Sinyal.csv")
    
    # Preprocessing
    features = ['SINYAL KUAT', 'SINYAL LEMAH', 'TIDAK ADA SINYAL', '4G/LTE']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # PCA untuk Scatter Plot (Reduksi ke 2 Dimensi)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    df['PCA1'] = principal_components[:, 0]
    df['PCA2'] = principal_components[:, 1]
    
    # Mapping Koordinat
    # Kita pastikan nama di CSV uppercase agar cocok dengan dictionary
    df['Temp_Name'] = df['KABUPATEN JAWA BARAT'].str.upper().str.strip()
    df['Latitude'] = df['Temp_Name'].map(lambda x: coords_jabar.get(x, [None, None])[0])
    df['Longitude'] = df['Temp_Name'].map(lambda x: coords_jabar.get(x, [None, None])[1])
    
    # Hapus baris yang tidak punya koordinat (jika ada nama kota yang tidak match)
    # df = df.dropna(subset=['Latitude', 'Longitude']) 
    
    return df

try:
    df_final = get_clustered_data()
except Exception as e:
    st.error(f"Gagal memuat file Sinyal.csv. Error: {e}")
    st.stop()

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("📡 Navigasi")
    selected_cluster_view = st.selectbox("Filter Tampilan Cluster (Data/Bar):", ["Semua"] + sorted(list(df_final['Cluster'].unique())))
    st.markdown("---")
    st.write("### 👥 Kelompok 7")
    st.info("1. Naura Afnandita\n2. Maura Azzahra\n3. Mimma Desmaya\n4. Mustika Taulina")

# --- 6. JUDUL ---
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>ANALISIS KUALITAS SINYAL & 4G JAWA BARAT</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Berdasarkan Pemodelan K-Means Clustering & Geospatial Analysis</p>", unsafe_allow_html=True)
st.markdown("---")

# Filter data untuk metrik dan tabel
if selected_cluster_view != "Semua":
    filtered_df = df_final[df_final['Cluster'] == selected_cluster_view]
else:
    filtered_df = df_final

# --- 7. METRIK UTAMA ---
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Total Wilayah", f"{len(filtered_df)}")
with m2:
    st.metric("Avg Sinyal Kuat", f"{filtered_df['SINYAL KUAT'].mean():.1f}")
with m3:
    st.metric("Avg Sinyal Lemah", f"{filtered_df['SINYAL LEMAH'].mean():.1f}")
with m4:
    st.metric("Avg No Sinyal", f"{filtered_df['TIDAK ADA SINYAL'].mean():.1f}")
with m5:
    st.metric("Avg 4G/LTE", f"{filtered_df['4G/LTE'].mean():.1f}")

st.markdown("---")

# --- 8. VISUALISASI MATPLOTLIB (PCA) & PLOTLY (BAR) ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📍 Scatter Plot Cluster (PCA - Matplotlib)")
    
    # Membuat Scatter Plot dengan Matplotlib
    fig_pca, ax = plt.subplots(figsize=(8, 6))
    
    # Warna untuk setiap cluster
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f'] # Merah, Biru, Hijau, Kuning
    
    for cluster_id in sorted(df_final['Cluster'].unique()):
        subset = df_final[df_final['Cluster'] == cluster_id]
        ax.scatter(
            subset['PCA1'], 
            subset['PCA2'], 
            c=colors[cluster_id], 
            label=f'Cluster {cluster_id}',
            edgecolor='k',
            s=100,
            alpha=0.7
        )
    
    ax.set_title('Distribusi Cluster (Reduksi Dimensi PCA)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig_pca)

with col2:
    st.subheader(f"📊 Karakteristik Sinyal")
    # Tampilkan grafik bar (jika 'Semua' terlalu ramai, ambil rata-rata per cluster)
    if selected_cluster_view == "Semua":
        # Group by Cluster untuk ringkasan
        grouped = df_final.groupby('Cluster')[['SINYAL KUAT', 'SINYAL LEMAH', 'TIDAK ADA SINYAL', '4G/LTE']].mean().reset_index()
        fig_bar = px.bar(
            grouped,
            x='Cluster',
            y=['SINYAL KUAT', 'SINYAL LEMAH', 'TIDAK ADA SINYAL', '4G/LTE'],
            barmode='group',
            title="Rata-rata Sinyal per Cluster",
            template="plotly_white"
        )
    else:
        # Tampilkan detail per kota jika filter aktif
        fig_bar = px.bar(
            filtered_df,
            x='KABUPATEN JAWA BARAT',
            y=['SINYAL KUAT', 'SINYAL LEMAH', 'TIDAK ADA SINYAL', '4G/LTE'],
            barmode='group',
            template="plotly_white"
        )
    st.plotly_chart(fig_bar, use_container_width=True)

# --- 9. PETA INTERAKTIF (FOLIUM) ---
st.markdown("---")
st.subheader("🗺️ Peta Persebaran Cluster di Jawa Barat")

# Inisialisasi Peta (Center di Jawa Barat)
m = folium.Map(location=[-6.9, 107.6], zoom_start=9)

# Warna Marker sesuai Cluster (sama dengan scatter plot)
folium_colors = ['red', 'blue', 'green', 'orange'] 

# Loop data untuk membuat marker
for index, row in df_final.iterrows():
    # Cek jika lat/long valid (tidak NaN)
    if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude']):
        
        # Konten Popup HTML
        popup_html = f"""
        <div style="width:200px">
            <b>{row['KABUPATEN JAWA BARAT']}</b><br>
            <hr style="margin:5px 0">
            <b>Cluster:</b> {row['Cluster']}<br>
            <b>Sinyal Kuat:</b> {row['SINYAL KUAT']}<br>
            <b>Sinyal Lemah:</b> {row['SINYAL LEMAH']}<br>
            <b>No Signal:</b> {row['TIDAK ADA SINYAL']}<br>
            <b>4G/LTE:</b> {row['4G/LTE']}
        </div>
        """
        
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=row['KABUPATEN JAWA BARAT'],
            icon=folium.Icon(color=folium_colors[row['Cluster']], icon="signal", prefix="fa")
        ).add_to(m)

# Tampilkan Peta
st_folium(m, width=1200, height=500)

# --- 10. TABEL DATA ---
st.markdown("---")
st.subheader("📋 Detail Data")
st.dataframe(
    filtered_df[['KABUPATEN JAWA BARAT', 'Cluster', 'BTS', 'SINYAL KUAT', 'SINYAL LEMAH', 'TIDAK ADA SINYAL', '4G/LTE']], 
    use_container_width=True
)
