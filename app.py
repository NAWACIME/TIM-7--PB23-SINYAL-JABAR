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
    
    /* Style Kotak Metrik */
    [data-testid="stMetricValue"] {
        color: #1E3A8A; /* Biru Gelap */
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        color: #333333;
    }
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .stDataFrame { color: #000000; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA KOORDINAT JAWA BARAT (HARDCODED) ---
# Kita perlu ini karena CSV tidak memiliki Lat/Long
coords = {
    "BOGOR": [-6.5971, 106.8060],
    "SUKABUMI": [-6.9277, 106.9300],
    "CIANJUR": [-6.8168, 107.1425],
    "BANDUNG": [-6.9175, 107.6191],
    "GARUT": [-7.2279, 107.9087],
    "TASIKMALAYA": [-7.3274, 108.2207],
    "CIAMIS": [-7.3333, 108.3500],
    "KUNINGAN": [-6.9744, 108.4800],
    "CIREBON": [-6.7320, 108.5523],
    "MAJALENGKA": [-6.8358, 108.2274],
    "SUMEDANG": [-6.8586, 107.9266],
    "INDRAMAYU": [-6.3264, 108.3200],
    "SUBANG": [-6.5716, 107.7587],
    "PURWAKARTA": [-6.5387, 107.4499],
    "KARAWANG": [-6.3042, 107.3079],
    "BEKASI": [-6.2383, 106.9756],
    "BANDUNG BARAT": [-6.8436, 107.5113],
    "PANGANDARAN": [-7.6976, 108.6539],
    "KOTA BOGOR": [-6.5950, 106.8166],
    "KOTA SUKABUMI": [-6.9237, 106.9287],
    "KOTA BANDUNG": [-6.9147, 107.6098],
    "KOTA CIREBON": [-6.7053, 108.5554],
    "KOTA BEKASI": [-6.2349, 106.9924],
    "KOTA DEPOK": [-6.4025, 106.7942],
    "KOTA CIMAHI": [-6.8723, 107.5421],
    "KOTA TASIKMALAYA": [-7.3506, 108.2177],
    "KOTA BANJAR": [-7.3667, 108.5333]
}

# --- 4. LOAD DATA & LOGIC CLUSTERING ---
@st.cache_data
def get_clustered_data():
    df = pd.read_csv("Sinyal.csv")
    
    # 1. Clustering Logic
    features = ['SINYAL KUAT', 'SINYAL LEMAH', 'TIDAK ADA SINYAL', '4G/LTE']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # 2. Mapping Koordinat (Lat/Lon)
    # Mencocokkan Nama Kabupaten dengan Dictionary 'coords'
    df['lat'] = df['KABUPATEN JAWA BARAT'].map(lambda x: coords.get(x, [None, None])[0])
    df['lon'] = df['KABUPATEN JAWA BARAT'].map(lambda x: coords.get(x, [None, None])[1])
    
    # Isi data kosong dengan rata-rata Jabar (fallback jika nama kota beda ejaan)
    df['lat'] = df['lat'].fillna(-6.9175)
    df['lon'] = df['lon'].fillna(107.6191)
    
    return df

try:
    df_final = get_clustered_data()
except Exception as e:
    st.error(f"Gagal memuat file Sinyal.csv. Pastikan file ada di folder yang sama. Error: {e}")
    st.stop()

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("📡 Navigasi")
    # Pilihan Cluster
    selected_cluster = st.selectbox("Pilih Opsi Cluster:", sorted(df_final['Cluster'].unique()))
    
    st.markdown("---")
    st.write("### ⚙️ Pengaturan Peta")
    # Pilihan Variabel Peta
    map_metric = st.radio(
        "Tampilkan Sebaran Berdasarkan:",
        ('SINYAL KUAT', 'SINYAL LEMAH', 'TIDAK ADA SINYAL', '4G/LTE')
    )
    
    st.markdown("---")
    st.write("### 👥 Kelompok 7")
    st.info("1. Naura Afnandita\n2. Maura Azzahra\n3. Mimma Desmaya\n4. Mustika Taulina")

# --- 6. JUDUL ---
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>ANALISIS KUALITAS SINYAL & 4G JAWA BARAT</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Berdasarkan Pemodelan K-Means Clustering</p>", unsafe_allow_html=True)
st.markdown("---")

# Filter data berdasarkan cluster yang dipilih
filtered_df = df_final[df_final['Cluster'] == selected_cluster]

# --- 7. METRIK UTAMA ---
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

# --- 8. VISUALISASI PETA (BARU) ---
st.markdown("---")
st.subheader(f"🗺️ Peta Sebaran: {map_metric} (Cluster {selected_cluster})")

# Logic Warna Peta agar dinamis
color_scale_map = {
    'SINYAL KUAT': 'Greens',
    'SINYAL LEMAH': 'Oranges',
    'TIDAK ADA SINYAL': 'Reds',
    '4G/LTE': 'Blues'
}

# Membuat Peta Scatter Mapbox
fig_map = px.scatter_mapbox(
    filtered_df,
    lat="lat",
    lon="lon",
    size=map_metric,           # Ukuran bulatan berdasarkan nilai sinyal
    color=map_metric,          # Warna juga berdasarkan nilai sinyal
    color_continuous_scale=color_scale_map[map_metric],
    size_max=30,               # Ukuran maksimal bulatan
    zoom=7.5,
    center={"lat": -6.9175, "lon": 107.6191}, # Center di Jawa Barat
    hover_name="KABUPATEN JAWA BARAT",
    hover_data={'lat': False, 'lon': False, 'BTS': True},
    mapbox_style="carto-positron", # Style peta gratis (OpenStreetMap based)
    title=f"Sebaran Geografis {map_metric} di Wilayah Cluster {selected_cluster}"
)

# Render Peta
st.plotly_chart(fig_map, use_container_width=True)

# --- 9. VISUALISASI GRAFIK BATANG & INTERPRETASI ---
st.markdown("---")
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"📊 Detail Perbandingan Variabel Cluster {selected_cluster}")
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
    st.subheader("💡 Interpretasi Data")
    avg_4g = filtered_df['4G/LTE'].mean()
    avg_no_sinyal = filtered_df['TIDAK ADA SINYAL'].mean()
    
    st.markdown(f"**Analisis Wilayah:**")
    st.write(f"Menampilkan data untuk **{len(filtered_df)} kabupaten/kota** dalam cluster ini.")
    
    if avg_4g > 500:
        st.success("✅ **Cluster Unggul**\nWilayah ini memiliki jangkauan 4G dan sinyal kuat yang sangat dominan. Cocok untuk layanan digital intensif.")
    elif avg_no_sinyal > 50:
        st.error("⚠️ **Cluster Prioritas**\nWilayah ini memiliki angka 'Tidak Ada Sinyal' yang cukup tinggi. Memerlukan penambahan BTS atau optimasi jaringan segera.")
    else:
        st.info("ℹ️ **Cluster Berkembang**\nWilayah dengan kualitas sinyal rata-rata. Cukup stabil namun masih memiliki potensi untuk ditingkatkan.")

# --- 10. TABEL DATA ---
st.markdown("---")
st.subheader("📋 Tabel Data Detail")
st.dataframe(
    filtered_df[['KABUPATEN JAWA BARAT', 'Cluster', 'BTS', 'SINYAL KUAT', 'SINYAL LEMAH', 'TIDAK ADA SINYAL', '4G/LTE']], 
    use_container_width=True
)
