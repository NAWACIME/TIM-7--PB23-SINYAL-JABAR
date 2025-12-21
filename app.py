import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 1. SETUP & KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard Sinyal Jawa Barat",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS CUSTOM (UNTUK MEMPERCANTIK TAMPILAN) ---
st.markdown("""
<style>
    /* Background utama */
    .stApp {
        background-color: #f4f6f9;
    }
    
    /* Styling Header */
    h1, h2, h3 {
        color: #0f172a;
        font-family: 'Sans-serif';
    }
    
    /* Styling Metrics (Kotak Angka) */
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #3b82f6;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #64748b;
        font-size: 14px;
    }
    
    div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-size: 24px;
        font-weight: 700;
    }
    
    /* Styling Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. DATA KOORDINAT JAWA BARAT (MANUAL) ---
coords_jabar = {
    "KABUPATEN BOGOR": [-6.5518, 106.6291], "KABUPATEN SUKABUMI": [-7.0601, 106.7214],
    "KABUPATEN CIANJUR": [-6.8205, 107.1416], "KABUPATEN BANDUNG": [-7.0253, 107.5197],
    "KABUPATEN GARUT": [-7.2279, 107.9087], "KABUPATEN TASIKMALAYA": [-7.3506, 108.1065],
    "KABUPATEN CIAMIS": [-7.3274, 108.3542], "KABUPATEN KUNINGAN": [-6.9775, 108.4862],
    "KABUPATEN CIREBON": [-6.7372, 108.5507], "KABUPATEN MAJALENGKA": [-6.8371, 108.2274],
    "KABUPATEN SUMEDANG": [-6.8385, 107.9272], "KABUPATEN INDRAMAYU": [-6.4429, 108.1738],
    "KABUPATEN SUBANG": [-6.5716, 107.7587], "KABUPATEN PURWAKARTA": [-6.5561, 107.4426],
    "KABUPATEN KARAWANG": [-6.3195, 107.3060], "KABUPATEN BEKASI": [-6.2416, 107.1456],
    "KABUPATEN BANDUNG BARAT": [-6.8437, 107.5029], "KABUPATEN PANGANDARAN": [-7.6976, 108.4975],
    "KOTA BOGOR": [-6.5971, 106.7991], "KOTA SUKABUMI": [-6.9277, 106.9300],
    "KOTA BANDUNG": [-6.9175, 107.6191], "KOTA CIREBON": [-6.7320, 108.5523],
    "KOTA BEKASI": [-6.2383, 106.9756], "KOTA DEPOK": [-6.4025, 106.7942],
    "KOTA CIMAHI": [-6.8715, 107.5457], "KOTA TASIKMALAYA": [-7.3274, 108.2207],
    "KOTA BANJAR": [-7.3685, 108.5310]
}

# --- 4. ENGINE DATA & MODELING (VALIDASI LOGIC) ---
@st.cache_data
def process_data_and_model():
    # Load Data
    try:
        df = pd.read_csv("Sinyal.csv")
    except FileNotFoundError:
        return None, "File 'Sinyal.csv' tidak ditemukan."

    # Fitur untuk Clustering
    features = ['SINYAL KUAT', 'SINYAL LEMAH', 'TIDAK ADA SINYAL', '4G/LTE']
    
    # 1. Scaling (Penting agar valid dengan Notebook)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    # 2. Modeling K-Means (Random State 42 agar hasil konsisten)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # 3. PCA untuk Visualisasi 2D Scatter Plot
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    df['PCA1'] = principal_components[:, 0]
    df['PCA2'] = principal_components[:, 1]
    
    # 4. Mapping Koordinat
    df['Temp_Name'] = df['KABUPATEN JAWA BARAT'].str.upper().str.strip()
    df['Latitude'] = df['Temp_Name'].map(lambda x: coords_jabar.get(x, [None, None])[0])
    df['Longitude'] = df['Temp_Name'].map(lambda x: coords_jabar.get(x, [None, None])[1])
    
    # 5. Total Traffic (Untuk Bubble Size)
    df['Total_Sinyal'] = df['SINYAL KUAT'] + df['SINYAL LEMAH'] + df['TIDAK ADA SINYAL']
    
    # Hapus data yang tidak punya koordinat
    df_clean = df.dropna(subset=['Latitude', 'Longitude'])
    
    return df_clean, None

# Load Data
df_final, error_msg = process_data_and_model()

if error_msg:
    st.error(error_msg)
    st.stop()

# --- 5. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910793.png", width=80)
    st.title("Sinyal Analytics")
    st.write("Dashboard Monitoring Kualitas Jaringan Jawa Barat")
    st.markdown("---")
    
    # Filter
    st.subheader("🔍 Filter Data")
    cluster_options = sorted(df_final['Cluster'].unique())
    selected_clusters = st.multiselect("Pilih Cluster:", cluster_options, default=cluster_options)
    
    if not selected_clusters:
        st.warning("Mohon pilih minimal satu cluster.")
        st.stop()
        
    filtered_df = df_final[df_final['Cluster'].isin(selected_clusters)]
    
    st.markdown("---")
    st.write("© Kelompok 7")

# --- 6. MAIN CONTENT ---

# Header
st.title("📡 Dashboard Analisis Sinyal & 4G")
st.markdown("Pemodelan K-Means Clustering untuk mengelompokkan wilayah berdasarkan kualitas sinyal di Jawa Barat.")

# KPI / Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Wilayah", f"{len(filtered_df)}")
with col2:
    st.metric("Total BTS", f"{int(filtered_df['BTS'].sum()):,}")
with col3:
    st.metric("Rata-rata 4G/LTE", f"{filtered_df['4G/LTE'].mean():.1f}")
with col4:
    st.metric("Avg Sinyal Kuat", f"{filtered_df['SINYAL KUAT'].mean():.1f}")

st.markdown("---")

# --- ROW 1: PETA & SCATTER PLOT ---
row1_col1, row1_col2 = st.columns([1.3, 1])

# A. PETA INTERAKTIF (BUBBLE MAP)
with row1_col1:
    st.subheader("🗺️ Peta Geografis Cluster")
    
    m = folium.Map(location=[-6.92, 107.60], zoom_start=9, tiles="CartoDB positron")
    
    # Warna Cluster
    colors = {0: '#EF4444', 1: '#3B82F6', 2: '#10B981', 3: '#F59E0B'} # Merah, Biru, Hijau, Kuning
    
    for _, row in filtered_df.iterrows():
        # Logic ukuran bubble
        radius = row['Total_Sinyal'] / 600
        radius = max(5, min(radius, 25))
        
        c_color = colors.get(row['Cluster'], 'gray')
        
        popup_html = f"""
        <b>{row['KABUPATEN JAWA BARAT']}</b><br>
        Cluster: {row['Cluster']}<br>
        Sinyal Kuat: {row['SINYAL KUAT']}<br>
        4G: {row['4G/LTE']}
        """
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=radius,
            color=c_color,
            fill=True,
            fill_color=c_color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{row['KABUPATEN JAWA BARAT']} (Cluster {row['Cluster']})"
        ).add_to(m)
        
    st_folium(m, width="100%", height=400)
    st.caption("*Besar lingkaran merepresentasikan total trafik sinyal.")

# B. SCATTER PLOT (PCA) - VALIDASI CLUSTERING
with row1_col2:
    st.subheader("📊 Sebaran Cluster (PCA)")
    
    fig_pca = px.scatter(
        filtered_df,
        x='PCA1',
        y='PCA2',
        color='Cluster',
        hover_data=['KABUPATEN JAWA BARAT'],
        color_discrete_map={
            0: '#EF4444', 1: '#3B82F6', 2: '#10B981', 3: '#F59E0B'
        },
        title="Reduksi Dimensi (Validasi Pemisahan Cluster)",
        template="plotly_white"
    )
    fig_pca.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    fig_pca.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_pca, use_container_width=True)

st.markdown("---")

# --- ROW 2: BAR CHART PER KABUPATEN ---
st.subheader("📈 Detail Kualitas Sinyal per Kabupaten")

# Sorting data agar grafik rapi (Urut berdasarkan Cluster lalu Nama)
plot_df = filtered_df.sort_values(by=['Cluster', 'SINYAL KUAT'], ascending=[True, False])

# Membuat Grouped Bar Chart
fig_bar = px.bar(
    plot_df,
    x='KABUPATEN JAWA BARAT',
    y=['SINYAL KUAT', 'SINYAL LEMAH', 'TIDAK ADA SINYAL', '4G/LTE'],
    barmode='group',
    title="Komparasi Variabel Sinyal Tiap Kabupaten",
    labels={'value': 'Jumlah / Frekuensi', 'variable': 'Kategori Sinyal'},
    color_discrete_map={
        'SINYAL KUAT': '#10B981',      # Hijau
        'SINYAL LEMAH': '#F59E0B',     # Kuning/Orange
        'TIDAK ADA SINYAL': '#EF4444', # Merah
        '4G/LTE': '#3B82F6'            # Biru
    },
    template="plotly_white"
)

# Kustomisasi Layout Grafik
fig_bar.update_layout(
    xaxis_tickangle=-45,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    height=500,
    margin=dict(l=20, r=20, t=50, b=100)
)

st.plotly_chart(fig_bar, use_container_width=True)

# --- ROW 3: DATAFRAME ---
with st.expander("📂 Lihat Data Mentah"):
    st.dataframe(
        filtered_df[['KABUPATEN JAWA BARAT', 'Cluster', 'BTS', 'SINYAL KUAT', 'SINYAL LEMAH', 'TIDAK ADA SINYAL', '4G/LTE']],
        use_container_width=True
    )
