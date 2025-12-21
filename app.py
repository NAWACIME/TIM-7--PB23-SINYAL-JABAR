import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 1. SETUP HALAMAN ---
st.set_page_config(
    page_title="Peta Sebaran Sinyal Indonesia (Jabar)",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS CUSTOM ---
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    h1, h2, h3 { color: #1e40af; }
    [data-testid="stMetric"] {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. DATA KOORDINAT (MANUAL) ---
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

# --- 4. ENGINE DATA & MODELING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Sinyal.csv")
    except:
        try:
            df = pd.read_csv("sinyal.csv")
        except:
            return None, "File CSV tidak ditemukan!"

    # Scaling & Clustering
    scaler = StandardScaler()
    features = ['SINYAL KUAT', 'SINYAL LEMAH', 'TIDAK ADA SINYAL', '4G/LTE']
    X_scaled = scaler.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # PCA
    pca = PCA(n_components=2)
    comps = pca.fit_transform(X_scaled)
    df['PCA1'], df['PCA2'] = comps[:, 0], comps[:, 1]
    
    # Koordinat
    df['Temp'] = df['KABUPATEN JAWA BARAT'].str.upper().str.strip()
    df['Lat'] = df['Temp'].map(lambda x: coords_jabar.get(x, [None,None])[0])
    df['Lon'] = df['Temp'].map(lambda x: coords_jabar.get(x, [None,None])[1])
    
    # Hitung Total
    df['Total_Trafik'] = df['SINYAL KUAT'] + df['SINYAL LEMAH'] + df['TIDAK ADA SINYAL']
    
    return df.dropna(subset=['Lat', 'Lon']), None

df_final, err = load_data()
if err: st.error(err); st.stop()

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("🎛️ Kontrol Peta")
    
    # 1. Filter Cluster
    st.subheader("Filter Cluster")
    clus_opts = sorted(df_final['Cluster'].unique())
    sel_clus = st.multiselect("Pilih Cluster:", clus_opts, default=clus_opts)
    
    st.markdown("---")
    
    # 2. PILIHAN METRIK VISUALISASI PETA (FITUR BARU)
    st.subheader("🌍 Visualisasi Peta")
    st.info("Pilih data yang ingin ditonjolkan ukurannya di peta:")
    map_metric = st.radio(
        "Ukuran Bubble Berdasarkan:",
        ["Total Trafik", "Sinyal Kuat", "Sinyal Lemah", "Tidak Ada Sinyal", "4G/LTE"]
    )
    
    # Mapping nama pilihan ke nama kolom DataFrame
    col_map = {
        "Total Trafik": "Total_Trafik",
        "Sinyal Kuat": "SINYAL KUAT",
        "Sinyal Lemah": "SINYAL LEMAH",
        "Tidak Ada Sinyal": "TIDAK ADA SINYAL",
        "4G/LTE": "4G/LTE"
    }
    selected_col = col_map[map_metric]

    df_filtered = df_final[df_final['Cluster'].isin(sel_clus)]

# --- 6. DASHBOARD CONTENT ---
st.title(f"📍 Peta Sebaran: {map_metric}")
st.markdown("Besar lingkaran menunjukkan jumlah sinyal kategori yang dipilih. Warna menunjukkan Cluster.")

# KPI
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Wilayah", len(df_filtered))
c2.metric("Total Sinyal Kuat", f"{int(df_filtered['SINYAL KUAT'].sum()):,}")
c3.metric("Total No Signal", f"{int(df_filtered['TIDAK ADA SINYAL'].sum()):,}")
c4.metric("Avg 4G Coverage", f"{df_filtered['4G/LTE'].mean():.1f}")

st.markdown("---")

col_map_view, col_charts = st.columns([1.5, 1])

# --- BAGIAN PETA (FOLIUM) ---
with col_map_view:
    # Logic untuk scaling ukuran bubble agar tidak terlalu kecil/besar
    max_val = df_final[selected_col].max()
    
    m = folium.Map(location=[-6.95, 107.65], zoom_start=9, tiles="CartoDB positron")
    
    colors = {0: '#e74c3c', 1: '#3498db', 2: '#2ecc71', 3: '#f1c40f'} # Merah, Biru, Hijau, Kuning
    
    for _, row in df_filtered.iterrows():
        # Rumus dinamis ukuran lingkaran
        val = row[selected_col]
        # Scaling: (Value / Max_Value) * Max_Radius + Min_Radius
        radius = (val / max_val) * 25 + 5 
        
        color = colors.get(row['Cluster'], 'gray')
        
        # Popup Lebih Detail
        popup_html = f"""
        <div style='font-family:sans-serif; width:180px;'>
            <b>{row['KABUPATEN JAWA BARAT']}</b><br>
            <span style='color:{color}; font-weight:bold;'>Cluster {row['Cluster']}</span>
            <hr style='margin:5px 0;'>
            <b>Data Terpilih ({map_metric}):</b><br>
            <span style='font-size:16px; font-weight:bold;'>{val:,.0f}</span><br>
            <br>
            <small>
            Kuat: {row['SINYAL KUAT']}<br>
            Lemah: {row['SINYAL LEMAH']}<br>
            No Sinyal: {row['TIDAK ADA SINYAL']}<br>
            4G: {row['4G/LTE']}
            </small>
        </div>
        """
        
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row['KABUPATEN JAWA BARAT']}: {val}"
        ).add_to(m)

    st_folium(m, width="100%", height=500)
    
    # Legenda Manual
    st.markdown("""
    <div style="text-align:center; font-size:12px;">
    🔴 Cluster 0 | 🔵 Cluster 1 | 🟢 Cluster 2 | 🟡 Cluster 3
    </div>
    """, unsafe_allow_html=True)

# --- BAGIAN GRAFIK ---
with col_charts:
    st.subheader(f"🏆 Top 10 Wilayah ({map_metric})")
    
    # Bar Chart Top 10 berdasarkan Metric yang dipilih di Sidebar
    top_df = df_filtered.sort_values(by=selected_col, ascending=True).tail(10)
    
    fig_top = px.bar(
        top_df,
        x=selected_col,
        y="KABUPATEN JAWA BARAT",
        orientation='h',
        color="Cluster",
        color_discrete_map=colors,
        text_auto='.2s',
        title=f"Ranking Daerah: {map_metric}",
        template="plotly_white"
    )
    fig_top.update_layout(height=500)
    st.plotly_chart(fig_top, use_container_width=True)

# --- TABEL DATA ---
with st.expander("📂 Lihat Detail Data Lengkap"):
    st.dataframe(df_filtered.drop(columns=['Temp','Lat','Lon','PCA1','PCA2']), use_container_width=True)
