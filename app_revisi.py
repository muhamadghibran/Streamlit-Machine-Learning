import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist
import warnings
import io

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

st.set_page_config(
    page_title="Analisis Clustering Bencana Alam Indonesia", 
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4ECDC4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .warning-box {
        background: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404 !important;
    }
    .success-box {
        background: #d4edda;
        border: 2px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724 !important;
    }
    .error-box {
        background: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #721c24 !important;
    }
    .info-box {
        background: #d1ecf1;
        border: 2px solid #17a2b8;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #0c5460 !important;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stSelectbox > div > div {
        background-color: white;
        color: #333333;
    }
    .stMultiSelect > div > div {
        background-color: white;
        color: #333333;
    }
    /* Perbaikan kontras untuk teks */
    .stMarkdown {
        color: #333333;
    }
    .stDataFrame {
        background-color: white;
    }
    /* Perbaikan untuk expander */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        color: #333333 !important;
        border: 1px solid #dee2e6 !important;
    }
    .streamlit-expanderContent {
        background-color: white !important;
        color: #333333 !important;
    }
    /* Perbaikan untuk tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        color: #333333;
        border: 1px solid #dee2e6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #007bff;
        color: white;
    }
    /* Perbaikan untuk tab content */
    .stTabs [data-baseweb="tab-panel"] {
        background-color: white;
        color: #333333;
        padding: 1rem;
    }
    /* Perbaikan untuk text di dalam tab */
    .stTabs [data-baseweb="tab-panel"] h4 {
        color: #333333 !important;
    }
    .stTabs [data-baseweb="tab-panel"] p {
        color: #333333 !important;
    }
    .stTabs [data-baseweb="tab-panel"] .stMarkdown {
        color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🌋 Analisis Clustering Wilayah Rawan Bencana Alam Indonesia</h1>
    <p>Sistem Pengelompokan Wilayah Berdasarkan Tingkat Kerawanan Bencana Menggunakan Machine Learning</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    np.random.seed(42)
    provinces = ['Jawa Barat', 'Jawa Tengah', 'Jawa Timur', 'Sumatra Utara', 'Sumatra Barat', 
                'Kalimantan Timur', 'Sulawesi Selatan', 'Bali', 'NTB', 'NTT', 'Papua', 
                'Maluku', 'Aceh', 'Riau', 'Jambi', 'Bengkulu', 'Lampung', 'DKI Jakarta',
                'Yogyakarta', 'Banten']
    
    kabupatens = ['Kabupaten Bandung', 'Kabupaten Semarang', 'Kabupaten Malang', 'Kabupaten Medan', 
                 'Kabupaten Padang', 'Kabupaten Samarinda', 'Kabupaten Makassar', 'Kabupaten Denpasar',
                 'Kabupaten Mataram', 'Kabupaten Kupang', 'Kabupaten Jayapura', 'Kabupaten Ambon',
                 'Kabupaten Banda Aceh', 'Kabupaten Pekanbaru', 'Kabupaten Jambi Kota', 'Kabupaten Bengkulu Kota',
                 'Kabupaten Bandar Lampung', 'Kabupaten Jakarta Pusat', 'Kabupaten Sleman', 'Kabupaten Tangerang']
    
    kecamatans = [f'Kecamatan {chr(65+i)}' for i in range(len(provinces))]
    desas = [f'Desa {chr(65+i)}' for i in range(len(provinces))]
    
    data = pd.DataFrame({
        'provinsi': provinces,
        'kabupaten': kabupatens,
        'kecamatan': kecamatans,
        'desa': desas,
        'gempa_bumi': np.random.poisson(3, len(provinces)),
        'banjir': np.random.poisson(8, len(provinces)),
        'tanah_longsor': np.random.poisson(4, len(provinces)),
        'kebakaran_hutan': np.random.poisson(2, len(provinces)),
        'kekeringan': np.random.poisson(3, len(provinces)),
        'tsunami': np.random.poisson(1, len(provinces)),
        'letusan_gunung': np.random.poisson(1, len(provinces)),
        'angin_puting_beliung': np.random.poisson(2, len(provinces))
    })
    return data

@st.cache_data
def load_and_clean_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                st.error("❌ Tidak dapat membaca file dengan encoding yang tersedia")
                return None
                
        except Exception as e:
            st.error(f"❌ Error membaca file: {str(e)}")
            return None
    else:
        return None

    original_shape = df.shape

    cols_to_drop = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['pkey', 'tags', 'title', 'text', 'url', 'image_url', 'id_', 'latitude', 'longitude', 'lat', 'lon']):
            cols_to_drop.append(col)
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors='ignore')
        st.info(f"ℹ️ Menghapus {len(cols_to_drop)} kolom yang tidak relevan: {', '.join(cols_to_drop[:3])}{'...' if len(cols_to_drop) > 3 else ''}")
    
    st.success(f"✅ Data berhasil dimuat: {original_shape[0]} baris → {df.shape[0]} baris, {original_shape[1]} kolom → {df.shape[1]} kolom")
    return df

def identify_columns(data):
    region_keywords = ['provinsi', 'kabupaten', 'kota', 'kecamatan', 'desa', 'kelurahan', 'wilayah', 'daerah', 'region', 'area', 'nama', 'lokasi', 'tempat']
    disaster_keywords = ['gempa', 'banjir', 'longsor', 'kebakaran', 'kekeringan', 'tsunami', 'letusan', 'angin', 'badai', 'bencana', 'puting', 'beliung', 'abrasi', 'karhutla']
    
    region_cols = []
    disaster_cols = []
    
    for col in data.columns:
        col_lower = col.lower()
        is_numeric = pd.api.types.is_numeric_dtype(data[col])

        if is_numeric and any(keyword in col_lower for keyword in disaster_keywords):
            disaster_cols.append(col)
        elif any(keyword in col_lower for keyword in region_keywords):
            region_cols.append(col)
        elif is_numeric and not any(keyword in col_lower for keyword in ['lat', 'lon', 'koordinat', 'latitude', 'longitude']):
            disaster_cols.append(col)

    if not region_cols:
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            for col in non_numeric_cols:
                unique_ratio = data[col].nunique() / len(data)
                if unique_ratio > 0.3:
                    region_cols.append(col)

        if not region_cols and non_numeric_cols:
            region_cols = [non_numeric_cols[0]]

    if not disaster_cols:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        disaster_cols = [col for col in numeric_cols if col not in region_cols]
    
    return region_cols, disaster_cols

def create_translation_dict():
    return {
        'provinsi': 'Provinsi', 'kabupaten': 'Kabupaten', 'kota': 'Kota',
        'kecamatan': 'Kecamatan', 'desa': 'Desa', 'kelurahan': 'Kelurahan',
        'wilayah': 'Wilayah', 'daerah': 'Daerah', 'area': 'Area', 'region': 'Region',
        'gempa_bumi': 'Gempa Bumi', 'banjir': 'Banjir', 'tanah_longsor': 'Tanah Longsor',
        'kebakaran_hutan': 'Kebakaran Hutan', 'kekeringan': 'Kekeringan',
        'tsunami': 'Tsunami', 'letusan_gunung': 'Letusan Gunung',
        'angin_puting_beliung': 'Angin Puting Beliung', 'abrasi': 'Abrasi',
        'karhutla': 'Kebakaran Hutan dan Lahan'
    }

def translate_column_name(col_name):
    translation_dict = create_translation_dict()
    
    if col_name in translation_dict:
        return translation_dict[col_name]

    formatted = col_name.replace('_', ' ').title()
    return formatted

@st.cache_data
def evaluate_clustering_methods(data_std, max_clusters=8):
    methods = ['ward', 'complete', 'average', 'single']
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_combinations = len(methods) * (max_clusters - 1)
    current_step = 0
    
    for method in methods:
        try:
            status_text.text(f"Mengevaluasi metode {method.title()}...")
            Z = linkage(data_std, method=method)
            
            for n in range(2, min(max_clusters + 1, len(data_std))):
                labels = fcluster(Z, t=n, criterion='maxclust')
                
                if len(np.unique(labels)) > 1:
                    sil_score = silhouette_score(data_std, labels)
                    db_score = davies_bouldin_score(data_std, labels)

                    coph_corr, _ = cophenet(Z, pdist(data_std))
                    
                    results.append({
                        'method': method,
                        'n_clusters': n,
                        'silhouette': sil_score,
                        'davies_bouldin': db_score,
                        'cophenetic': coph_corr
                    })
                
                current_step += 1
                progress_bar.progress(current_step / total_combinations)
        
        except Exception as e:
            st.warning(f"⚠️ Error pada metode {method}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results) if results else pd.DataFrame()

def generate_cluster_insights(data_with_clusters, n_clusters):
    insights = {}
    global_mean = data_with_clusters.drop('Cluster', axis=1).mean()
    global_total = global_mean.sum()

    cluster_totals = []
    for cluster_id in range(1, n_clusters + 1):
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster_id]
        cluster_profile = cluster_data.drop('Cluster', axis=1).mean()
        total_disasters = cluster_profile.sum()
        cluster_totals.append(total_disasters)

    cluster_totals_sorted = sorted(cluster_totals)

    q1 = np.percentile(cluster_totals, 25)
    q2 = np.percentile(cluster_totals, 50)
    q3 = np.percentile(cluster_totals, 75)
    
    for cluster_id in range(1, n_clusters + 1):
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster_id]
        cluster_profile = cluster_data.drop('Cluster', axis=1).mean()

        total_disasters = cluster_profile.sum()

        dominant_disaster = cluster_profile.idxmax()
        dominant_value = cluster_profile.max()

        above_avg = cluster_profile[cluster_profile > global_mean]
        below_avg = cluster_profile[cluster_profile < global_mean]

        if total_disasters >= q3:
            risk_level = "Sangat Tinggi"
            risk_color = "🔴"
        elif total_disasters >= q2:
            risk_level = "Tinggi"
            risk_color = "🟠"
        elif total_disasters >= q1:
            risk_level = "Sedang"
            risk_color = "🟡"
        else:
            risk_level = "Rendah"
            risk_color = "🟢"
        
        insights[cluster_id] = {
            'risk_level': risk_level,
            'risk_color': risk_color,
            'total_disasters': total_disasters,
            'dominant_disaster': translate_column_name(dominant_disaster),
            'dominant_value': dominant_value,
            'above_average': [translate_column_name(col) for col in above_avg.index],
            'below_average': [translate_column_name(col) for col in below_avg.index],
            'count': len(cluster_data),
            'regions': cluster_data.index.tolist()
        }
    
    return insights

def create_enhanced_visualizations(data_std, labels, method_name, n_clusters, data_agregat, selected_disaster_cols):
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))

    if data_std.shape[1] >= 1:
        if data_std.shape[1] == 1:
            fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
            
            for i, cluster_id in enumerate(np.unique(labels)):
                mask = labels == cluster_id
                y_values = np.random.normal(0, 0.1, sum(mask))
                ax_pca.scatter(data_std[mask, 0], y_values, 
                              c=[colors[i]], label=f'Cluster {cluster_id}', 
                              alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
            
            ax_pca.set_xlabel(f'Nilai Standar - {translate_column_name(selected_disaster_cols[0])}')
            ax_pca.set_ylabel('Jitter (untuk visualisasi)')
            ax_pca.set_title(f'Visualisasi 1D - {method_name.title()} Clustering')
            ax_pca.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_pca.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_pca)
            
        else:
            pca = PCA(n_components=min(2, data_std.shape[1]))
            data_pca = pca.fit_transform(data_std)
            
            fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
            
            for i, cluster_id in enumerate(np.unique(labels)):
                mask = labels == cluster_id
                ax_pca.scatter(data_pca[mask, 0], data_pca[mask, 1], 
                              c=[colors[i]], label=f'Cluster {cluster_id}', 
                              alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
            
            ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            if data_pca.shape[1] > 1:
                ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            ax_pca.set_title(f'Visualisasi PCA - {method_name.title()} Clustering')
            ax_pca.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_pca.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_pca)

    if len(data_std) > 10 and data_std.shape[1] > 1:
        perplexity = min(30, len(data_std) - 1, 5)
        if perplexity >= 5:
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                data_tsne = tsne.fit_transform(data_std)
                
                fig_tsne, ax_tsne = plt.subplots(figsize=(10, 6))
                
                for i, cluster_id in enumerate(np.unique(labels)):
                    mask = labels == cluster_id
                    ax_tsne.scatter(data_tsne[mask, 0], data_tsne[mask, 1], 
                                   c=[colors[i]], label=f'Cluster {cluster_id}', 
                                   alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
                
                ax_tsne.set_xlabel('t-SNE Dimension 1')
                ax_tsne.set_ylabel('t-SNE Dimension 2')
                ax_tsne.set_title(f'Visualisasi t-SNE - {method_name.title()} Clustering')
                ax_tsne.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax_tsne.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_tsne)
            except Exception as e:
                st.warning(f"⚠️ t-SNE tidak dapat dibuat: {str(e)}")
    elif data_std.shape[1] == 1:
        st.info("ℹ️ t-SNE memerlukan minimal 2 kolom bencana untuk visualisasi 2D")

    if data_std.shape[1] <= 3:
        st.markdown("#### Visualisasi Distribusi per Cluster")
        
        fig_dist, ax_dist = plt.subplots(figsize=(12, 6))

        cluster_data_list = []
        cluster_labels_list = []
        
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_values = data_std[mask].flatten()
            cluster_data_list.extend(cluster_values)
            cluster_labels_list.extend([f'Cluster {cluster_id}'] * len(cluster_values))

        plot_df = pd.DataFrame({
            'Values': cluster_data_list,
            'Cluster': cluster_labels_list
        })

        import seaborn as sns
        sns.boxplot(data=plot_df, x='Cluster', y='Values', ax=ax_dist)
        ax_dist.set_title('Distribusi Nilai Standar per Cluster')
        ax_dist.set_ylabel('Nilai Standar')
        ax_dist.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_dist)

st.sidebar.header("📁 Pengaturan Data")

data_source = st.sidebar.radio(
    "Pilih Sumber Data:",
    ["📊 Gunakan Data Contoh", "📤 Upload File CSV"],
    help="Pilih apakah ingin menggunakan data contoh atau upload file sendiri"
)

if data_source == "📊 Gunakan Data Contoh":
    data = load_sample_data()
    st.sidebar.markdown("""
    <div class="info-box">
        <strong>ℹ️ Info Data Contoh:</strong><br>
        • 20 provinsi di Indonesia<br>
        • Termasuk kabupaten, kecamatan, desa<br>
        • 8 jenis bencana alam<br>
        • Data simulasi untuk demonstrasi
    </div>
    """, unsafe_allow_html=True)
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload File CSV",
        type=['csv'],
        help="File harus berformat CSV dengan kolom wilayah dan data bencana"
    )
    
    if uploaded_file is not None:
        data = load_and_clean_data(uploaded_file)
        if data is None:
            st.stop()
    else:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Perhatian:</strong><br>
            Silakan upload file CSV atau gunakan data contoh untuk melanjutkan analisis.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

if data is None or data.empty:
    st.markdown("""
    <div class="error-box">
        <strong>❌ Error:</strong><br>
        Data tidak valid atau kosong. Silakan periksa file yang diupload.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

region_cols, disaster_cols = identify_columns(data)

if not region_cols or not disaster_cols:
    st.markdown("""
    <div class="error-box">
        <strong>❌ Error:</strong><br>
        Tidak dapat mengidentifikasi kolom wilayah atau kolom bencana dalam dataset.
        Pastikan dataset memiliki kolom yang sesuai.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

st.sidebar.header("🎯 Pemilihan Kolom")

st.sidebar.subheader("🗺️ Kolom Wilayah")
region_labels = [translate_column_name(col) for col in region_cols]
region_map = {translate_column_name(col): col for col in region_cols}

selected_region_label = st.sidebar.selectbox(
    "Pilih kolom wilayah:",
    region_labels,
    index=0,
    help="Kolom yang berisi nama wilayah/daerah (provinsi, kabupaten, kecamatan, desa, dll)"
)
selected_region_col = region_map[selected_region_label]

st.sidebar.subheader("📈 Kolom Bencana")
disaster_labels = [translate_column_name(col) for col in disaster_cols]
disaster_map = {translate_column_name(col): col for col in disaster_cols}

selected_disaster_labels = st.sidebar.multiselect(
    "Pilih kolom bencana:",
    disaster_labels,
    default=disaster_labels,
    help="Pilih satu atau lebih jenis bencana untuk dianalisis"
)
selected_disaster_cols = [disaster_map[label] for label in selected_disaster_labels]

if not selected_disaster_cols:
    st.markdown("""
    <div class="error-box">
        <strong>❌ Error:</strong><br>
        Pilih setidaknya satu kolom bencana untuk melanjutkan analisis.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

st.sidebar.header("⚙️ Pengaturan Clustering")

clustering_mode = st.sidebar.radio(
    "Mode Clustering:",
    ["🎯 Otomatis (Rekomendasi)", "🔧 Manual"],
    help="Pilih mode otomatis untuk mendapat rekomendasi terbaik atau manual untuk pengaturan sendiri"
)

st.subheader("📊 Ringkasan Dataset")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Baris", f"{data.shape[0]:,}")
with col2:
    st.metric("Total Kolom", f"{data.shape[1]:,}")
with col3:
    st.metric("Kolom Wilayah", f"{len(region_cols):,}")
with col4:
    st.metric("Kolom Bencana", f"{len(disaster_cols):,}")

with st.expander("👁️ Preview Data", expanded=False):
    st.dataframe(data.head(10), use_container_width=True)

    missing_values = data.isnull().sum().sum()
    if missing_values > 0:
        st.markdown(f"""
        <div class="warning-box">
            <strong>⚠️ Perhatian:</strong> Ditemukan {missing_values:,} nilai kosong dalam dataset.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-box">
            <strong>✅ Bagus:</strong> Tidak ada nilai kosong dalam dataset.
        </div>
        """, unsafe_allow_html=True)

st.subheader("🔄 Preprocessing Data")

try:
    df_clean = data.dropna(subset=[selected_region_col]).copy()
    
    data_agregat = df_clean.groupby(selected_region_col)[selected_disaster_cols].sum().fillna(0)

    data_agregat = data_agregat[(data_agregat != 0).any(axis=1)]
    
    if len(data_agregat) < 2:
        st.markdown("""
        <div class="error-box">
            <strong>❌ Error:</strong><br>
            Minimal diperlukan 2 wilayah dengan data bencana untuk melakukan clustering.
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    st.markdown(f"""
    <div class="success-box">
        <strong>✅ Preprocessing Berhasil:</strong><br>
        • {len(data_agregat):,} wilayah siap untuk dianalisis<br>
        • {len(selected_disaster_cols):,} jenis bencana dipilih
    </div>
    """, unsafe_allow_html=True)
    
except Exception as e:
    st.markdown(f"""
    <div class="error-box">
        <strong>❌ Error Preprocessing:</strong><br>
        {str(e)}
    </div>
    """, unsafe_allow_html=True)
    st.stop()

with st.expander("📋 Data Agregat per Wilayah", expanded=False):
    st.dataframe(data_agregat, use_container_width=True)

st.subheader("📈 Analisis Eksploratori")

fig_dist, ax_dist = plt.subplots(figsize=(12, 6))
total_per_disaster = data_agregat.sum().sort_values(ascending=False)

bars = ax_dist.bar(range(len(total_per_disaster)), total_per_disaster.values, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(total_per_disaster))))

for bar, value in zip(bars, total_per_disaster.values):
    ax_dist.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01, 
                f'{int(value):,}', ha='center', va='bottom', fontweight='bold')

ax_dist.set_title('Distribusi Total Kejadian Bencana per Jenis', fontsize=14, fontweight='bold')
ax_dist.set_xlabel('Jenis Bencana', fontsize=12)
ax_dist.set_ylabel('Total Kejadian', fontsize=12)
ax_dist.set_xticks(range(len(total_per_disaster)))
ax_dist.set_xticklabels([translate_column_name(col) for col in total_per_disaster.index], 
                       rotation=45, ha='right')
ax_dist.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig_dist)

if len(selected_disaster_cols) > 1:
    st.subheader("🔥 Heatmap Korelasi Antar Jenis Bencana")
    
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    correlation_matrix = data_agregat.corr()

    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdYlBu_r', center=0, ax=ax_corr,
                square=True, cbar_kws={'label': 'Koefisien Korelasi'})
    
    ax_corr.set_title('Korelasi Antar Jenis Bencana', fontsize=14, fontweight='bold')

    translated_labels = [translate_column_name(col) for col in correlation_matrix.columns]
    ax_corr.set_xticklabels(translated_labels, rotation=45, ha='right')
    ax_corr.set_yticklabels(translated_labels, rotation=0)
    
    plt.tight_layout()
    st.pyplot(fig_corr)


st.subheader("🤖 Analisis Clustering")

scaler = StandardScaler()
data_std = scaler.fit_transform(data_agregat)

if clustering_mode == "🎯 Otomatis (Rekomendasi)":
    st.markdown('<h3 style="color: white;">🔍 Evaluasi Otomatis Parameter Clustering</h3>', unsafe_allow_html=True)
    
    with st.spinner("Sedang mengevaluasi berbagai metode clustering..."):
        max_clusters = min(8, len(data_agregat) // 2)
        eval_results = evaluate_clustering_methods(data_std, max_clusters)
    
    if not eval_results.empty:
        st.markdown('<h4 style="color: white;">📊 Hasil Evaluasi</h4>', unsafe_allow_html=True)

        display_results = eval_results.copy()
        display_results['method'] = display_results['method'].str.title()
        display_results = display_results.round(4)
        
        st.dataframe(display_results, use_container_width=True)

        best_config = eval_results.loc[eval_results['silhouette'].idxmax()]
        
        st.markdown(f"""
        <div class="success-box">
            <strong>🏆 Konfigurasi Terbaik:</strong><br>
            • <strong>Metode:</strong> {best_config['method'].title()}<br>
            • <strong>Jumlah Cluster:</strong> {int(best_config['n_clusters'])}<br>
            • <strong>Silhouette Score:</strong> {best_config['silhouette']:.4f}<br>
            • <strong>Davies-Bouldin Score:</strong> {best_config['davies_bouldin']:.4f}
        </div>
        """, unsafe_allow_html=True)

        best_method = best_config['method']
        best_n_clusters = int(best_config['n_clusters'])

        col1, col2 = st.columns(2)
        
        with col1:
            fig_eval1, ax_eval1 = plt.subplots(figsize=(8, 5))
            for method in eval_results['method'].unique():
                subset = eval_results[eval_results['method'] == method]
                ax_eval1.plot(subset['n_clusters'], subset['silhouette'], 'o-', 
                             label=method.title(), linewidth=2, markersize=6)
            
            ax_eval1.set_xlabel('Jumlah Cluster')
            ax_eval1.set_ylabel('Silhouette Score')
            ax_eval1.set_title('Evaluasi Silhouette Score')
            ax_eval1.legend()
            ax_eval1.grid(True, alpha=0.3)
            st.pyplot(fig_eval1)
        
        with col2:
            fig_eval2, ax_eval2 = plt.subplots(figsize=(8, 5))
            for method in eval_results['method'].unique():
                subset = eval_results[eval_results['method'] == method]
                ax_eval2.plot(subset['n_clusters'], subset['davies_bouldin'], 'o-', 
                             label=method.title(), linewidth=2, markersize=6)
            
            ax_eval2.set_xlabel('Jumlah Cluster')
            ax_eval2.set_ylabel('Davies-Bouldin Index')
            ax_eval2.set_title('Evaluasi Davies-Bouldin Index')
            ax_eval2.legend()
            ax_eval2.grid(True, alpha=0.3)
            st.pyplot(fig_eval2)
    
    else:
        st.markdown("""
        <div class="error-box">
            <strong>❌ Error:</strong><br>
            Tidak dapat melakukan evaluasi clustering. Silakan gunakan mode manual.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

else:
    st.markdown("### 🔧 Pengaturan Manual Clustering")

    linkage_methods = {
        'ward': 'Ward (Minimizes variance)',
        'complete': 'Complete (Maximum distance)',
        'average': 'Average (Average distance)',
        'single': 'Single (Minimum distance)'
    }
    
    best_method = st.sidebar.selectbox(
        "Metode Linkage:",
        list(linkage_methods.keys()),
        format_func=lambda x: linkage_methods[x],
        index=0
    )

    max_clusters = min(8, len(data_agregat) // 2)
    best_n_clusters = st.sidebar.slider(
        "Jumlah Cluster:",
        min_value=2,
        max_value=max(2, max_clusters),
        value=min(4, max(2, max_clusters)),
        help=f"Maksimal {max_clusters} cluster berdasarkan ukuran data"
    )

try:
    Z = linkage(data_std, method=best_method)
    labels = fcluster(Z, t=best_n_clusters, criterion='maxclust')

    sil_score = silhouette_score(data_std, labels)
    db_score = davies_bouldin_score(data_std, labels)

    st.sidebar.markdown('<h3 style="color: white;">📊 Metrik Evaluasi</h3>', unsafe_allow_html=True)
    st.sidebar.metric("Silhouette Score", f"{sil_score:.4f}")
    st.sidebar.metric("Davies-Bouldin Index", f"{db_score:.4f}")

    if sil_score > 0.7:
        quality = "Sangat Baik"
        quality_color = "🟢"
    elif sil_score > 0.5:
        quality = "Baik"
        quality_color = "🟡"
    elif sil_score > 0.25:
        quality = "Sedang"
        quality_color = "🟠"
    else:
        quality = "Kurang"
        quality_color = "🔴"
    
    st.sidebar.markdown(f"""
    <div style="color: white;">
    <strong>Kualitas Clustering:</strong> {quality_color} {quality}
    </div>
    """, unsafe_allow_html=True)
    
except Exception as e:
    st.markdown(f"""
    <div style="color: white; background-color: #d32f2f; padding: 15px; border-radius: 5px; border-left: 4px solid #b71c1c;">
        <strong>❌ Error Clustering:</strong><br>
        {str(e)}
    </div>
    """, unsafe_allow_html=True)
    st.stop()

st.subheader("🎯 Hasil Clustering")

cluster_counts = pd.Series(labels).value_counts().sort_index()
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Cluster", f"{best_n_clusters}")
with col2:
    st.metric("Silhouette Score", f"{sil_score:.4f}")
with col3:
    st.metric("Davies-Bouldin Index", f"{db_score:.4f}")

st.markdown('<h4 style="color: white;">📊 Distribusi Wilayah per Cluster</h4>', unsafe_allow_html=True)
for i, count in cluster_counts.items():
    st.markdown(f'<p style="color: white;"><strong>Cluster {i}:</strong> {count} wilayah</p>', unsafe_allow_html=True)

st.subheader("📊 Visualisasi Hasil Clustering")

tab1, tab2, tab3 = st.tabs(["🌐 Proyeksi Dimensi", "📈 Analisis Silhouette", "🌳 Dendrogram"])

with tab1:
    st.markdown('<div style="color: #333333;"><h4>Visualisasi Proyeksi Dimensi</h4></div>', unsafe_allow_html=True)

    create_enhanced_visualizations(data_std, labels, best_method, best_n_clusters, data_agregat, selected_disaster_cols)

with tab2:
    st.markdown('<div style="color: #333333;"><h4>Analisis Silhouette</h4></div>', unsafe_allow_html=True)

    colors = plt.cm.Set3(np.linspace(0, 1, best_n_clusters))

    sample_silhouette_values = silhouette_samples(data_std, labels)
    
    fig_sil, ax_sil = plt.subplots(figsize=(10, 6))
    y_lower = 10
    
    for i, cluster_id in enumerate(np.unique(labels)):
        cluster_silhouette_values = sample_silhouette_values[labels == cluster_id]
        cluster_silhouette_values.sort()
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        ax_sil.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values,
                            facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
        
        ax_sil.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster_id))
        y_lower = y_upper + 10
    
    avg_score = silhouette_score(data_std, labels)
    ax_sil.axvline(x=avg_score, color="red", linestyle="--", linewidth=2,
                   label=f'Rata-rata: {avg_score:.3f}')
    
    ax_sil.set_xlabel('Nilai Silhouette Coefficient')
    ax_sil.set_ylabel('Label Cluster')
    ax_sil.set_title('Analisis Silhouette untuk Setiap Cluster')
    ax_sil.legend()
    ax_sil.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_sil)

    st.markdown('<div style="color: #333333;"><h4>Heatmap Karakteristik Cluster</h4></div>', unsafe_allow_html=True)
    data_with_clusters = data_agregat.copy()
    data_with_clusters['Cluster'] = labels
    cluster_profiles = data_with_clusters.groupby('Cluster').mean()
    
    fig_heat, ax_heat = plt.subplots(figsize=(12, 8))

    heatmap_data = cluster_profiles.T
    
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                ax=ax_heat, cbar_kws={'label': 'Rata-rata Kejadian'})
    
    ax_heat.set_title('Karakteristik Setiap Cluster', fontsize=14, fontweight='bold')
    ax_heat.set_xlabel('Cluster', fontsize=12)
    ax_heat.set_ylabel('Jenis Bencana', fontsize=12)

    ax_heat.set_yticklabels([translate_column_name(col) for col in heatmap_data.index], rotation=0)
    
    plt.tight_layout()
    st.pyplot(fig_heat)

with tab3:
    st.markdown('<div style="color: #333333;"><h4>Dendrogram Hierarchical Clustering</h4></div>', unsafe_allow_html=True)
    
    fig_dend, ax_dend = plt.subplots(figsize=(15, 8))
    
    if len(data_agregat) <= 30:
        dendrogram(Z, labels=data_agregat.index.tolist(), leaf_rotation=90, 
                  leaf_font_size=8, ax=ax_dend)
    else:
        dendrogram(Z, leaf_rotation=90, ax=ax_dend)
    
    ax_dend.set_title(f'Dendrogram - {best_method.title()} Linkage', fontsize=14, fontweight='bold')
    ax_dend.set_xlabel('Wilayah' if len(data_agregat) <= 30 else 'Index Wilayah')
    ax_dend.set_ylabel('Jarak')
    ax_dend.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig_dend)


st.markdown('<h3 style="color: white;">🧠 Analisis dan Interpretasi Cluster</h3>', unsafe_allow_html=True)

data_with_clusters = data_agregat.copy()
data_with_clusters['Cluster'] = labels
insights = generate_cluster_insights(data_with_clusters, best_n_clusters)

for cluster_id in range(1, best_n_clusters + 1):
    insight = insights[cluster_id]
    
    with st.expander(f"{insight['risk_color']} Cluster {cluster_id} - Risiko {insight['risk_level']}", expanded=(cluster_id == 1)):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div style="color: white;">
            <strong>📊 Karakteristik Utama:</strong>
            - <strong>Tingkat Risiko:</strong> {insight['risk_color']} {insight['risk_level']}
            - <strong>Jumlah Wilayah:</strong> {insight['count']} wilayah
            - <strong>Total Rata-rata Kejadian:</strong> {insight['total_disasters']:.1f}
            - <strong>Bencana Dominan:</strong> {insight['dominant_disaster']} ({insight['dominant_value']:.1f})
            
            <strong>⬆️ Di Atas Rata-rata Global:</strong>
            {', '.join(insight['above_average']) if insight['above_average'] else 'Tidak ada'}
            
            <strong>⬇️ Di Bawah Rata-rata Global:</strong>
            {', '.join(insight['below_average']) if insight['below_average'] else 'Tidak ada'}
            
            <strong>🗺️ Wilayah dalam Cluster:</strong>
            </div>
            """, unsafe_allow_html=True)

            regions_to_show = insight['regions'][:10]  
            for region in regions_to_show:
                st.markdown(f'<p style="color: white;">• {region}</p>', unsafe_allow_html=True)
            
            if len(insight['regions']) > 10:
                st.markdown(f'<p style="color: white;">• ... dan {len(insight["regions"]) - 10} wilayah lainnya</p>', unsafe_allow_html=True)
        
        with col2:
            cluster_profile = data_with_clusters[data_with_clusters['Cluster'] == cluster_id].drop('Cluster', axis=1).mean()
            
            fig_profile, ax_profile = plt.subplots(figsize=(8, 6))

            if insight['risk_level'] == "Sangat Tinggi":
                color = 'darkred'
            elif insight['risk_level'] == "Tinggi":
                color = 'red'
            elif insight['risk_level'] == "Sedang":
                color = 'orange'
            else:
                color = 'green'
            
            bars = ax_profile.bar(range(len(cluster_profile)), cluster_profile.values, color=color, alpha=0.7)
            
            for bar, value in zip(bars, cluster_profile.values):
                ax_profile.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01, 
                               f'{value:.1f}', ha='center', va='bottom', fontsize=8)
            
            ax_profile.set_title(f'Profil Cluster {cluster_id}', fontweight='bold')
            ax_profile.set_ylabel('Rata-rata Kejadian')
            ax_profile.set_xticks(range(len(cluster_profile)))
            ax_profile.set_xticklabels([translate_column_name(col) for col in cluster_profile.index], 
                                     rotation=45, ha='right', fontsize=8)
            ax_profile.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_profile)

st.subheader("📋 Ringkasan Eksekutif")

high_risk_clusters = [k for k, v in insights.items() if v['risk_level'] in ['Sangat Tinggi', 'Tinggi']]
medium_risk_clusters = [k for k, v in insights.items() if v['risk_level'] == 'Sedang']
low_risk_clusters = [k for k, v in insights.items() if v['risk_level'] == 'Rendah']

total_high_risk_regions = sum([insights[k]['count'] for k in high_risk_clusters])
total_medium_risk_regions = sum([insights[k]['count'] for k in medium_risk_clusters])
total_low_risk_regions = sum([insights[k]['count'] for k in low_risk_clusters])

all_disasters = []
for cluster_id, insight in insights.items():
    all_disasters.extend(insight['above_average'])

most_common_disasters = pd.Series(all_disasters).value_counts().head(3) if all_disasters else pd.Series()

st.markdown(f"""
<div style="color: white;">

### 🎯 Temuan Utama

<strong>📊 Distribusi Risiko:</strong>
- 🔴 <strong>Risiko Tinggi:</strong> {len(high_risk_clusters)} cluster ({total_high_risk_regions} wilayah)
- 🟡 <strong>Risiko Sedang:</strong> {len(medium_risk_clusters)} cluster ({total_medium_risk_regions} wilayah)  
- 🟢 <strong>Risiko Rendah:</strong> {len(low_risk_clusters)} cluster ({total_low_risk_regions} wilayah)

<strong>🌪️ Bencana Paling Umum:</strong>

</div>
""", unsafe_allow_html=True)

if not most_common_disasters.empty:
    for i, (disaster, count) in enumerate(most_common_disasters.items(), 1):
        st.markdown(f'<p style="color: white;">{i}. <strong>{disaster}</strong> (muncul di {count} cluster)</p>', unsafe_allow_html=True)
else:
    st.markdown('<p style="color: white;">Tidak ada bencana yang dominan di atas rata-rata global.</p>', unsafe_allow_html=True)

st.markdown(f"""
<div style="color: white;">
<h3>🔬 Kualitas Analisis:</h3>
- <strong>Metode Clustering:</strong> {best_method.title()} Linkage
- <strong>Jumlah Cluster Optimal:</strong> {best_n_clusters}
- <strong>Silhouette Score:</strong> {sil_score:.4f} ({quality})
- <strong>Davies-Bouldin Index:</strong> {db_score:.4f} (semakin rendah semakin baik)

<h3>💡 Rekomendasi</h3>

<h4>🎯 Strategi Mitigasi:</h4>
1. <strong>Prioritas Tinggi:</strong> Fokus pada {total_high_risk_regions} wilayah berisiko tinggi
2. <strong>Alokasi Sumber Daya:</strong> Distribusikan berdasarkan profil risiko setiap cluster
3. <strong>Sistem Peringatan Dini:</strong> Implementasi khusus untuk bencana dominan di setiap cluster
4. <strong>Infrastruktur:</strong> Pengembangan infrastruktur tahan bencana sesuai karakteristik wilayah

<h4>📈 Monitoring dan Evaluasi:</h4>
- Pantau perkembangan kejadian bencana secara berkala
- Update clustering setiap tahun dengan data terbaru
- Evaluasi efektivitas strategi mitigasi berdasarkan cluster
</div>
""", unsafe_allow_html=True)

st.subheader("💾 Export Hasil Analisis")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Download Data Clustering", type="primary"):
        export_data = data_with_clusters.copy()
        export_data.index.name = selected_region_col

        cluster_info = []
        for idx, row in export_data.iterrows():
            cluster_id = row['Cluster']
            insight = insights[cluster_id]
            cluster_info.append({
                'Wilayah': idx,
                'Cluster': cluster_id,
                'Tingkat_Risiko': insight['risk_level'],
                'Total_Bencana': row.drop('Cluster').sum(),
                'Bencana_Dominan': insight['dominant_disaster']
            })
        
        cluster_info_df = pd.DataFrame(cluster_info)

        csv_buffer = io.StringIO()
        cluster_info_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"hasil_clustering_bencana_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📋 Download Ringkasan Insights"):
        insights_summary = []
        for cluster_id, insight in insights.items():
            insights_summary.append({
                'Cluster': cluster_id,
                'Tingkat_Risiko': insight['risk_level'],
                'Jumlah_Wilayah': insight['count'],
                'Total_Rata_rata_Bencana': round(insight['total_disasters'], 2),
                'Bencana_Dominan': insight['dominant_disaster'],
                'Bencana_Di_Atas_Rata_rata': '; '.join(insight['above_average']),
                'Contoh_Wilayah': '; '.join(insight['regions'][:5])
            })
        
        insights_df = pd.DataFrame(insights_summary)

        csv_buffer = io.StringIO()
        insights_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Insights CSV",
            data=csv_data,
            file_name=f"insights_clustering_bencana_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 2rem;">
    <p><strong>🌋 Analisis Clustering Bencana Alam Indonesia</strong></p>
    <p>Sistem analisis berbasis Machine Learning untuk pengelompokan wilayah rawan bencana</p>
    <p><em>Dikembangkan untuk mendukung strategi mitigasi bencana yang lebih efektif</em></p>
    <p>Kelompok 4 </p>
</div>
""", unsafe_allow_html=True)

