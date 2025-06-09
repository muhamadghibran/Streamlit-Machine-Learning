import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

st.set_page_config(
    page_title="Analisis Clustering Bencana Alam", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌋 Analisis Clustering Wilayah Rawan Bencana Alam")
st.markdown("""
Aplikasi ini melakukan **Hierarchical Clustering** untuk mengelompokkan wilayah berdasarkan 
tingkat kerawanan bencana alam di Indonesia menggunakan algoritma machine learning.
""")

@st.cache_data
def load_and_clean_data(uploaded_file=None, sample_data=False):
    if sample_data:
        np.random.seed(42)
        data = pd.DataFrame({
            'provinsi': [f'Provinsi {chr(65+i)}' for i in range(20)],
            'kabupaten': [f'Kabupaten {chr(65+i)}' for i in range(20)],
            'kota': [f'Kota {chr(65+i)}' for i in range(20)],
            'kecamatan': [f'Kecamatan {chr(65+i)}' for i in range(20)],
            'desa': [f'Desa {chr(65+i)}' for i in range(20)],
            'gempa_bumi': np.random.poisson(3, 20),
            'banjir': np.random.poisson(5, 20),
            'tanah_longsor': np.random.poisson(2, 20),
            'kebakaran_hutan': np.random.poisson(1, 20),
            'kekeringan': np.random.poisson(2, 20),
            'tsunami': np.random.poisson(0.5, 20),
            'letusan_gunung': np.random.poisson(0.3, 20)
        })
        return data
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error membaca file: {str(e)}")
            return None
    else:
        return None
    kata_kunci_buang = ['pkey', 'tags', 'title', 'text', 'long', 'lat', 
                        'latitude', 'longitude', 'url', 'image_url', 'id']
    kata_kunci_prefix = ['ID_', 'id_']
    kol_buang = []
    for col in df.columns:
        col_lower = col.lower()
        if (col_lower in kata_kunci_buang or any(col.startswith(pref) for pref in kata_kunci_prefix)):
            kol_buang.append(col)
    if kol_buang:
        df = df.drop(columns=kol_buang, errors='ignore')
    return df

@st.cache_data
def evaluate_clustering(data_std, max_clusters=8):
    methods = ['ward', 'complete', 'average', 'single']
    results = []
    for method in methods:
        try:
            Z = linkage(data_std, method=method)
            for n in range(2, min(max_clusters+1, len(data_std)//2)):
                labels = fcluster(Z, t=n, criterion='maxclust')
                if len(np.unique(labels)) > 1:
                    sil = silhouette_score(data_std, labels)
                    db = davies_bouldin_score(data_std, labels)
                    results.append({'method': method, 'n_clusters': n, 'silhouette': sil, 'davies_bouldin': db})
        except: continue
    return pd.DataFrame(results) if results else pd.DataFrame()

st.sidebar.header("📁 Sumber Data")
use_sample = st.sidebar.checkbox("Gunakan Data Contoh", value=True)
if use_sample:
    data = load_and_clean_data(sample_data=True)
    st.info("🔄 Menggunakan data contoh untuk demonstrasi")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload file CSV", type=['csv'], help="File harus berformat CSV dengan kolom wilayah dan data bencana"
    )
    if uploaded_file is not None:
        data = load_and_clean_data(uploaded_file)
        if data is None:
            st.stop()
    else:
        st.warning("⚠️ Silakan upload file CSV atau gunakan data contoh")
        st.stop()

afDataEmpty = data is None or data.empty
if afDataEmpty:
    st.error("❌ Data tidak valid atau kosong")
    st.stop()

st.subheader("📊 Data Setelah Pembersihan")
st.write(f"**Jumlah baris:** {data.shape[0]} | **Jumlah kolom:** {data.shape[1]}")
col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(data.head(10), use_container_width=True)
with col2:
    st.write("**Info Dataset:**")
    missing_vals = data.isnull().sum().sum()
    if missing_vals > 0:
        st.warning(f"⚠️ Ditemukan {missing_vals} nilai kosong")
    else:
        st.success("✅ Tidak ada nilai kosong")

st.subheader("📈 Heatmap Korelasi Antar Jenis Bencana")
fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
sns.heatmap(data.select_dtypes(include=[np.number]).corr(), annot=True, fmt='.2f', ax=ax_corr)
ax_corr.set_title('Heatmap Korelasi Bencana')
plt.tight_layout()
st.pyplot(fig_corr)

translation_dict = {
    'date': 'Tanggal','time': 'Waktu','source': 'Sumber','status': 'Status',
    'url': 'URL','image_url': 'URL Gambar','disaster_type': 'Jenis Bencana',
    'report_date': 'Tanggal Pelaporan','provinsi': 'Provinsi','kabupaten': 'Kabupaten',
    'kota': 'Kota','kecamatan': 'Kecamatan','desa': 'Desa','kelurahan': 'Kelurahan',
    'wilayah': 'Wilayah','daerah': 'Daerah','area': 'Area','region': 'Region',
    'gempa_bumi': 'Gempa Bumi','banjir': 'Banjir','tanah_longsor': 'Tanah Longsor',
    'kebakaran_hutan': 'Kebakaran Hutan','kekeringan': 'Kekeringan',
    'tsunami': 'Tsunami','letusan_gunung': 'Letusan Gunung'
}
def kolom_label(col):
    if col in translation_dict:
        return translation_dict[col]
    parts = col.replace('_', ' ').split()
    return ' '.join(p.capitalize() for p in parts)

wilayer_kw = ['provinsi','kabupaten','kota','kecamatan','desa','kelurahan','daerah','wilayah','region','area','nama']
bencana_kw = ['gempa','banjir','longsor','kebakaran','kekeringan','tsunami','letusan','angin','badai','bencana']

kolom_wilayah, kolom_num = [], []
for col in data.columns:
    low = col.lower()
    is_num = pd.api.types.is_numeric_dtype(data[col])
    is_benc = any(k in low for k in bencana_kw)
    is_wil = any(k in low for k in wilayer_kw)
    if is_num and (not is_wil or is_benc): kolom_num.append(col)
    elif not is_num and not is_benc:
        uniq = data[col].nunique()/len(data)
        if uniq>0.5 or is_wil: kolom_wilayah.append(col)
if not kolom_wilayah:
    non_num = data.select_dtypes(exclude=[np.number]).columns.tolist()
    kolom_wilayah = non_num[:1] if non_num else [data.columns[0]]
if not kolom_num:
    kolom_num = [c for c in data.select_dtypes(include=[np.number]).columns if c not in kolom_wilayah]

labels_wil = [kolom_label(c) for c in kolom_wilayah]
map_wil = {kolom_label(c):c for c in kolom_wilayah}
col_wil = st.sidebar.selectbox("🗺️ Kolom Wilayah", labels_wil, index=0)
col_wil = map_wil[col_wil]

lbls_ben = [kolom_label(c) for c in kolom_num]
map_num = {kolom_label(c):c for c in kolom_num}
col_ben = st.sidebar.multiselect("📈 Kolom Bencana", lbls_ben, default=lbls_ben)
col_ben = [map_num[l] for l in col_ben]
if not col_ben:
    st.error("❌ Pilih setidaknya satu kolom bencana")
    st.stop()

df_clean = data.dropna(subset=[col_wil]).copy()
data_agregat = df_clean.groupby(col_wil)[col_ben].sum().replace(np.nan,0)
nonzero = data_agregat[(data_agregat!=0).any(axis=1)]
if len(nonzero)>0: data_agregat=nonzero
if data_agregat.shape[0]<2:
    st.error("❌ Minimal 2 wilayah untuk clustering")
    st.stop()

st.subheader("📋 Data Agregat per Wilayah")
st.write(f"**Wilayah:** {data_agregat.shape[0]} | **Jenis bencana:** {data_agregat.shape[1]}")
st.dataframe(data_agregat.head(10), use_container_width=True)

st.subheader("📊 Distribusi Total Kejadian")
total = data_agregat.sum().sort_values(ascending=False)
fig1, ax1 = plt.subplots(figsize=(10,6))
bars=ax1.bar([kolom_label(c) for c in total.index], total.values)
for bar,v in zip(bars,total.values):
    ax1.text(bar.get_x()+bar.get_width()/2, v, f"{int(v)}", ha='center')
ax1.set_xticklabels([kolom_label(c) for c in total.index], rotation=45)
st.pyplot(fig1)

scaler=StandardScaler()
data_std=scaler.fit_transform(data_agregat)

st.subheader("🤖 Evaluasi Otomatis Parameter")
eval_results = evaluate_clustering(data_std)
if not eval_results.empty:
    best = eval_results.loc[eval_results['silhouette'].idxmax()]
    st.success(f"✅ Rekomendasi: {best['method'].title()} dengan {int(best['n_clusters'])} cluster (Silhouette: {best['silhouette']:.3f})")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_eval1, ax_eval1 = plt.subplots(figsize=(8,5))
        for method in eval_results['method'].unique():
            subset = eval_results[eval_results['method']==method]
            ax_eval1.plot(subset['n_clusters'], subset['silhouette'], 'o-', label=method.title())
        ax_eval1.set_xlabel('Jumlah Cluster'); ax_eval1.set_ylabel('Silhouette Score')
        ax_eval1.legend(); ax_eval1.set_title('Evaluasi Silhouette Score')
        st.pyplot(fig_eval1)
    
    with col2:
        fig_eval2, ax_eval2 = plt.subplots(figsize=(8,5))
        for method in eval_results['method'].unique():
            subset = eval_results[eval_results['method']==method]
            ax_eval2.plot(subset['n_clusters'], subset['davies_bouldin'], 'o-', label=method.title())
        ax_eval2.set_xlabel('Jumlah Cluster'); ax_eval2.set_ylabel('Davies-Bouldin Index')
        ax_eval2.legend(); ax_eval2.set_title('Evaluasi Davies-Bouldin Index')
        st.pyplot(fig_eval2)

st.sidebar.header("⚙️ Pengaturan Manual")
use_auto = st.sidebar.checkbox("Gunakan Rekomendasi Otomatis", value=True if not eval_results.empty else False)
if use_auto and not eval_results.empty:
    metode, n_clusters = best['method'], int(best['n_clusters'])
else:
    methods={'ward':'Ward','complete':'Complete','average':'Average','single':'Single'}
    metode=st.sidebar.selectbox("Linkage Method", list(methods), format_func=lambda x:methods[x])
    maxc=min(8,len(data_agregat)//2)
    n_clusters=st.sidebar.slider("Jumlah Cluster",2,max(2,maxc),min(4,max(2,maxc)))

Z=linkage(data_std, method=metode)
labels=fcluster(Z, t=n_clusters, criterion='maxclust')
data_hasil=data_agregat.copy()
data_hasil['Cluster']=labels

sil=silhouette_score(data_std,labels)
db=davies_bouldin_score(data_std,labels)
st.sidebar.header("📊 Validasi Clustering")
st.sidebar.metric("Silhouette Score", f"{sil:.3f}")
st.sidebar.metric("Davies-Bouldin Index", f"{db:.3f}")

st.subheader("🎯 Hasil Clustering")
counts=data_hasil['Cluster'].value_counts().sort_index()
for i,c in counts.items(): st.write(f"Cluster {i}: {c} wilayah")

st.subheader("📊 Visualisasi Clustering")
tab1, tab2, tab3, tab4 = st.tabs(["PCA Plot", "t-SNE Plot", "Silhouette Plot", "Heatmap Cluster"])

with tab1:
    pca=PCA(2)
    proj=pca.fit_transform(data_std)
    fig_pca,ax_pca=plt.subplots(figsize=(10,6))
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    for i, cid in enumerate(np.unique(labels)):
        ax_pca.scatter(proj[labels==cid,0],proj[labels==cid,1],label=f"Cluster {cid}", c=[colors[i]])
    ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax_pca.legend(); ax_pca.set_title('PCA Clustering Visualization')
    st.pyplot(fig_pca)

with tab2:
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data_std)-1))
    proj_tsne = tsne.fit_transform(data_std)
    fig_tsne, ax_tsne = plt.subplots(figsize=(10,6))
    for i, cid in enumerate(np.unique(labels)):
        ax_tsne.scatter(proj_tsne[labels==cid,0], proj_tsne[labels==cid,1], label=f"Cluster {cid}", c=[colors[i]])
    ax_tsne.set_xlabel('t-SNE 1'); ax_tsne.set_ylabel('t-SNE 2')
    ax_tsne.legend(); ax_tsne.set_title('t-SNE Clustering Visualization')
    st.pyplot(fig_tsne)

with tab3:
    sample_silhouette_values = silhouette_samples(data_std, labels)
    fig_sil, ax_sil = plt.subplots(figsize=(10,6))
    y_lower = 10
    for i in range(n_clusters):
        cluster_silhouette_values = sample_silhouette_values[labels == i+1]
        cluster_silhouette_values.sort()
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = colors[i]
        ax_sil.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values, 
                            facecolor=color, edgecolor=color, alpha=0.7)
        ax_sil.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1))
        y_lower = y_upper + 10
    ax_sil.axvline(x=sil, color="red", linestyle="--", label=f'Average Score: {sil:.3f}')
    ax_sil.set_xlabel('Silhouette Coefficient Values')
    ax_sil.set_ylabel('Cluster Label')
    ax_sil.set_title('Silhouette Analysis'); ax_sil.legend()
    st.pyplot(fig_sil)

with tab4:
    cluster_profiles = data_hasil.groupby('Cluster').mean()
    fig_heat, ax_heat = plt.subplots(figsize=(10,6))
    sns.heatmap(cluster_profiles.T, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax_heat)
    ax_heat.set_title('Heatmap Karakteristik Cluster')
    ax_heat.set_ylabel('Jenis Bencana')
    st.pyplot(fig_heat)

st.subheader("🌳 Dendrogram")
fig2,ax2=plt.subplots(figsize=(12,6))
dendrogram(Z, labels=[str(i) for i in data_agregat.index], leaf_rotation=90, ax=ax2)
st.pyplot(fig2)

st.subheader("🧠 Interpretasi Dinamis Cluster")

def generate_interpretation(cluster_data):
    interpretations = {}
    global_mean = data_agregat.mean()
    
    for cid in range(1, n_clusters + 1):
        cluster_profile = cluster_data[cluster_data['Cluster'] == cid].drop('Cluster', axis=1).mean()
        total_disasters = cluster_profile.sum()
        dominant_disaster = cluster_profile.idxmax()

        above_avg = cluster_profile[cluster_profile > global_mean]
        below_avg = cluster_profile[cluster_profile < global_mean]
        
        risk_level = "Tinggi" if total_disasters > global_mean.sum() else ("Sedang" if total_disasters > global_mean.sum()*0.5 else "Rendah")
        
        interpretation = {
            'risk_level': risk_level,
            'total_disasters': total_disasters,
            'dominant_disaster': kolom_label(dominant_disaster),
            'above_average': [kolom_label(col) for col in above_avg.index],
            'below_average': [kolom_label(col) for col in below_avg.index],
            'count': len(cluster_data[cluster_data['Cluster'] == cid])
        }
        interpretations[cid] = interpretation
    
    return interpretations

interpretations = generate_interpretation(data_hasil)

for cid in range(1, n_clusters + 1):
    interp = interpretations[cid]
    with st.expander(f"🎯 Cluster {cid} - Risiko {interp['risk_level']}", expanded=(cid == 1)):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            **📊 Karakteristik Utama:**
            - **Tingkat Risiko:** {interp['risk_level']}
            - **Jumlah Wilayah:** {interp['count']} wilayah
            - **Total Kejadian:** {interp['total_disasters']:.1f} rata-rata
            - **Bencana Dominan:** {interp['dominant_disaster']}
            
            **⬆️ Di Atas Rata-rata:** {', '.join(interp['above_average'])}
            
            **⬇️ Di Bawah Rata-rata:** {', '.join(interp['below_average'])}
            """)
        
        with col2:
            prof = data_hasil[data_hasil['Cluster'] == cid].drop('Cluster', axis=1).mean()
            fig_prof, ax_prof = plt.subplots(figsize=(8, 5))
            bars = ax_prof.bar([kolom_label(c) for c in prof.index], prof.values, 
                              color='lightcoral' if interp['risk_level']=='Tinggi' else 
                                   'orange' if interp['risk_level']=='Sedang' else 'lightgreen')
            ax_prof.set_title(f'Profil Cluster {cid} - Risiko {interp["risk_level"]}')
            ax_prof.set_ylabel('Rata-rata Kejadian')
            ax_prof.set_xticklabels([kolom_label(c) for c in prof.index], rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig_prof)

st.subheader("📋 Ringkasan Eksekutif")
high_risk = [k for k,v in interpretations.items() if v['risk_level']=='Tinggi']
med_risk = [k for k,v in interpretations.items() if v['risk_level']=='Sedang']
low_risk = [k for k,v in interpretations.items() if v['risk_level']=='Rendah']

st.markdown(f"""
**🎯 Hasil Analisis Clustering ({metode.title()} - {n_clusters} Cluster):**

- **Kualitas Clustering:** Silhouette Score {sil:.3f} {'(Baik)' if sil>0.5 else '(Cukup)' if sil>0.3 else '(Perlu Perbaikan)'}
- **Cluster Risiko Tinggi:** {len(high_risk)} cluster ({', '.join([f'C{c}' for c in high_risk])})
- **Cluster Risiko Sedang:** {len(med_risk)} cluster ({', '.join([f'C{c}' for c in med_risk])})
- **Cluster Risiko Rendah:** {len(low_risk)} cluster ({', '.join([f'C{c}' for c in low_risk])})

**🚨 Rekomendasi Prioritas:**
1. Fokus mitigasi pada cluster risiko tinggi
2. Penguatan sistem peringatan dini
3. Alokasi sumber daya berdasarkan profil cluster
""")

# --- Download ---
st.download_button("📥 Download Hasil CSV", data_hasil.to_csv(), file_name="hasil_clustering_enhanced.csv")

st.success("✅ Analisis Clustering Selesai!")