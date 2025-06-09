import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Prediksi Diabetes", page_icon="🩺", layout="wide")

@st.cache_data
def load_raw_data():
    """Load data mentah tanpa preprocessing untuk perbandingan"""
    df_raw = pd.read_csv("Data Set Diabetes.csv", delimiter=";")
    df_raw.columns = df_raw.columns.str.strip()
    # Ganti 0 dengan NaN untuk kolom tertentu untuk visualisasi
    df_raw_viz = df_raw.copy()
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df_raw_viz[col] = df_raw_viz[col].replace(0, np.nan)
    return df_raw, df_raw_viz

@st.cache_data
def load_data():
    """Load data yang sudah dibersihkan"""
    df = pd.read_csv("Data Set Diabetes.csv", delimiter=";")
    df.columns = df.columns.str.strip()
    
    # Ganti 0 dengan NaN untuk kolom tertentu
    for col in ['Glucose', 'BloodPressure', 'BMI']:
        df[col] = df[col].replace(0, np.nan)
    
    # Isi NaN dengan median kolom
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # Hapus kolom Insulin dan SkinThickness
    df = df.drop(['Insulin', 'SkinThickness'], axis=1)
    
    return df

@st.cache_resource
def load_model():
    return joblib.load("naive_bayes_diabetes_model.pkl")

def plot_roc_curve(model, df):
    X = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"]]
    y = df["Outcome"]

    y_prob = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    return fig

def show_missing_value_comparison(df_raw_viz, df_clean):
    """Menampilkan perbandingan missing values sebelum dan sesudah preprocessing"""
    
    st.subheader("📊 Perbandingan Missing Values: Sebelum vs Sesudah Preprocessing")
    
    # Buat kolom untuk menampilkan side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔴 **Data Mentah (Sebelum Dibersihkan)**")
        
        # Hitung persentase missing values
        missing_before = df_raw_viz.isnull().sum()
        missing_pct_before = (missing_before / len(df_raw_viz)) * 100
        
        # Tampilkan tabel missing values
        missing_df_before = pd.DataFrame({
            'Kolom': missing_before.index,
            'Missing Count': missing_before.values,
            'Missing %': missing_pct_before.values
        }).round(2)
        missing_df_before = missing_df_before[missing_df_before['Missing Count'] > 0]
        
        if not missing_df_before.empty:
            st.dataframe(missing_df_before)
        else:
            st.info("Tidak ada missing values terdeteksi")
        
        # Visualisasi missing values - sebelum
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        msno.bar(df_raw_viz, ax=ax1, color='red')
        ax1.set_title('Missing Values - Data Mentah')
        st.pyplot(fig1)
    
    with col2:
        st.markdown("#### 🟢 **Data Bersih (Sesudah Dibersihkan)**")
        
        # Hitung persentase missing values
        missing_after = df_clean.isnull().sum()
        missing_pct_after = (missing_after / len(df_clean)) * 100
        
        # Tampilkan tabel missing values
        missing_df_after = pd.DataFrame({
            'Kolom': missing_after.index,
            'Missing Count': missing_after.values,
            'Missing %': missing_pct_after.values
        }).round(2)
        missing_df_after = missing_df_after[missing_df_after['Missing Count'] > 0]
        
        if not missing_df_after.empty:
            st.dataframe(missing_df_after)
        else:
            st.success("✅ Tidak ada missing values!")
        
        # Visualisasi missing values - sesudah
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        msno.bar(df_clean, ax=ax2, color='green')
        ax2.set_title('Missing Values - Data Bersih')
        st.pyplot(fig2)
    
    # Summary perbandingan
    st.markdown("---")
    st.markdown("#### 📈 **Ringkasan Preprocessing:**")
    
    total_missing_before = df_raw_viz.isnull().sum().sum()
    total_missing_after = df_clean.isnull().sum().sum()
    
    col_summary1, col_summary2, col_summary3 = st.columns(3)
    
    with col_summary1:
        st.metric("Missing Values Sebelum", total_missing_before)
    
    with col_summary2:
        st.metric("Missing Values Sesudah", total_missing_after)
    
    with col_summary3:
        reduction = total_missing_before - total_missing_after
        st.metric("Pengurangan Missing Values", reduction, delta=f"-{reduction}")
    
    # Informasi preprocessing yang dilakukan
    st.info("""
    **Langkah Preprocessing yang dilakukan:**
    - ✅ Mengubah nilai 0 menjadi NaN untuk kolom: Glucose, BloodPressure, BMI
    - ✅ Mengisi NaN dengan nilai median masing-masing kolom
    - ✅ Menghapus kolom Insulin dan SkinThickness dari model
    - ✅ Kolom yang tersisa untuk prediksi: Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age
    """)

def show_data_comparison(df_raw, df_clean):
    """Menampilkan perbandingan statistik data sebelum dan sesudah"""
    
    st.subheader("📋 Perbandingan Statistik Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### **Data Mentah**")
        st.dataframe(df_raw.describe().round(2))
    
    with col2:
        st.markdown("#### **Data Bersih**")
        st.dataframe(df_clean.describe().round(2))

def show_clustering(df, input_df):
    #preprocess
    features = df.drop("Outcome", axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    #k-means
    kmeans= KMeans(n_clusters=3, random_state=42, n_init="auto")
    clusters =kmeans.fit_predict(scaled_features)

    #PCA
    pca=PCA(n_components=2)
    pca_result= pca.fit_transform(scaled_features)

    #Transform user input
    user_scaled = scaler.transform(input_df)
    user_pca = pca.transform(user_scaled)
    user_cluster = kmeans.predict(user_scaled)[0]

    # Hitung Silhouette Score
    sil_score = silhouette_score(scaled_features, clusters)

    st.markdown(f"📈 **Silhouette Score untuk KMeans (k=3):** `{sil_score:.4f}`")
    if sil_score < 0.5:
        st.info("🔍 Skor ini menunjukkan bahwa clustering masih bisa diperbaiki. Mungkin k terlalu kecil atau data overlap.")
    elif sil_score < 0.7:
        st.success("👍 Clustering cukup baik, tapi masih ada sedikit overlap.")
    else:
        st.success("🚀 Clustering sangat baik! Cluster saling terpisah jelas.")

    # DataFrame untuk visualisasi
    pca_df = pd.DataFrame(pca_result, columns=["PCA1", "PCA2"])
    pca_df["Cluster"] = clusters.astype(str)

    # Plot
    st.subheader("📊 Hasil Clustering (berdasarkan input Anda)")

    fig = go.Figure()

    # Tambahkan data semua cluster
    for clust in sorted(pca_df["Cluster"].unique()):
        subset = pca_df[pca_df["Cluster"] == clust]
        fig.add_trace(go.Scatter(
            x=subset["PCA1"], y=subset["PCA2"],
            mode='markers',
            name=f'Cluster {clust}',
            marker=dict(size=7),
            opacity=0.6
        ))

    # Tambahkan titik user
    fig.add_trace(go.Scatter(
        x=[user_pca[0][0]], y=[user_pca[0][1]],
        mode='markers+text',
        name="Anda",
        marker=dict(size=12, color='black', symbol='x'),
        text=["Anda"],
        textposition="top center"
    ))

    fig.update_layout(title="Visualisasi PCA Clustering (Dengan Input Anda)", 
                      xaxis_title="PCA1", yaxis_title="PCA2")
    st.plotly_chart(fig)

    st.markdown(f"📌 **Input Anda termasuk dalam Cluster {user_cluster}**")

# Load data & model
df_raw, df_raw_viz = load_raw_data()
df = load_data()
model = load_model()

st.title("Dashboard Prediksi Diabetes 🩺")
st.markdown("Prediksi kemungkinan diabetes berdasarkan data medis menggunakan model Naive Bayes.")

# Buat tabs
tab1, tab2 = st.tabs(["🔮 Prediksi", "📊 Visualisasi"])

with tab1:
    st.subheader("🔧 Input Parameter untuk Prediksi")
    
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.slider("Pregnancies", int(df.Pregnancies.min()), int(df.Pregnancies.max()), int(df.Pregnancies.mean()))
        glucose = st.slider("Glucose", int(df.Glucose.min()), int(df.Glucose.max()), int(df.Glucose.mean()))
        bp = st.slider("BloodPressure", int(df.BloodPressure.min()), int(df.BloodPressure.max()), int(df.BloodPressure.mean()))

    with col2:
        bmi = st.slider("BMI", float(df.BMI.min()), float(df.BMI.max()), float(df.BMI.mean()))
        dpf = st.slider("DiabetesPedigreeFunction", float(df.DiabetesPedigreeFunction.min()), float(df.DiabetesPedigreeFunction.max()), float(df.DiabetesPedigreeFunction.mean()))
        age = st.slider("Age", int(df.Age.min()), int(df.Age.max()), int(df.Age.mean()))

    # Input dataframe sesuai dengan kolom yang digunakan model (tanpa Insulin dan SkinThickness)
    feature_names = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"]
    input_df = pd.DataFrame([[pregnancies, glucose, bp, bmi, dpf, age]], columns=feature_names)

    if st.button("Prediksi"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        st.subheader("🔍 Hasil Prediksi")
        if prediction == 1:
            st.error(f"🩸 Diabetes dengan probabilitas {probability[1]:.2%}")
        else:
            st.success(f"✅ Tidak Diabetes dengan probabilitas {probability[0]:.2%}")

        st.plotly_chart(
            go.Figure(data=[go.Bar(x=["Tidak Diabetes", "Diabetes"], y=probability, marker_color=['blue', 'red'])])
            .update_layout(title="Probabilitas Kelas", yaxis=dict(range=[0,1]))
        )

        st.pyplot(plot_roc_curve(model, df))

        st.markdown("### 💡 Rekomendasi")
        if prediction == 1:
            st.markdown("- Konsultasi dengan dokter segera.\n- Pantau kadar gula darah secara rutin.")
        else:
            st.markdown("- Jaga pola makan sehat.\n- Lanjutkan gaya hidup aktif.")

        # Tambahan: Clustering
        st.markdown("---")
        show_clustering(df, input_df)

with tab2:
    # Tambahkan visualisasi missing values
    show_missing_value_comparison(df_raw_viz, df)
    
    st.markdown("---")
    show_data_comparison(df_raw, df)