import streamlit as st
import joblib
import numpy as np
import os

# Configuración estética de la página
st.set_page_config(
    page_title="Iris Predictor Pro",
    page_icon="🌸",
    layout="centered"
)

# Estilos personalizados para mejorar la apariencia
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌸 Clasificador de Especies Iris")
st.info("Esta aplicación utiliza modelos de Machine Learning (KNN y SVM) para identificar la especie de una flor basada en sus medidas.")

# --- BARRA LATERAL ---
st.sidebar.header("Configuración del Modelo")
modelo_choice = st.sidebar.selectbox(
    "Selecciona el algoritmo:",
    ("K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)")
)

# Diccionario para mapear la selección al nombre del archivo
archivos_modelos = {
    "K-Nearest Neighbors (KNN)": "modelo_iris_knn.pkl",
    "Support Vector Machine (SVM)": "modelo_iris_svm.pkl"
}

# --- FUNCIÓN DE CARGA SEGURA ---
@st.cache_resource # Esto hace que el modelo se cargue una sola vez y sea más rápido
def cargar_modelo(nombre_archivo):
    if os.path.exists(nombre_archivo):
        try:
            return joblib.load(nombre_archivo)
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
            return None
    else:
        st.error(f"⚠️ No se encontró el archivo '{nombre_archivo}' en el repositorio.")
        return None

# Intentar cargar el modelo seleccionado
archivo_actual = archivos_modelos[modelo_choice]
model = cargar_modelo(archivo_actual)

# --- ENTRADA DE DATOS ---
st.subheader("Ingresa las dimensiones")
col1, col2 = st.columns(2)

with col1:
    sepal_l = st.slider("Largo del Sépalo (cm)", 4.0, 8.0, 5.1)
    sepal_w = st.slider("Ancho del Sépalo (cm)", 2.0, 4.5, 3.5)

with col2:
    petal_l = st.slider("Largo del Pétalo (cm)", 1.0, 7.0, 1.4)
    petal_w = st.slider("Ancho del Pétalo (cm)", 0.1, 2.5, 0.2)

# --- PREDICCIÓN ---
st.divider()

if model is not None:
    if st.button("Realizar Predicción"):
        # Preparar los datos
        input_data = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
        
        # Realizar predicción
        prediction = model.predict(input_data)
        
        # Mapeo de especies
        especies = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        resultado = especies.get(prediction[0], "Desconocida")
        
        # Mostrar resultado con un diseño llamativo
        st.balloons()
        st.success(f"### Resultado: Iris {resultado}")
        
        # Opcional: Mostrar probabilidades si el modelo lo permite
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_data)
            st.write("**Confianza de la predicción:**")
            st.progress(float(max(probs[0])))
else:
    st.warning("El modelo no está disponible para realizar predicciones. Verifica que subiste los archivos .pkl a tu GitHub.")
