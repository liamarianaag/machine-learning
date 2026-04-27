import streamlit as st
import joblib
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Iris Classifier", page_icon="🌸")

st.title("Clasificador de Flores Iris")
st.write("Ingresa las dimensiones del sépalo y pétalo para predecir la especie.")

# Sidebar para selección de modelo
st.sidebar.header("Configuración")
modelo_choice = st.sidebar.selectbox(
    "Selecciona el modelo:",
    ("KNN (modelo_iris_knn.pkl)", "SVM (modelo_iris_svm.pkl)")
)

# Cargar el modelo seleccionado
def load_model(name):
    if "KNN" in name:
        return joblib.load("modelo_iris_knn.pkl")
    else:
        return joblib.load("modelo_iris_svm.pkl")

model = load_model(modelo_choice)

# Inputs de usuario
col1, col2 = st.columns(2)

with col1:
    sepal_l = st.number_input("Largo del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.1)
    sepal_w = st.number_input("Ancho del Sépalo (cm)", min_value=0.0, max_value=10.0, value=3.5)

with col2:
    petal_l = st.number_input("Largo del Pétalo (cm)", min_value=0.0, max_value=10.0, value=1.4)
    petal_w = st.number_input("Ancho del Pétalo (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Predicción
if st.button("Predecir"):
    # Organizar datos para el modelo
    input_data = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    
    prediction = model.predict(input_data)
    
    # Mapeo de nombres (ajusta según cómo entrenaste tu modelo)
    target_names = ['Setosa', 'Versicolor', 'Virginica']
    result = target_names[prediction[0]]
    
    st.success(f"El modelo predice que es una: **{result}**")
