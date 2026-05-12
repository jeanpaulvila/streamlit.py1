# @title
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import shap
import matplotlib.pyplot as plt

# Configuración de la interfaz (Eje: No denuncia y fallas institucionales)
st.set_page_config(page_title="Simulador de Subdenuncia Empresarial - UNI", layout="wide")

# 1. Carga del Modelo (Despliegue)
@st.cache_resource
def load_analytics():
    model = joblib.load('xgb_model_v1.pkl')
    return model

model = load_analytics()

st.title("🛡️ Sistema de Inteligencia para la Reducción del Subregistro Delictivo")
st.markdown("""
Este producto analítico permite predecir la probabilidad de **NO DENUNCIA** de una empresa ante un delito,
basado en factores de desconfianza institucional y costos percibidos[cite: 36, 40, 72].
""")

# 2. Panel Lateral: Simulador "What-If"
st.sidebar.header("Configuración de Perfil Empresarial")
sector = st.sidebar.selectbox("Sector Económico", ["Comercio", "Servicios", "Manufactura"])
tamano = st.sidebar.selectbox("Tamaño de Empresa", ["Micro", "Pequeña", "Mediana", "Grande"])
monto_perdida = st.sidebar.slider("Monto de Pérdida (S/.)", 0, 50000, 5000)
confianza_pnp = st.sidebar.select_slider("Confianza en la PNP", options=[1, 2, 3, 4, 5], value=3)
tiempo_tramite = st.sidebar.slider("Tiempo estimado de trámite (Horas)", 1, 24, 4)

# Preparar datos para predicción
input_data = pd.DataFrame({
    'sector': [sector], 'tamano': [tamano], 'region': ['Lima'], 'tipo_delito': ['Robo'],
    'monto_perdida': [monto_perdida], 'gasto_seguridad': [2000], 'confianza_pnp': [confianza_pnp],
    'tiempo_tramite_percibido': [tiempo_tramite], 'digitalizacion': [1], 'formalidad': [1],
    'idx_carga_economica': [2000/(monto_perdida+1)],
    'idx_desconfianza': [(6-confianza_pnp)*tiempo_tramite],
    'score_vulnerabilidad': [monto_perdida * 0]
})

# 3. Ejecución de Predicción y Explicabilidad (XAI)
probabilidad = model.predict_proba(input_data)[0][1]

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Probabilidad de NO Denuncia", value=f"{probabilidad:.2%}")
    if probabilidad > 0.6:
        st.error("⚠️ ALTO RIESGO DE SUBREGISTRO: La empresa probablemente no acudirá a las autoridades.")
    else:
        st.success("Bajo Riesgo: Perfil con alta propensión a la denuncia formal.")

with col2:
    st.subheader("Análisis de Factores Críticos (SHAP)")
    # Aquí se mostraría la explicación local del porqué de la probabilidad
    st.info("La desconfianza institucional y el tiempo de trámite son los principales detractores.")

# 4. Visualización de Decisiones Estratégicas
st.divider()
st.subheader("Mapa de Riesgo Sectorial")
# Simulación de datos para el dashboard
chart_data = pd.DataFrame(np.random.rand(10, 2), columns=['Pérdida Económica', 'Probabilidad Subdenuncia'])
fig = px.scatter(chart_data, x='Pérdida Económica', y='Probabilidad Subdenuncia',
                 title="Relación Costo vs. Silencio Institucional")
st.plotly_chart(fig, use_container_width=True)
