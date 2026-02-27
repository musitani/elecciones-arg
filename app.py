import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 1. CONFIGURACIÓN DE LA PÁGINA: Debe ser la primera instrucción de Streamlit -----
st.set_page_config(
    page_title="Predicción Electoral 2027", 
    layout="wide", 
    page_icon="📊"
)

# 2. CARGA Y TRANSFORMACIÓN DE DATOS (DATA WRANGLING)
@st.cache_data  # Esto optimiza la app para que no lea el CSV cada vez que tocas un botón
def get_clean_data():
    # Leemos el archivo original
    df = pd.read_csv('elecciones-argentina.csv')
    
    # El CSV original es "Ancho" (una columna por año). 
    # Para graficar fácilmente, lo pasamos a formato "Largo" (una fila por cada observación).
    years = ['2015', '2017', '2019', '2021', '2023', '2025']
    lista_temporal = []
    
    for year in years:
        # Extraemos las columnas de ese año específico
        temp = df[['Provincia', f'Tipo_Eleccion_{year}', f'Participacion_{year}']].copy()
        temp.columns = ['Provincia', 'Tipo', 'Participacion']
        temp['Año'] = int(year) # Convertimos el año a número para poder hacer cálculos matemáticos
        lista_temporal.append(temp)
    
    return pd.concat(lista_temporal)

df_long = get_clean_data()

# 3. TÍTULO Y ESTÉTICA (HTML/CSS)
st.markdown("<h1 style='text-align: center; color: #00441b;'>Monitor de Participación Electoral</h1>", unsafe_allow_html=True)
st.markdown("---")

# 4. LÓGICA DE ESTIMACIÓN 2027 (MATEMÁTICAS)
# Solo usaremos años de elecciones Presidenciales para predecir otra Presidencial (2027)
# Así evitamos que la baja participación de las legislativas ensucie la tendencia.
df_presidencial = df_long[df_long['Tipo'].str.contains("Presidencial")].groupby('Año')['Participacion'].mean().reset_index()

# Aplicamos Regresión Lineal: y = mx + b
# x = Años (2015, 2019, 2023) | y = Participación
x = df_presidencial['Año'].values
y = df_presidencial['Participacion'].values
coeficientes = np.polyfit(x, y, 1) # Calcula la pendiente (m) y la intersección (b)
prediccion_2027 = coeficientes[0] * 2027 + coeficientes[1]

# 5. DISEÑO DE LA INTERFAZ (COLUMNAS)
col_grafico, col_info = st.columns([2, 1])

with col_grafico:
    st.subheader("📈 Evolución y Proyección 2027")
    
    # Creamos un gráfico interactivo con Plotly
    fig = go.Figure()

    # Línea de Datos Reales
    fig.add_trace(go.Scatter(
        x=df_presidencial['Año'], 
        y=df_presidencial['Participacion'],
        name="Datos Reales (Presidenciales)",
        mode='lines+markers',
        line=dict(color='#1f77b4', width=4)
    ))

    # Punto de Predicción 2027
    fig.add_trace(go.Scatter(
        x=[2023, 2027], 
        y=[y[-1], prediccion_2027], # Une el último dato real con la predicción
        name="Proyección IA",
        line=dict(color='red', dash='dash'),
        marker=dict(size=10, symbol='star')
    ))

    fig.update_layout(template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with col_info:
    st.subheader("🎯 Resultado del Modelo")
    st.metric(label="Estimación Participación 2027", value=f"{prediccion_2027:.2f}%")
    
    st.write(f"""
    **Explicación para principiantes:**
    1. **Filtro Inteligente:** El modelo solo mira los años 2015, 2019 y 2023.
    2. **Pendiente:** Hemos detectado que la participación presidencial cae aproximadamente 
    **{abs(coeficientes[0]*4):.2f}%** cada ciclo de 4 años.
    3. **Resultado:** Si la tendencia social se mantiene, en 2027 votaría cerca del **{prediccion_2027:.1f}%** del padrón.
    """)

# 6. TABLA DETALLADA POR PROVINCIA
with st.expander("Ver detalle por Provincia (Estimación 2027)"):
    # Creamos una tabla donde a cada provincia le restamos la tendencia calculada
    df_ult = df_long[df_long['Año'] == 2023].copy()
    df_ult['Predicción 2027'] = df_ult['Participacion'] + (coeficientes[0] * 4)
    
    # Mostramos la tabla formateada
    st.dataframe(
        df_ult[['Provincia', 'Participacion', 'Predicción 2027']].style.format(precision=2),
        use_container_width=True
    )


st.success("💡 Tip de Programador: He usado `np.polyfit` porque es más ligero que cargar toda la librería de Scikit-Learn solo para una línea.")
