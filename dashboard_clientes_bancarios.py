import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# =====================================================
# DASHBOARD STREAMLIT: ANÁLISIS DE FUGA DE CLIENTES
# =====================================================

# Configuración de la página
st.set_page_config(page_title="Dashboard de Clientes Bancarios", layout="wide")
sns.set(style="whitegrid")

# =====================================================
# CARGA DE DATOS
# =====================================================
@st.cache_data
def cargar_datos():
    return pd.read_csv("Bank-Customer-Attrition-Insights-Data.csv", sep=";")

df = cargar_datos()

# =====================================================
# TÍTULO PRINCIPAL
# =====================================================
st.title("Dashboard Analítico de Clientes Bancarios")
st.markdown("""
Este dashboard permite visualizar el perfil de los clientes que han abandonado la entidad bancaria.
Puedes explorar diferentes segmentos y entender los factores asociados a la fuga de clientes.
""")

# =====================================================
# BARRA LATERAL DE FILTROS
# =====================================================
st.sidebar.header("Filtros Interactivos")

Regiones = st.sidebar.multiselect("Filtrar por Región", options=df["Geography"].unique(), default=df["Geography"].unique())
filtro_genero = st.sidebar.selectbox("Género", ["Todos", "Masculino", "Femenino"])
filtro_productos = st.sidebar.multiselect("Número de productos contratados", sorted(df["NumOfProducts"].unique()), default=sorted(df["NumOfProducts"].unique()))
filtro_credito = st.sidebar.radio("¿Posee tarjeta de crédito?", ["Todos", "Sí", "No"])
filtro_queja = st.sidebar.radio("¿Ha presentado quejas?", ["Todos", "Sí", "No"])
rango_balance = st.sidebar.slider(
    "Filtrar por Balance Promedio ($)",
    min_value=float(df["Balance"].min()),
    max_value=float(df["Balance"].max()),
    value=(float(df["Balance"].min()), float(df["Balance"].max()))
)

# =====================================================
# APLICACIÓN DE FILTROS
# =====================================================
df_filtrado = df.copy()

if filtro_genero != "Todos":
    df_filtrado = df_filtrado[df_filtrado["Gender"] == ("Male" if filtro_genero == "Masculino" else "Female")]

if filtro_productos:
    df_filtrado = df_filtrado[df_filtrado["NumOfProducts"].isin(filtro_productos)]

if filtro_credito == "Sí":
    df_filtrado = df_filtrado[df_filtrado["HasCrCard"] == 1]
elif filtro_credito == "No":
    df_filtrado = df_filtrado[df_filtrado["HasCrCard"] == 0]

if filtro_queja == "Sí":
    df_filtrado = df_filtrado[df_filtrado["Complain"] == 1]
elif filtro_queja == "No":
    df_filtrado = df_filtrado[df_filtrado["Complain"] == 0]

df_filtrado = df_filtrado[
    (df_filtrado["Balance"] >= rango_balance[0]) &
    (df_filtrado["Balance"] <= rango_balance[1]) &
    (df_filtrado["Geography"].isin(Regiones))
]

# =====================================================
# MÉTRICAS PRINCIPALES
# =====================================================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Clientes Totales", len(df_filtrado))
with col2:
    st.metric("Fuga de clientes (%)", f"{df_filtrado['Exited'].mean() * 100:.2f}%")
with col3:
    st.metric("Satisfacción promedio", f"{df_filtrado['Satisfaction Score'].mean():.2f} / 5")

# =====================================================
# GRAFICO 1: DISTRIBUCIÓN DE EDAD
# =====================================================
st.markdown("###  Distribución de Edad")
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.histplot(df_filtrado["Age"], bins=30, kde=True, color="skyblue", edgecolor="black", ax=ax1)
ax1.axvline(df_filtrado["Age"].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df_filtrado["Age"].mean():.1f} años')
ax1.axvline(df_filtrado["Age"].median(), color='green', linestyle='-.', linewidth=2, label=f'Mediana: {df_filtrado["Age"].median():.1f} años')
ax1.set_xlabel("Edad (años)")
ax1.set_ylabel("Cantidad de Clientes")
ax1.legend()
st.pyplot(fig1)

# =====================================================
# GRAFICO 2: FUGA SEGÚN NÚMERO DE PRODUCTOS
# =====================================================
st.markdown("###  Fuga de Clientes según Número de Productos")
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.barplot(data=df_filtrado, x="NumOfProducts", y="Exited", estimator='mean', palette="Set2", ax=ax2)
ax2.set_ylabel("Proporción que Abandonó")
ax2.set_xlabel("Número de Productos")
st.pyplot(fig2)

# =====================================================
# GRAFICO 3: FUGA POR GÉNERO
# =====================================================
st.markdown("###  Fuga por Género")
fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.barplot(data=df_filtrado, x="Gender", y="Exited", estimator='mean', palette="pastel", ax=ax3)
ax3.set_ylabel("Proporción que Abandonó")
ax3.set_xlabel("Género")
st.pyplot(fig3)

# =====================================================
# GRAFICO 4: BALANCE VS FUGA
# =====================================================
st.markdown("###  Balance promedio vs Fuga de Clientes")
fig4, ax4 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df_filtrado, x="Exited", y="Balance", palette="coolwarm", ax=ax4)
ax4.set_xlabel("¿Cliente se fue? (0 = No, 1 = Sí)")
ax4.set_ylabel("Balance en Cuenta ($)")
st.pyplot(fig4)

# =====================================================
# GRAFICO 5: CORRELACIÓN CON VARIABLE 'EXITED'
# =====================================================
st.markdown("###  Variables más relacionadas con la Fuga")
corr = df_filtrado.corr(numeric_only=True)
fig5, ax5 = plt.subplots(figsize=(10, 6))
sns.heatmap(corr[['Exited']].sort_values(by='Exited', ascending=False), annot=True, cmap='Spectral', linewidths=0.5, ax=ax5)
ax5.set_title("Correlación de Variables con la Fuga de Clientes")
st.pyplot(fig5)

# =====================================================
# GRAFICO 6: FUGA POR REGIÓN
# =====================================================
st.markdown("###  Fuga de Clientes por Región")
df_region = df_filtrado.groupby("Geography")["Exited"].mean().sort_values(ascending=False).reset_index()
fig6, ax6 = plt.subplots()
sns.barplot(data=df_region, x="Geography", y="Exited", palette="coolwarm", ax=ax6)
ax6.set_ylabel("Tasa de Fuga (%)")
st.pyplot(fig6)

# =====================================================
# GRAFICO 7: SATISFACCIÓN VS RETENCIÓN
# =====================================================
st.markdown("###  ¿Cómo influye la satisfacción en la retención?")
fig7, ax7 = plt.subplots()
sns.boxplot(data=df_filtrado, x="Exited", y="Satisfaction Score", palette="viridis", ax=ax7)
ax7.set_xticklabels(["Se quedaron", "Se fueron"])
ax7.set_title("Nivel de Satisfacción por Estado del Cliente")
st.pyplot(fig7)

# =====================================================
# MÉTRICA EXTRA: QUEJAS ANTES DE IRSE
# =====================================================
quejas_fuga = df_filtrado[df_filtrado["Exited"] == 1]["Complain"].mean() * 100
st.metric("📢Clientes que se quejaron antes de irse", f"{quejas_fuga:.1f}%")

# =====================================================
# CONCLUSIONES NARRATIVAS
# =====================================================
st.markdown("---")
st.markdown("## Conclusiones")
st.markdown("""
- ###  Edad promedio en fuga: 
La edad promedio de los clientes que abandonan la entidad se ubica en 38.9 años, con una mediana de 37 años. Esta concentración en edades económicamente activas sugiere que la fuga no proviene de extremos (jóvenes impacientes o adultos mayores desatendidos), sino de una franja madura, con capacidad de análisis financiero y expectativa de valor agregado.

  ######  Recomendación: diseñar programas de fidelización personalizados para este rango etario, que combinen beneficios tangibles y atención proactiva.
            
- ###  El número ideal de productos no es mayor, es equilibrado: 
Sorprendentemente, los clientes con 2 productos contratados son los más fieles, mientras que quienes tienen 3 o 4 productos muestran índices de fuga superiores al 80%. Esto rompe el mito de que más productos = más fidelización.

  ######  Hipótesis clave: los clientes con muchos productos pueden sentirse sobrevendidos o no acompañados en su experiencia posventa.
            
  ######  Recomendación: revisar el proceso de acompañamiento de clientes multiproducto, y evaluar si existe fatiga por complejidad o desatención.
            
- ###  Género y fuga:
Las mujeres presentan una mayor tasa de fuga que los hombres (~25% vs 17%). Esta diferencia estadísticamente visible invita a revisar si la propuesta de valor es percibida de manera desigual según el género.

  ######  Preguntas clave: ¿Hay diferencias en la atención o en los canales de servicio? ¿Qué nivel de personalización tienen las comunicaciones?
            
- ###  Balance medio-alto, pero poco satisfecho:
Los clientes que se fugan tienen balances en promedio más altos que los que se quedan, lo que sugiere que no es un problema de liquidez sino de percepción de valor. Son clientes que podrían quedarse, pero no sienten reciprocidad.

  ######  Recomendación: priorizar este segmento en campañas de retención con incentivos diferenciales, atención VIP, y revisión del portafolio.

- ###  Variables clave: la queja es la gran bandera roja
Según la matriz de correlación, la variable con mayor poder predictivo de la fuga es Complain (correlación perfecta), seguida por Age (0.29) y Balance (0.12).

  ######  Dato alarmante: el 99.8% de los clientes que se fueron presentaron una queja antes de hacerlo. Esto convierte al sistema de PQRS en una herramienta de predicción poderosa.

- ###  Análisis territorial: Bogotá como punto crítico
A nivel geográfico, Bogotá representa el foco más alto de fuga con una tasa del 32%, por encima de regiones como Antioquia y Valle del Cauca. Esto puede estar vinculado a la sobrecarga de canales, la rotación del personal o la desconexión emocional con la marca.

  ######  Recomendación: segmentar campañas de retención por región e iniciar con pilotos en Bogotá centrados en mejorar experiencia, cercanía y agilidad.        
""")

# =====================================================
# CRÉDITOS
# =====================================================
st.markdown("---")
st.caption("Desarrollado por Team Controller | Análisis académico de fuga de clientes bancarios. Streamlit + Python.")
