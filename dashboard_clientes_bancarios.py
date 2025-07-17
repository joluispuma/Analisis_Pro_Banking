#=====================================================
# DASHBOARD STREAMLIT: ANÁLISIS DE FUGA DE CLIENTES
# =====================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================
# Cargar y preparar los datos
# ========================================
st.set_page_config(page_title="Dashboard de Clientes", layout="wide")

@st.cache_data
def cargar_datos():
    df = pd.read_csv("Bank-Customer-Attrition-Insights-Data.csv", sep=";")
    return df

df = cargar_datos()

# ========================================
# Título principal del dashboard
# ========================================
st.title("Dashboard Analítico de Clientes Bancarios")
st.markdown("""
Este dashboard permite visualizar el perfil de los clientes que han abandonado la entidad bancaria.
A través de los filtros puedes segmentar a los usuarios por género, número de productos, tarjetas de crédito, quejas y balance promedio en su cuenta.
""")

# ========================================
# Barra lateral con filtros
# ========================================
st.sidebar.header("Filtros")

# Filtro por Region
Regiones = st.sidebar.multiselect("Filtrar por Region", options=df["Geography"].unique(), default=df["Geography"].unique())
# Filtro por Género
filtro_genero = st.sidebar.selectbox("Género", ["Todos", "Masculino", "Femenino"])
# Filtro por Número de Productos
filtro_productos = st.sidebar.multiselect("Número de productos contratados", sorted(df["NumOfProducts"].unique()), default=sorted(df["NumOfProducts"].unique()))
# Filtro por Tarjeta de Crédito
filtro_credito = st.sidebar.radio("¿Posee tarjeta de crédito?", ["Todos", "Sí", "No"])
# Filtro por Queja
filtro_queja = st.sidebar.radio("¿Ha presentado quejas?", ["Todos", "Sí", "No"])
# Filtro por Balance (slider interactivo)
rango_balance = st.sidebar.slider(
    "Filtrar por Balance Promedio ($)",
    min_value=float(df["Balance"].min()),
    max_value=float(df["Balance"].max()),
    value=(float(df["Balance"].min()), float(df["Balance"].max()))
)

# ========================================
# Aplicar filtros al dataframe
# ========================================
df_filtrado = df.copy()

# Filtro de género
if filtro_genero != "Todos":
    df_filtrado = df_filtrado[df_filtrado["Gender"] == ("Male" if filtro_genero == "Masculino" else "Female")]

# Filtro de productos
if filtro_productos:
    df_filtrado = df_filtrado[df_filtrado["NumOfProducts"].isin(filtro_productos)]

# Filtro tarjeta de crédito
if filtro_credito == "Sí":
    df_filtrado = df_filtrado[df_filtrado["HasCrCard"] == 1]
elif filtro_credito == "No":
    df_filtrado = df_filtrado[df_filtrado["HasCrCard"] == 0]

# Filtro de quejas
if filtro_queja == "Sí":
    df_filtrado = df_filtrado[df_filtrado["Complain"] == 1]
elif filtro_queja == "No":
    df_filtrado = df_filtrado[df_filtrado["Complain"] == 0]

# Filtro de balance
df_filtrado = df_filtrado[(df_filtrado["Balance"] >= rango_balance[0]) & 
                          (df_filtrado["Balance"] <= rango_balance[1]) & (df["Geography"].isin(Regiones))]


# ========================================
# KPIs Principales
# ========================================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Clientes Totales", len(df_filtrado))
with col2:
    porcentaje_fuga = df_filtrado["Exited"].mean() * 100
    st.metric("Fuga de clientes (%)", f"{porcentaje_fuga:.2f}%")
with col3:
    promedio_satisfaccion = df_filtrado["Satisfaction Score"].mean()
    st.metric("Satisfacción promedio", f"{promedio_satisfaccion:.2f} / 5")

# ========================================
# GRAFICO 1: Distribución de Edad
# ========================================
st.markdown("###Distribución de Edad de los Clientes")
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.histplot(df_filtrado["Age"], bins=30, kde=True, color="skyblue", edgecolor="black", ax=ax1)
ax1.axvline(df_filtrado["Age"].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df_filtrado["Age"].mean():.1f} años')
ax1.axvline(df_filtrado["Age"].median(), color='green', linestyle='-.', linewidth=2, label=f'Mediana: {df_filtrado["Age"].median():.1f} años')
ax1.set_xlabel("Edad (años)")
ax1.set_ylabel("Cantidad de Clientes")
ax1.set_title("Distribución de Edad")
ax1.legend()
st.pyplot(fig1)

# ========================================
# GRAFICO 2: Fuga según Número de Productos
# ========================================
st.markdown("###Fuga de Clientes según Número de Productos")
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.barplot(data=df_filtrado, x="NumOfProducts", y="Exited", estimator='mean', palette="Set2", ax=ax2)
ax2.set_title("Proporción de Fuga por Productos Contratados")
ax2.set_ylabel("Proporción que Abandonó")
ax2.set_xlabel("Número de Productos")
st.pyplot(fig2)

# ========================================
# GRAFICO 3: Fuga por Género
# ========================================
st.markdown("###Fuga por Género")
fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.barplot(data=df_filtrado, x="Gender", y="Exited", estimator='mean', palette="pastel", ax=ax3)
ax3.set_title("Proporción de Fuga por Género")
ax3.set_ylabel("Proporción que Abandonó")
ax3.set_xlabel("Género")
st.pyplot(fig3)

# ========================================
# GRAFICO 4: Boxplot de Balance vs Fuga
# ========================================
st.markdown("###Balance promedio vs Fuga de Clientes")
fig4, ax4 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df_filtrado, x="Exited", y="Balance", palette="coolwarm", ax=ax4)
ax4.set_title("Distribución del Balance según si el cliente se fue")
ax4.set_xlabel("¿Cliente se fue? (0 = No, 1 = Sí)")
ax4.set_ylabel("Balance en Cuenta ($)")
st.pyplot(fig4)

# ========================================
# Créditos
# ========================================
st.markdown("---")
st.caption("Desarrollado por Team Controller | Análisis académico de fuga de clientes con Streamlit y Python.")
