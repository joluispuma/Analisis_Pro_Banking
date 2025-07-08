import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Super Store Dashboard",
    page_icon=":)"
)

def cargar_datos():
    data = pd.read_csv('C:\Analisis_Pro_Banking\super_store.csv')
    data['Order Date'] = pd.to_datetime(data['Order Date'])
    return data

df = cargar_datos()

st.sidebar.header("Filtro del Dashboard")
min_date = df['Order Date'].min()
max_date = df['Order Date'].max()
print(min_date)
print(type(min_date))
fecha_inicial, fecha_final = st.sidebar.date_input(
    "Selecciona un rango de fechas",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

df_filtrado = df[df['Order Date'].between(pd.to_datetime(fecha_inicial), pd.to_datetime(fecha_final))]

st.title("Super Store Dashboard")
st.markdown('##')

ventas_totales = df['Sales'].sum()
utilidad_total = df['Profit'].sum()
ordenes_totales = df['Order ID'].nunique()
clientes_totales = df['Customer ID'].nunique()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Ventas Totales", value=f"${ventas_totales:.2f}")
with col2:
    st.metric(label="Utilidad Total", value=f"${utilidad_total:.2f}")
with col3:
    st.metric(label="Ordenes Totales", value=f"{ordenes_totales}")
with col4:
    st.metric(label="Clientes Totales", value=f"{clientes_totales}")

st.header("Ventas y Utilidades a lo largo del tiempo")
ventas_por_utilidad = df_filtrado.set_index('Order Date').resample('M').agg({'Sales':'sum', 'Profit':'sum'}).reset_index()

fig_area = px.area(
    ventas_por_utilidad,
    x = 'Order Date',
    y = ['Sales', 'Profit'],
    title = "Evolución de Ventas y Utilidades en el Tiempo"
)

st.plotly_chart(fig_area, use_container_width=True)

st.markdown("---")

colpie, coldona = st.columns(2)

with colpie:
    ventas_por_region = df_filtrado.groupby('Region')['Sales'].sum().reset_index()
    fig_pie_region = px.pie(
        ventas_por_region,
        names='Region',
        values='Sales',
        title='Ventas por Región'
    )

st.plotly_chart(fig_pie_region, use_container_width=True)

