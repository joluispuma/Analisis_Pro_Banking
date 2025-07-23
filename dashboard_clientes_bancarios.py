import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix, RocCurveDisplay
from sklearn.calibration import calibration_curve

# =====================================================
# CONFIGURACIÓN DEL DASHBOARD
# =====================================================
st.set_page_config(page_title="Dashboard de Clientes Bancarios", layout="wide")
sns.set(style="whitegrid")

# =====================================================
# CARGA DE DATOS (DEFAULT O CARGADO POR USUARIO)
# =====================================================
st.sidebar.header("Cargar Datos")
#Se brinda la opcion de cargar un archivo propio en formato CSV que conserve la estructura 
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
#Validacion de carga en el caso de hacerlo de lo contrario se realiza una carga aleatoria de datos
if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=";")
    st.success("Dataset personalizado cargado con éxito.")
else:
    st.info("Usando dataset simulado por defecto.")
    #En este punto le damos al randomizer una semilla predefinida para que nuestra valoracion aleatoria sea la misma
    #Esto para efectos de la exposicion.
    np.random.seed(42) #<-----Semilla 42
    n = 10000 #generar diez mil registros 
    #Se randomizan solo las variables importantes para el analisis de datos.
    df = pd.DataFrame({
        'CreditScore': np.random.randint(300, 850, n),
        'Geography': np.random.choice(['Bogota', 'Valle del cauca', 'Antioquia'], n),
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Age': np.random.randint(18, 92, n),
        'Tenure': np.random.randint(0, 11, n),
        'Balance': np.random.uniform(0, 250000, n),
        'NumOfProducts': np.random.randint(1, 4, n),
        'HasCrCard': np.random.randint(0, 2, n),
        'IsActiveMember': np.random.randint(0, 2, n),
        'EstimatedSalary': np.random.uniform(10000, 200000, n),
        'Complain': np.random.randint(0, 2, n),
        'Exited': np.random.randint(0, 2, n),
        'Satisfaction Score': np.random.randint(1, 6, n)
    })

# =====================================================
# BARRA DE FILTROS
# =====================================================

#Barra lateral con label
st.sidebar.header("Filtros Interactivos")
#Filtro por regiones con multiset, permite seleccionar las regiones que contenga el DF mediante la funcion options, 
#   por defecto precarga todas las opciones con default.
Regiones = st.sidebar.multiselect("Filtrar por Región", options=df["Geography"].unique(), default=df["Geography"].unique())
#Filtro de genero mediante selectbox que permite elegir una solo opcion.
filtro_genero = st.sidebar.selectbox("Género", ["Todos", "Masculino", "Femenino"])
#Filtro de productos donde se dice la cantidad de productos que contiene el cliente para filtrar con multiselect.
filtro_productos = st.sidebar.multiselect("Número de productos", sorted(df["NumOfProducts"].unique()), default=sorted(df["NumOfProducts"].unique()))
#Filtro segun si tiene tarjeta de credito o no.
filtro_credito = st.sidebar.radio("¿Posee tarjeta de crédito?", ["Todos", "Sí", "No"])
#Filtro segun si se ha quejado o no con radio button
filtro_queja = st.sidebar.radio("¿Ha presentado quejas?", ["Todos", "Sí", "No"])
#Filtro de balance en la cuenta con slider que da un rango menor y mayor.
rango_balance = st.sidebar.slider("Filtrar por Balance Promedio ($)", float(df["Balance"].min()), float(df["Balance"].max()), (float(df["Balance"].min()), float(df["Balance"].max())))
#Aquí es donde los valores elegidos por el usuario se aplican al DataFrame
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
df_filtrado = df_filtrado[(df_filtrado["Balance"] >= rango_balance[0]) & (df_filtrado["Balance"] <= rango_balance[1]) & (df_filtrado["Geography"].isin(Regiones))]

# =====================================================
# MÉTRICAS PRINCIPALES
# =====================================================
st.title("Dashboard Analítico de Clientes Bancarios")
col1, col2, col3 = st.columns(3)
col1.metric("Clientes Totales", len(df_filtrado))
col2.metric("Fuga de clientes (%)", f"{df_filtrado['Exited'].mean() * 100:.2f}%")
col3.metric("Satisfacción promedio", f"{df_filtrado['Satisfaction Score'].mean():.2f} / 5")

# =====================================================
# GRÁFICOS DE EXPLORACIÓN
# =====================================================
def graficar(title, fig):
    st.markdown(f"### {title}")
    st.pyplot(fig)

fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.histplot(df_filtrado["Age"], bins=30, kde=True, ax=ax1)
ax1.axvline(df_filtrado["Age"].mean(), color='red', linestyle='--')
ax1.axvline(df_filtrado["Age"].median(), color='green', linestyle='-.')
graficar("Distribución de Edad", fig1)

fig2, ax2 = plt.subplots()
sns.barplot(data=df_filtrado, x="NumOfProducts", y="Exited", estimator='mean', ax=ax2)
graficar("Fuga según Número de Productos", fig2)

fig3, ax3 = plt.subplots()
sns.barplot(data=df_filtrado, x="Gender", y="Exited", estimator='mean', ax=ax3)
graficar("Fuga por Género", fig3)

fig4, ax4 = plt.subplots()
sns.boxplot(data=df_filtrado, x="Exited", y="Balance", ax=ax4)
graficar("Balance vs Fuga", fig4)

fig5, ax5 = plt.subplots()
corr = df_filtrado.corr(numeric_only=True)
sns.heatmap(corr[['Exited']].sort_values(by='Exited', ascending=False), annot=True, cmap='coolwarm', ax=ax5)
graficar("Correlación con 'Exited'", fig5)

fig6, ax6 = plt.subplots()
region_exit = df_filtrado.groupby("Geography")["Exited"].mean().sort_values(ascending=False).reset_index()
sns.barplot(data=region_exit, x="Geography", y="Exited", ax=ax6)
graficar("Fuga por Región", fig6)

fig7, ax7 = plt.subplots()
sns.boxplot(data=df_filtrado, x="Exited", y="Satisfaction Score", ax=ax7)
ax7.set_xticklabels(["Se quedaron", "Se fueron"])
graficar("Satisfacción y Fuga", fig7)

# =====================================================
# MODELO PREDICTIVO RANDOM FOREST
# =====================================================
st.markdown("---")
st.markdown("## Entrenar Modelo Predictivo de Fuga")
#Implementacion de boton de carga de datos y entrenamiento del modelo de machine learning
if st.button("Entrenar y Evaluar Modelo"):
    #Se crea una copia del DF para no afectar el original, se codifican las variables como numeros 
    #ya que los modelos de Skicit no aceptan valores en String directamente
    df_ml = df.copy()
    df_ml['Gender'] = LabelEncoder().fit_transform(df_ml['Gender'])
    df_ml['Geography'] = LabelEncoder().fit_transform(df_ml['Geography'])
    #Seleccion de variables predictoras, y Y es la variable objetivo de si el cliente salio o no.
    features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    X = df_ml[features]
    y = df_ml['Exited']
#Division para entrenamiento y prueba se divide el dataset con: 75% para entrenamiento y 25% para probar
#Con el valor stratify = Y aseguramos que la proporcion de clientes que salen o se quedan se mantenga para ambos conjuntos
#
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
#Modelo aplicado Random Forest donde se regularizan los parametros para evitar cualquier sobreajuste
#Parametrizamos Max_depth=7 para que no sean arboles muy profundos, Min_sample_leaf=10 para asegurar que cada hoja tenga 
#ejemplos suficientes, y max_features=sqrt para que solo seleccione algunas variables en cada division.
    modelo_rf = RandomForestClassifier(
        n_estimators=100, max_depth=7, min_samples_leaf=10,
        min_samples_split=20, max_features='sqrt', random_state=42
    )
#Aplicacion de validacion cruzada que es la tecnica que ayuda a evaluar el rendimiento real del modelo y evita alucinaciones
#Con resultados demasiado perfectos, El modelo se entrena y evalua 5 veces con N_splits=5 lo cual mejora las metricas frente a
#Divisiones aleatorias, tambien calculamos ROC-AUC que separa los clientes y con el Recall detectamos a los clientes que se nos van.
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    roc_cv = cross_val_score(modelo_rf, X, y, cv=kfold, scoring='roc_auc')
    recall_cv = cross_val_score(modelo_rf, X, y, cv=kfold, scoring='recall')
    st.write(f"ROC-AUC (CV promedio): {roc_cv.mean():.3f} ± {roc_cv.std():.3f}")
    st.write(f"Recall (CV promedio): {recall_cv.mean():.3f} ± {recall_cv.std():.3f}")
#Aplicacion del entrenamiento y evaluacion final, se entrena el modelo con el conjunto de entrenamiento y predice sobre el conjunto de prueba
#y_pred para las clases 0 o 1 y y_proba con las probabilidades de que se vaya.
    modelo_rf.fit(X_train, y_train)
    y_pred = modelo_rf.predict(X_test)
    y_proba = modelo_rf.predict_proba(X_test)[:, 1]
#Visuales metricas realis del conjunto de pruebas
    roc_auc = roc_auc_score(y_test, y_proba)
    recall = recall_score(y_test, y_pred)
    st.metric("ROC-AUC (Test)", f"{roc_auc:.3f}")
    st.metric("Recall (Test)", f"{recall:.3f}")
#Mostrar la cantidad de clientes que fueron clasificados correctamente o no. (TN, FP,FN,TP)
    fig_cm, ax_cm = plt.subplots()
#La matriz de confusion es una herramienta para evaluar el rendimiento de modelos permitiendo ver las predicciones acertadas
# Tambien cuantas erradas y organizarlas en una tabla para interpretar.
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="YlGnBu", ax=ax_cm)
    ax_cm.set_title("Matriz de Confusión")
    st.pyplot(fig_cm)
#Curva de ROC muestra la curva de prediccion del modelo.
    fig_roc, ax_roc = plt.subplots()
    RocCurveDisplay.from_estimator(modelo_rf, X_test, y_test, ax=ax_roc)
    ax_roc.set_title("Curva ROC")
    st.pyplot(fig_roc)
#Muestra la curva de Calibracion que verifica las probabilidades que da el modelo si corresponde a la realidad.
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    fig_cal, ax_cal = plt.subplots()
    ax_cal.plot(prob_pred, prob_true, marker='o')
    ax_cal.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax_cal.set_title("Curva de Calibración")
    st.pyplot(fig_cal)
#Plus con resultados de salida del top 10 de clientes que tienen mas tendencia a abandonar el banco.
    df_riesgo = X_test.copy()
    df_riesgo['ProbabilidadFuga'] = y_proba
    top_riesgo = df_riesgo.sort_values(by='ProbabilidadFuga', ascending=False).head(10)
    st.markdown("### 🚨 Top 10 Clientes con Mayor Riesgo de Fuga")
    st.dataframe(top_riesgo.style.format({"ProbabilidadFuga": "{:.2%}"}))
