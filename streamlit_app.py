import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configuración inicial de la página
st.set_page_config(page_title="Análisis Avanzado de Proveedores - Datamind", layout="wide")

# Estilo personalizado para la app
st.markdown(
    """
    <style>
    /* Estilo general */
    .title {
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        color: #007ACC;
        margin-bottom: 20px;
    }
    .header {
        font-size: 24px;
        font-weight: bold;
        color: #004B73;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #333333;
        margin-bottom: 15px;
    }
    .info-box {
        border: 2px solid #007ACC;
        background-color: #e6f7ff;
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 10px;
        color: #000
    }
    .sidebar .sidebar-content {
        padding: 10px;
    }
    .creditos {
        font-size: 14px;
        text-align: center;
        color: gray;
    }
    .about-box {
        padding: 20px;
        background-color: #f5f5f5;
        border-radius: 10px;
        margin-top: 20px;
        color: #000
    }
    .button {
        background-color: #007ACC;
        color: white;
        padding: 12px 25px;
        border-radius: 5px;
        font-size: 16px;
    }
    .bold {
        font-weight: bold;
    }
    .code {
        background-color: #f7f7f7;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
    }
    </style>
    <div class="title">Análisis Avanzado de Proveedores - Datamind</div>
    """,
    unsafe_allow_html=True
)

# Función para cargar y limpiar datos
@st.cache_data
def cargar_datos():
    url = "https://www.datos.gov.co/api/odata/v4/p6dx-8zbt"
    try:
        df_json = pd.read_json(url)
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None

    if 'value' in df_json.columns:
        df = pd.json_normalize(df_json['value'])
    else:
        st.error("No se encontró la columna 'value' en los datos")
        return None

    required_columns = [
        'precio_base', 'valor_total_adjudicacion', 'modalidad_de_contratacion',
        'nombre_del_proveedor', 'departamento_proveedor', 'fase',
        'estado_del_procedimiento', 'nombre_del_procedimiento'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.warning(f"Las siguientes columnas están ausentes: {', '.join(missing_columns)}")
        return None

    df.dropna(subset=required_columns, inplace=True)

    # Convertir valores a números
    def convertir_a_numero(valor):
        if isinstance(valor, str):
            valor = valor.replace('$', '').replace(',', '')
            try:
                return float(valor)
            except ValueError:
                return np.nan
        return valor

    df['precio_base'] = df['precio_base'].apply(convertir_a_numero)
    df['valor_total_adjudicacion'] = df['valor_total_adjudicacion'].apply(convertir_a_numero)

    return df

# Cargar los datos
df = cargar_datos()

if df is not None:
    # Sidebar para filtros interactivos
    st.sidebar.header("Filtros de Información")
    departamento_filtro = st.sidebar.selectbox(
        "Selecciona un Departamento", ['Sin seleccionar'] + list(df['departamento_proveedor'].unique())
    )
    modalidad_filtro = st.sidebar.selectbox(
        "Selecciona Modalidad de Contratación", ['Sin seleccionar'] + list(df['modalidad_de_contratacion'].unique())
    )
    proveedor_filtro = st.sidebar.text_input("Buscar por Nombre de Proveedor")
    nombre_procedimiento_filtro = st.sidebar.text_input("Buscar por Nombre del Procedimiento")
    fase_filtro = st.sidebar.selectbox(
        "Selecciona la Fase del Proceso", ['Sin seleccionar'] + list(df['fase'].unique())
    )
    estado_filtro = st.sidebar.selectbox(
        "Selecciona el Estado del Procedimiento", ['Sin seleccionar'] + list(df['estado_del_procedimiento'].unique())
    )

    # Aplicar los filtros
    df_filtrado = df
    if departamento_filtro != 'Sin seleccionar':
        df_filtrado = df_filtrado[df_filtrado['departamento_proveedor'] == departamento_filtro]
    if modalidad_filtro != 'Sin seleccionar':
        df_filtrado = df_filtrado[df_filtrado['modalidad_de_contratacion'] == modalidad_filtro]
    if proveedor_filtro:
        df_filtrado = df_filtrado[df_filtrado['nombre_del_proveedor'].str.contains(proveedor_filtro, case=False, na=False)]
    if nombre_procedimiento_filtro:
        df_filtrado = df_filtrado[df_filtrado['nombre_del_procedimiento'].str.contains(nombre_procedimiento_filtro, case=False, na=False)]
    if fase_filtro != 'Sin seleccionar':
        df_filtrado = df_filtrado[df_filtrado['fase'] == fase_filtro]
    if estado_filtro != 'Sin seleccionar':
        df_filtrado = df_filtrado[df_filtrado['estado_del_procedimiento'] == estado_filtro]

    # Visualización de datos filtrados
    st.write(f"### Datos Filtrados ({len(df_filtrado)} registros)")
    st.write(df_filtrado.head())

    # Gráficas
    st.write("### Distribución de Valor Total Adjudicado")
    fig = px.histogram(df_filtrado, x='valor_total_adjudicacion', nbins=30, title="Distribución de Valor Total Adjudicado")
    st.plotly_chart(fig)

    st.write("### Relación entre Precio Base y Valor Total Adjudicado")
    fig2 = px.scatter(df_filtrado, x='precio_base', y='valor_total_adjudicacion', title="Relación entre Precio Base y Valor Total Adjudicado")
    st.plotly_chart(fig2)

    # Correlación entre variables
    st.write("### Correlación entre variables")

    # Filtrar solo las columnas numéricas
    df_numerico = df_filtrado.select_dtypes(include=[np.number])

    # Verificar si hay columnas numéricas para evitar errores
    if not df_numerico.empty:
        fig3 = px.imshow(df_numerico.corr(), title="Correlación entre las variables", 
                     width=1200, height=800)  # Ajusta el tamaño del gráfico aquí
        st.plotly_chart(fig3)
    else:
        st.warning("No hay columnas numéricas disponibles para calcular la correlación.")


    # Cálculo de efectividad
    st.write("### Efectividad de Proveedores")
    df_filtrado['efectividad'] = df_filtrado['valor_total_adjudicacion'] / df_filtrado['precio_base']
    mejores_proveedores = df_filtrado.nlargest(10, 'efectividad')
    st.write(mejores_proveedores[['nombre_del_proveedor', 'efectividad']])

    # Predicción avanzada
    st.write("### Predicción Avanzada de Valor Total Adjudicado")
    df_regression = df_filtrado[['precio_base', 'valor_total_adjudicacion']].dropna()
    X = df_regression[['precio_base']]
    y = df_regression['valor_total_adjudicacion']

    # Modelos de predicción
    reg_lin = LinearRegression()
    reg_rf = RandomForestRegressor()

    # Entrenar modelos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    reg_lin.fit(X_train, y_train)
    reg_rf.fit(X_train, y_train)

    # Predicción para usuario
    futuro_input = st.number_input("Precio base futuro para predicción:", min_value=0, value=1000000)
    prediccion_lin = reg_lin.predict([[futuro_input]])[0]
    prediccion_rf = reg_rf.predict([[futuro_input]])[0]

    st.write(f"**Modelo Lineal**: El valor total adjudicado estimado es de ${prediccion_lin:,.2f}.")
    st.write(f"**Random Forest**: El valor total adjudicado estimado es de ${prediccion_rf:,.2f}.")

    # Recomendación basada en ponderación
    st.write("### Recomendación de Proveedores")
    st.sidebar.subheader("Criterios de Ponderación")
    peso_precio = st.sidebar.slider("Peso del Precio Base (%)", min_value=0, max_value=100, value=50)
    peso_efectividad = st.sidebar.slider("Peso de la Efectividad (%)", min_value=0, max_value=100, value=50)
    
    if peso_precio + peso_efectividad != 100:
        st.sidebar.warning("La suma de los pesos debe ser 100%.")
    else:
        df_filtrado['score'] = (
            peso_precio * (1 / df_filtrado['precio_base']) +
            peso_efectividad * df_filtrado['efectividad']
        )
        mejor_proveedor = df_filtrado.loc[df_filtrado['score'].idxmax()]
        st.write(f"El mejor proveedor recomendado es: **{mejor_proveedor['nombre_del_proveedor']}**")

    # Información sobre DataMind
    st.write("### Acerca de DataMind")
    st.markdown(
        """
        <div class="about-box">
        *DataMind* es un grupo de análisis e innovación especializado en la gestión de datos y modelos predictivos. 
        Nuestro objetivo es transformar grandes volúmenes de datos en insights valiosos para la toma de decisiones estratégicas.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="creditos">
        Creado por el equipo de DataMind. Todos los derechos reservados.
        </div>
        """, unsafe_allow_html=True)
