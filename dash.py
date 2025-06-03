#dash
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from itertools import product

st.set_page_config(page_title="Dashboard de Llamadas", layout="wide", initial_sidebar_state="expanded")

# Sidebar para seleccionar sección
seccion = st.sidebar.selectbox("Selecciona una sección:", ["Dashboard de Llamadas", "Recomendador de Interacciones con Clientes"])

# Cargar archivos y limpiar nombres de columnas
def cargar_csv(nombre):
    try:
        df = pd.read_csv(nombre)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error al cargar {nombre}: {e}")
        return pd.DataFrame()

df_contacto_filtrado = cargar_csv("df_contacto_filtrado.csv")
df = cargar_csv("mi_archivo (2).csv")
Llamadas_filtradas = cargar_csv("Llamadas_filtradas.csv")
df_recomendaciones = cargar_csv("df_recomendaciones.csv")

if seccion == "Dashboard de Llamadas":
    st.title("Dashboard de Llamadas")

    if df.empty:
        st.warning("No se pudo cargar el archivo de llamadas.")
    else:
        st.write("Primeras filas del DataFrame:")
        st.dataframe(df.head())

        df["Promesa_de_Pago"] = df["Promesa_de_Pago"].fillna(0)
        df["Llamada_Contestada"] = df["Llamada_Contestada"].fillna(0)
        df["Engagement"] = df["Llamada_Contestada"] + df["Promesa_de_Pago"]

        # Tabla resumen
        grouped = df.groupby(["Dia_Semana", "Rango_Hora"]).agg({
            "Promesa_de_Pago": "sum",
            "Llamada_Contestada": "sum",
            "ID Ally": "count"
        }).rename(columns={"ID Ally": "Total Llamadas"}).reset_index()

        grouped["Tasa Promesa"] = grouped["Promesa_de_Pago"] / grouped["Total Llamadas"]
        grouped["Tasa Contestación"] = grouped["Llamada_Contestada"] / grouped["Total Llamadas"]

        st.subheader("Tabla Resumen: Día de la Semana y Rango Horario")
        st.dataframe(grouped.head(20))

        # Filtros
        st.sidebar.header("Filtros")
        dias = st.sidebar.multiselect("Selecciona Día(s) de la Semana:", options=df["Dia_Semana"].dropna().unique(), default=df["Dia_Semana"].dropna().unique())
        rangos = st.sidebar.multiselect("Selecciona Rango(s) Horario:", options=df["Rango_Hora"].dropna().unique(), default=df["Rango_Hora"].dropna().unique())

        df_filtrado = df[(df["Dia_Semana"].isin(dias)) & (df["Rango_Hora"].isin(rangos))]

        # HEATMAP
        st.subheader("Heatmap: Tasa de Promesa de Pago (filtrado)")
        grouped_filtrado = df_filtrado.groupby(["Dia_Semana", "Rango_Hora"]).agg({
            "Promesa_de_Pago": "sum",
            "ID Ally": "count"
        }).rename(columns={"ID Ally": "Total Llamadas"}).reset_index()

        grouped_filtrado["Tasa Promesa"] = grouped_filtrado["Promesa_de_Pago"] / grouped_filtrado["Total Llamadas"]
        pivot_promesa = grouped_filtrado.pivot(index="Dia_Semana", columns="Rango_Hora", values="Tasa Promesa")

        fig1, ax1 = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot_promesa, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax1)
        ax1.set_title("Tasa de Promesa de Pago por Día y Rango Horario")
        st.pyplot(fig1)

        # Conteo por día y rango
        st.subheader("Conteo de llamadas por Día y Rango Horario")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df_filtrado, x='Dia_Semana', hue='Rango_Hora', ax=ax2, palette="Blues", edgecolor='black')
        ax2.set_title('Conteo de llamadas')
        st.pyplot(fig2)

        # Pie chart
        conteo = df['Promesa_de_Pago'].value_counts().reset_index()
        conteo.columns = ['Promesa_de_Pago', 'Count']
        conteo['Promesa_de_Pago'] = conteo['Promesa_de_Pago'].map({1: 'Promesa de Pago', 0: 'Sin Promesa'})

        fig3 = px.pie(conteo, values='Count', names='Promesa_de_Pago', title='Distribución de Promesas de Pago', hole=0.3)
        fig3.update_traces(textinfo='percent+label')
        st.plotly_chart(fig3)

        # Tasa por intensidad
        st.subheader("Tasa de Promesa de Pago por Intensidad de Llamada")
        intensity_group = df.groupby("Intensidad").agg(
            promesas=("Promesa_de_Pago", "sum"),
            llamadas=("ID Ally", "count")
        ).reset_index()
        intensity_group["Tasa Promesa"] = intensity_group["promesas"] / intensity_group["llamadas"]

        fig4, ax4 = plt.subplots()
        sns.barplot(data=intensity_group, x="Intensidad", y="Tasa Promesa", ax=ax4)
        ax4.set_title("Tasa de Promesa por Intensidad")
        st.pyplot(fig4)

        # Histograma días de atraso
        st.subheader("Histograma: Días de Atraso")
        col_atraso = "Dias_de_atraso"
        if col_atraso in df.columns:
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            ax5.hist(df[col_atraso].dropna(), bins=20, color='steelblue', edgecolor='black')
            ax5.set_title('Frecuencia de Días de Atraso')
            ax5.set_xlabel('Días de Atraso')
            ax5.set_ylabel('Frecuencia')
            st.pyplot(fig5)
        else:
            st.warning(f"No se encontró la columna '{col_atraso}' en el DataFrame.")

elif seccion == "Recomendador de Interacciones con Clientes":
    st.title("Recomendador de Interacciones con Clientes")
    st.info("Esta sección aún está en construcción.")
