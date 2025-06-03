

import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from itertools import product

# Configurar página antes de cualquier otro componente
st.set_page_config(page_title="Dashboard de Llamadas", layout="wide", initial_sidebar_state="expanded")

# Sidebar para seleccionar sección
seccion = st.sidebar.selectbox("Selecciona una sección:", ["Dashboard de Llamadas", "Recomendador de Interacciones con Clientes"])

# Cargar datos
try:
    df = pd.read_csv("mi_archivo (2).csv")
    st.sidebar.success("Datos cargados correctamente.")
except Exception as e:
    st.sidebar.error(f"Error al cargar CSV: {e}")
    st.stop()

# Preprocesamiento general
for col in ["Promesa_de_Pago", "Llamada_Contestada", "Dias _de_atraso"]:
    df[col] = df[col].fillna(0)
    df["Engagement"] = df["Llamada_Contestada"] + df["Promesa_de_Pago"]

# Filtros en sidebar
st.sidebar.header("Filtros")
dias_filtrados = st.sidebar.multiselect("Filtrar por Día de la Semana", options=sorted(df["Dia_Semana"].dropna().unique()), default=sorted(df["Dia_Semana"].dropna().unique()))
rango_horas_filtradas = st.sidebar.multiselect("Filtrar por Rango Horario", options=sorted(df["Rango_Hora"].dropna().unique()), default=sorted(df["Rango_Hora"].dropna().unique()))
intensidades_filtradas = st.sidebar.multiselect("Filtrar por Intensidad", options=sorted(df["Intensidad"].dropna().unique()), default=sorted(df["Intensidad"].dropna().unique()))

# Aplicar filtros
df_filtrado = df[
    (df["Dia_Semana"].isin(dias_filtrados)) &
    (df["Rango_Hora"].isin(rango_horas_filtradas)) &
    (df["Intensidad"].isin(intensidades_filtradas))
]

# Sección: Dashboard
if seccion == "Dashboard de Llamadas":
    st.title("Dashboard de Llamadas")

    st.subheader("Tabla Resumen: Día de la Semana y Rango Horario")
    grouped = df_filtrado.groupby(["Dia_Semana", "Rango_Hora", "ID Ally"]).agg({
        "Promesa_de_Pago": "sum",
        "Llamada_Contestada": "sum"
    }).reset_index()

    resumen = grouped.groupby(["Dia_Semana", "Rango_Hora"]).agg({
        "Promesa_de_Pago": "sum",
        "Llamada_Contestada": "sum",
        "ID Ally": "count"
    }).rename(columns={"ID Ally": "Total Llamadas"}).reset_index()

    resumen["Tasa Promesa"] = resumen["Promesa_de_Pago"] / resumen["Total Llamadas"]
    resumen["Tasa Contestación"] = resumen["Llamada_Contestada"] / resumen["Total Llamadas"]

    st.dataframe(resumen)

    st.subheader("Heatmap: Tasa de Promesa de Pago")
    pivot_promesa = resumen.pivot(index="Dia_Semana", columns="Rango_Hora", values="Tasa Promesa")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot_promesa, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax1)
    ax1.set_title("Tasa de Promesa de Pago por Día y Rango Horario")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    st.subheader("Tasa de Promesa por Intensidad")
    intensity_group = df_filtrado.groupby("Intensidad").agg(
        promesas=("Promesa_de_Pago", "sum"),
        llamadas=("ID Ally", "count")
    ).reset_index()
    intensity_group["Tasa Promesa"] = intensity_group["promesas"] / intensity_group["llamadas"]

    fig2, ax2 = plt.subplots()
    sns.barplot(data=intensity_group, x="Intensidad", y="Tasa Promesa", ax=ax2)
    ax2.set_title("Tasa de Promesa de Pago por Intensidad")
    st.pyplot(fig2)

    st.subheader("Distribución de Promesas de Pago")
    conteo = df_filtrado['Promesa_de_Pago'].value_counts().reset_index()
    conteo.columns = ['Promesa_de_Pago', 'Count']
    conteo['Promesa_de_Pago'] = conteo['Promesa_de_Pago'].map({1: 'Promesa de Pago', 0: 'Sin Promesa'})

    fig3 = px.pie(conteo, values='Count', names='Promesa_de_Pago', title='Distribución de Promesas de Pago', hole=0.3)
    fig3.update_traces(textinfo='percent+label')
    st.plotly_chart(fig3)

    st.subheader("Conteo de Llamadas por Día y Rango Horario")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df_filtrado, x='Dia_Semana', hue='Rango_Hora', ax=ax4)
    ax4.set_title('Conteo de llamadas por Día de la Semana y Rango Horario')
    plt.xticks(rotation=45)
    st.pyplot(fig4)

    st.subheader("Histograma de Días de Atraso")
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    ax5.hist(df_filtrado['Dias _de_atraso'], bins=20, color='steelblue', edgecolor='black')
    ax5.set_title('Frecuencia de Días de Atraso')
    st.pyplot(fig5)

    st.subheader("Histograma de Promesas de Pago")
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    ax6.hist(df_filtrado['Promesa_de_Pago'], color='steelblue', edgecolor='black')
    ax6.set_title('Frecuencia de Promesas de Pago')
    st.pyplot(fig6)

# Sección: Recomendador





elif seccion == "Recomendador de Interacciones con Clientes":
    st.title("Recomendador de Interacciones con Clientes")

    df_recomendaciones = pd.read_csv("df_recomendaciones.csv")
    df_model_original = df.copy()
    df_model = df.dropna(subset=['Promesa_de_Pago'])

    for col in ['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora']:
        df_model[col] = df_model[col].fillna("desconocido")

    label_encoders = {}
    for col in ['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora']:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le

    df_model['Saldo_vencido'] = df_model['Saldo_vencido'].fillna(0)
    df_model['Dias _de_atraso'] = df_model['Dias _de_atraso'].fillna(0)
    df_model.rename(columns=lambda x: x.strip(), inplace=True)
    df_model_original.rename(columns=lambda x: x.strip(), inplace=True)

    X = df_model[['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora', 'Saldo_vencido', 'Dias _de_atraso']]
    y = df_model['Promesa_de_Pago']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    valores = {col: label_encoders[col].transform(df_model_original[col].dropna().unique()) for col in label_encoders}
    combinaciones = list(product(valores['Voz'], valores['Intensidad'], valores['Prompt'], valores['Dia_Semana'], valores['Rango_Hora']))

    def recomendar_para_cliente(cliente_id):
        cliente = df_model_original[df_model_original['ID Ally'].astype(str) == str(cliente_id)]
        if cliente.empty:
            return f"No se encontró cliente con ID: {cliente_id}"

        cliente = cliente.iloc[-1]
        saldo = cliente['Saldo_vencido'] if pd.notnull(cliente['Saldo_vencido']) else 0
        atraso = cliente['Dias _de_atraso'] if pd.notnull(cliente['Dias _de_atraso']) else 0

        df_combos = pd.DataFrame(combinaciones, columns=['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora'])
        df_combos['Saldo_vencido'] = saldo
        df_combos['Dias _de_atraso'] = atraso

        proba = model.predict_proba(df_combos)[:, 1]
        df_combos['prob'] = proba
        mejor = df_combos.sort_values(by='prob', ascending=False).iloc[0]

        return {
            "Probabilidad estimada": round(mejor['prob'], 4),
            "Día recomendado": label_encoders['Dia_Semana'].inverse_transform([int(mejor['Dia_Semana'])])[0],
            "Rango horario recomendado": label_encoders['Rango_Hora'].inverse_transform([int(mejor['Rango_Hora'])])[0],
            "Voz recomendada": label_encoders['Voz'].inverse_transform([int(mejor['Voz'])])[0],
            "Intensidad recomendada": label_encoders['Intensidad'].inverse_transform([int(mejor['Intensidad'])])[0],
            "Prompt recomendado": label_encoders['Prompt'].inverse_transform([int(mejor['Prompt'])])[0]
        }

    cliente_ids = df_model_original['ID Ally'].dropna().astype(str).unique()
    opcion = st.radio("¿Cómo quieres ingresar el ID del cliente?", ["Seleccionar de lista", "Escribir manualmente"])

    if opcion == "Seleccionar de lista":
        cliente_id = st.selectbox("Selecciona un ID de cliente", cliente_ids)
    else:
        cliente_id = st.text_input("Escribe el ID del cliente")

    if st.button("Recomendar interacción"):
        if cliente_id:
            resultado = recomendar_para_cliente(cliente_id)
            if isinstance(resultado, dict):
                st.success("Recomendación generada:")
                for k, v in resultado.items():
                    st.write(f"**{k}**: {v}")
            else:
                st.warning(resultado)
        else:
            st.error("Por favor ingresa o selecciona un ID válido.")

    
    llamadas_por_dia = df_recomendaciones['dia_recomendado'].value_counts().sort_index()

    ids_por_dia = df_recomendaciones.groupby('dia_recomendado')['id_ally'].apply(list).reset_index()

    print("Cantidad de llamadas por día:")
    print(llamadas_por_dia)

    print("\n IDs de clientes por día:")
    print(ids_por_dia)