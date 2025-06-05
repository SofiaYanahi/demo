import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from itertools import product
from collections import defaultdict
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard de Llamadas", layout="wide", initial_sidebar_state="expanded")

seccion = st.sidebar.selectbox("Selecciona una sección:", ["Dashboard de Llamadas", "Recomendador de Interacciones con Clientes"])

try:
    df = pd.read_csv("mi_archivo (2).csv")
    df.columns = df.columns.str.strip()
    st.sidebar.success("Datos cargados correctamente.")
except Exception as e:
    st.sidebar.error(f"Error al cargar CSV: {e}")
    st.stop()

# Preparación general
categoricas = ['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora']
for col in categoricas:
    df[col] = df[col].fillna("desconocido")

# Filtramos registros donde Intensidad o Prompt no sean desconocido
df = df[(df['Intensidad'] != 'desconocido') & (df['Prompt'] != 'desconocido')]

# Dashboard de llamadas
if seccion == "Dashboard de Llamadas":
    st.title("Dashboard de Llamadas")

    df["Promesa_de_Pago"] = df["Promesa_de_Pago"].fillna(0)
    df["Llamada_Contestada"] = df["Llamada_Contestada"].fillna(0)
    df["Engagement"] = df["Llamada_Contestada"] + df["Promesa_de_Pago"]

    st.write("Primeras filas del DataFrame:")
    st.dataframe(df.head())

    dias = st.sidebar.multiselect("Selecciona Día(s) de la Semana:", df["Dia_Semana"].dropna().unique(), default=df["Dia_Semana"].dropna().unique())
    rangos = st.sidebar.multiselect("Selecciona Rango(s) Horario:", df["Rango_Hora"].dropna().unique(), default=df["Rango_Hora"].dropna().unique())

    df_filtrado = df[df["Dia_Semana"].isin(dias) & df["Rango_Hora"].isin(rangos)]

    resumen = df_filtrado.groupby(["Dia_Semana", "Rango_Hora"]).agg({"Promesa_de_Pago": "sum", "Llamada_Contestada": "sum", "ID Ally": "count"}).rename(columns={"ID Ally": "Total Llamadas"}).reset_index()
    resumen["Tasa Promesa"] = resumen["Promesa_de_Pago"] / resumen["Total Llamadas"]
    resumen["Tasa Contestación"] = resumen["Llamada_Contestada"] / resumen["Total Llamadas"]

    st.subheader("Tabla Resumen: Día de la Semana y Rango Horario")
    st.dataframe(resumen)

    st.subheader("Heatmap: Tasa de Promesa de Pago (filtrado)")
    pivot = resumen.pivot(index="Dia_Semana", columns="Rango_Hora", values="Tasa Promesa")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax1)
    st.pyplot(fig1)

    st.subheader("Conteo de llamadas por Día de la Semana y Rango Horario")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df_filtrado, x='Dia_Semana', hue='Rango_Hora', ax=ax2, palette="Blues", edgecolor='black')
    st.pyplot(fig2)

    conteo = df['Promesa_de_Pago'].value_counts().reset_index()
    conteo.columns = ['Promesa_de_Pago', 'Count']
    conteo['Promesa_de_Pago'] = conteo['Promesa_de_Pago'].map({1: 'Promesa de Pago', 0: 'Sin Promesa'})
    fig3 = px.pie(conteo, values='Count', names='Promesa_de_Pago', title='Distribución de Promesas de Pago', hole=0.3)
    fig3.update_traces(textinfo='percent+label')
    st.plotly_chart(fig3)

    st.subheader("Tasa de Promesa de Pago por Intensidad de Llamada")
    intensidad_group = df.groupby("Intensidad").agg(promesas=("Promesa_de_Pago", "sum"), llamadas=("ID Ally", "count")).reset_index()
    intensidad_group["Tasa Promesa"] = intensidad_group["promesas"] / intensidad_group["llamadas"]
    fig4, ax4 = plt.subplots()
    sns.barplot(data=intensidad_group, x="Intensidad", y="Tasa Promesa", ax=ax4)
    st.pyplot(fig4)

    st.subheader("Histograma: Días de Atraso")
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    ax5.hist(df['Dias_de_atraso'].dropna(), bins=20, color='steelblue', edgecolor='black')
    st.pyplot(fig5)

# Recomendador de Interacciones
elif seccion == "Recomendador de Interacciones con Clientes":
    st.title("Recomendador de Interacciones con Clientes")

    df_model = df.dropna(subset=['Promesa_de_Pago']).copy()
    label_encoders = {}
    for col in categoricas:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le

    df_model['Saldo_vencido'] = df_model['Saldo_vencido'].fillna(0)
    df_model['Dias_de_atraso'] = df_model['Dias_de_atraso'].fillna(0)

    X = df_model[categoricas + ['Saldo_vencido', 'Dias_de_atraso']]
    y = df_model['Promesa_de_Pago']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    cliente_ids = df['ID Ally'].dropna().astype(str).unique()
    opcion = st.radio("¿Cómo quieres ingresar el ID del cliente?", ["Seleccionar de lista", "Escribir manualmente"])
    cliente_id = st.selectbox("Selecciona un ID de cliente", cliente_ids) if opcion == "Seleccionar de lista" else st.text_input("Escribe el ID del cliente")

    def recomendar_para_cliente(cliente_id):
        cliente = df[df['ID Ally'].astype(str) == str(cliente_id)]
        if cliente.empty:
            return f"No se encontró cliente con ID: {cliente_id}"

        cliente = cliente.iloc[-1]
        saldo = cliente.get('Saldo_vencido', 0) or 0
        atraso = cliente.get('Dias_de_atraso', 0) or 0

        valores = {col: label_encoders[col].transform(label_encoders[col].classes_) for col in categoricas}
        combinaciones = list(product(*[valores[col] for col in categoricas]))

        df_combos = pd.DataFrame(combinaciones, columns=categoricas)
        df_combos['Saldo_vencido'] = saldo
        df_combos['Dias_de_atraso'] = atraso
        df_combos['prob'] = model.predict_proba(df_combos)[:, 1]
        mejor = df_combos.sort_values(by='prob', ascending=False).iloc[0]

        return {
            "Probabilidad estimada": round(mejor['prob'], 4),
            **{f"{col} recomendado": label_encoders[col].inverse_transform([int(mejor[col])])[0] for col in categoricas}
        }

    if cliente_id:
        st.subheader("Recomendación Individual")
        resultado = recomendar_para_cliente(cliente_id)
        if isinstance(resultado, dict):
            for k, v in resultado.items():
                st.write(f"**{k}**: {v}")
        else:
            st.warning(resultado)

        # Top 5 recomendaciones únicas
        st.subheader("Top 5 Recomendaciones Únicas")
        estrategias = list(product(
            df['Voz'].dropna().unique(),
            df['Intensidad'].dropna().unique(),
            df['Prompt'].dropna().unique(),
            df['Dia_Semana'].dropna().unique(),
            df['Rango_Hora'].dropna().unique()
        ))
        estrategias_df = pd.DataFrame(estrategias, columns=categoricas)
        estrategias_df = estrategias_df[(estrategias_df['Intensidad'] != 'desconocido') & (estrategias_df['Prompt'] != 'desconocido')]
        for col in categoricas:
            estrategias_df[col] = label_encoders[col].transform(estrategias_df[col])

        cliente = df[df['ID Ally'].astype(str) == str(cliente_id)].iloc[-1]
        estrategias_df['Saldo_vencido'] = cliente.get('Saldo_vencido', 0) or 0
        estrategias_df['Dias_de_atraso'] = cliente.get('Dias_de_atraso', 0) or 0
        estrategias_df['prob'] = model.predict_proba(estrategias_df[X_train.columns])[:, 1]
        estrategias_df = estrategias_df.sort_values(by='prob', ascending=False)

        usados = defaultdict(set)
        top5_unicos = []
        for _, fila in estrategias_df.iterrows():
            valido = all(fila[col] not in usados[col] for col in categoricas)
            if valido:
                top5_unicos.append(fila)
                for col in categoricas:
                    usados[col].add(fila[col])
            if len(top5_unicos) == 5:
                break

        df_top5 = pd.DataFrame(top5_unicos)
        st.dataframe(pd.DataFrame([
            {
                "Probabilidad estimada": round(fila['prob'], 4),
                **{f"{col} recomendado": label_encoders[col].inverse_transform([int(fila[col])])[0] for col in categoricas}
            }
            for _, fila in df_top5.iterrows()
        ]))
