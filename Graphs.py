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

seccion = st.sidebar.selectbox("Selecciona una sección:", ["Dashboard de Llamadas", "Recomendador de Interacciones con Clientes"])


try:
    df = pd.read_csv("mi_archivo (2).csv")
    dfDias = df.copy()  # Copia del DataFrame original para evitar problemas de referencia
    df.columns = df.columns.str.strip()  # <--- SOLUCIÓN CLAVE
    st.sidebar.success("Datos cargados correctamente.")
except Exception as e:
    st.sidebar.error(f"Error al cargar CSV: {e}")
    st.stop()

# Sección: Dashboard
if seccion == "Dashboard de Llamadas":
    st.title("Dashboard de Llamadas")

    st.write("Primeras filas del DataFrame:")
    st.dataframe(df.head())


    df["Promesa_de_Pago"] = df["Promesa_de_Pago"].fillna(0)
    df["Llamada_Contestada"] = df["Llamada_Contestada"].fillna(0)
    df["Engagement"] = df["Llamada_Contestada"] + df["Promesa_de_Pago"]

    #TABLA RESUMEN 
    grouped = df.groupby(["Dia_Semana", "Rango_Hora"]).agg({
        "Promesa_de_Pago": "sum",
        "Llamada_Contestada": "sum",
        "ID Ally": "count"
    }).rename(columns={"ID Ally": "Total Llamadas"}).reset_index()

    grouped["Tasa Promesa"] = grouped["Promesa_de_Pago"] / grouped["Total Llamadas"]
    grouped["Tasa Contestación"] = grouped["Llamada_Contestada"] / grouped["Total Llamadas"]

    st.subheader("Tabla Resumen: Día de la Semana y Rango Horario")
    st.dataframe(grouped.head(20))
    st.text("Lunes + Mañana = En todo el período se hicieron 21 llamadas los lunes por la mañana, de las cuales 4 prometieron pagar (19% de efectividad). Esto te indica que los lunes por la mañana son un buen horario para llamar comparado con otros momentos.")

    #FILTROS 
    st.sidebar.header("Filtros")

    dias = st.sidebar.multiselect(
        "Selecciona Día(s) de la Semana:",
        options=df["Dia_Semana"].dropna().unique(),
        default=df["Dia_Semana"].dropna().unique()
    )

    rangos = st.sidebar.multiselect(
        "Selecciona Rango(s) Horario:",
        options=df["Rango_Hora"].dropna().unique(),
        default=df["Rango_Hora"].dropna().unique()
    )

    #DataFrame filtrado solo para gráficos específicos
    df_filtrado = df[
        (df["Dia_Semana"].isin(dias)) &
        (df["Rango_Hora"].isin(rangos))
    ]

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
    ax1.set_title("Tasa de Promesa de Pago por Día de la Semana y Rango Horario (Filtrado)")
    ax1.set_xlabel("Rango Horario")
    ax1.set_ylabel("Día de la Semana")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    #Conteo por Día y Rango
    st.subheader("Conteo de llamadas por Día de la Semana y Rango Horario")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df_filtrado, x='Dia_Semana', hue='Rango_Hora', ax=ax2, palette="Blues", edgecolor='black')
    ax2.set_title('Conteo de llamadas por Día de la Semana y Rango Horario (Filtrado)')
    ax2.set_xlabel('Día de la Semana')
    ax2.set_ylabel('Conteo de Llamadas')
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    #PIE CHART
    conteo = df['Promesa_de_Pago'].value_counts().reset_index()
    conteo.columns = ['Promesa_de_Pago', 'Count']
    conteo['Promesa_de_Pago'] = conteo['Promesa_de_Pago'].map({1: 'Promesa de Pago', 0: 'Sin Promesa'})

    fig3 = px.pie(
        conteo,
        values='Count',
        names='Promesa_de_Pago',
        title='Distribución de Promesas de Pago',
        hole=0.3
    )
    fig3.update_traces(textinfo='percent+label')
    st.plotly_chart(fig3)

    # TASA por INTENSIDAD 
    st.subheader("Tasa de Promesa de Pago por Intensidad de Llamada")

    intensity_group = df.groupby("Intensidad").agg(
        promesas=("Promesa_de_Pago", "sum"),
        llamadas=("ID Ally", "count")
    ).reset_index()

    intensity_group["Tasa Promesa"] = intensity_group["promesas"] / intensity_group["llamadas"]

    fig4, ax4 = plt.subplots()
    sns.barplot(data=intensity_group, x="Intensidad", y="Tasa Promesa", ax=ax4)
    ax4.set_title("Tasa de Promesa de Pago por Intensidad de Llamada")
    ax4.set_ylabel("Tasa de Promesa")
    st.pyplot(fig4)

    #  HISTOGRAMAS
    st.subheader("Histograma: Días de Atraso")
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    ax5.hist(dfDias['Dias_de_atraso'].dropna(), bins=20, color='steelblue', edgecolor='black')
    ax5.set_title('Frecuencia de Días de Atraso')
    ax5.set_xlabel('Días de Atraso')
    ax5.set_ylabel('Frecuencia')
    st.pyplot(fig5)
    
elif seccion == "Recomendador de Interacciones con Clientes":
    st.title("Recomendador de Interacciones con Clientes")

    df_model_original = df.copy()
    df_model = df.dropna(subset=['Promesa_de_Pago']).copy()
    df_model.columns = df_model.columns.str.strip()
    df_model_original.columns = df_model_original.columns.str.strip()

    for col in ['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora']:
        df_model[col] = df_model[col].fillna("desconocido")

    label_encoders = {}
    for col in ['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora']:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le

    df_model['Saldo_vencido'] = df_model['Saldo_vencido'].fillna(0)
    df_model['Dias_de_atraso'] = dfDias['Dias_de_atraso'].fillna(0)
    df_model.rename(columns=lambda x: x.strip(), inplace=True)
    df_model_original.rename(columns=lambda x: x.strip(), inplace=True)

    X = df_model[['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora', 'Saldo_vencido', 'Dias_de_atraso']]
    y = df_model['Promesa_de_Pago']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Selección de cliente
    cliente_ids = df_model_original['ID Ally'].dropna().astype(str).unique()
    opcion = st.radio("¿Cómo quieres ingresar el ID del cliente?", ["Seleccionar de lista", "Escribir manualmente"])

    if opcion == "Seleccionar de lista":
        cliente_id = st.selectbox("Selecciona un ID de cliente", cliente_ids)
    else:
        cliente_id = st.text_input("Escribe el ID del cliente")

    # Recomendación individual
    if cliente_id:
        def recomendar_para_cliente(cliente_id):
            cliente = df_model_original[df_model_original['ID Ally'].astype(str) == str(cliente_id)]
            if cliente.empty:
                return f"No se encontró cliente con ID: {cliente_id}"

            cliente = cliente.iloc[-1]
            saldo = cliente['Saldo_vencido'] if pd.notnull(cliente['Saldo_vencido']) else 0
            atraso = cliente['Dias_de_atraso'] if pd.notnull(cliente['Dias_de_atraso']) else 0

            valores = {col: label_encoders[col].transform(df_model_original[col].dropna().unique()) for col in label_encoders}
            combinaciones = list(product(valores['Voz'], valores['Intensidad'], valores['Prompt'], valores['Dia_Semana'], valores['Rango_Hora']))

            df_combos = pd.DataFrame(combinaciones, columns=['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora'])
            df_combos['Saldo_vencido'] = saldo
            df_combos['Dias_de_atraso'] = atraso

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

        st.subheader("📌 Recomendación Individual")
        resultado = recomendar_para_cliente(cliente_id)
        if isinstance(resultado, dict):
            for k, v in resultado.items():
                st.write(f"**{k}**: {v}")
        else:
            st.warning(resultado)

        # Top 5 recomendaciones únicas para el cliente seleccionado
        st.subheader("Top 5 Recomendaciones Únicas")

        from collections import defaultdict

        estrategias = [
            (voz, intensidad, prompt, dia, hora)
            for voz in df_model_original['Voz'].dropna().unique()
            for intensidad in df_model_original['Intensidad'].dropna().unique()
            for prompt in df_model_original['Prompt'].dropna().unique()
            for dia in df_model_original['Dia_Semana'].dropna().unique()
            for hora in df_model_original['Rango_Hora'].dropna().unique()
        ]

        estrategias_df = pd.DataFrame(estrategias, columns=['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora'])
        for col in estrategias_df.columns:
            estrategias_df[col] = label_encoders[col].transform(estrategias_df[col])

        cliente = df_model_original[df_model_original['ID Ally'].astype(str) == str(cliente_id)]
        if not cliente.empty:
            cliente = cliente.iloc[-1]
            saldo = cliente.get('Saldo_vencido', 0)
            atraso = cliente.get('Dias _de_atraso', 0)

            cliente_df = estrategias_df.copy()
            cliente_df['Saldo_vencido'] = saldo
            cliente_df['Dias_de_atraso'] = atraso
            cliente_df['probabilidad'] = model.predict_proba(cliente_df)[:, 1]
            cliente_df_sorted = cliente_df.sort_values(by='probabilidad', ascending=False).copy()

            usados = defaultdict(set)
            top5_unicos = []

            for _, fila in cliente_df_sorted.iterrows():
                valido = all(fila[col] not in usados[col] for col in ['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora'])
                if valido:
                    top5_unicos.append(fila)
                    for col in ['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora']:
                        usados[col].add(fila[col])
                if len(top5_unicos) == 5:
                    break

            if len(top5_unicos) < 5:
                faltantes = 5 - len(top5_unicos)
                usados_indices = [fila.name for fila in top5_unicos]
                adicionales = cliente_df_sorted.drop(index=usados_indices).head(faltantes)
                top5_unicos.extend(adicionales.itertuples(index=False))

            recomendaciones_unicas = []
            for fila in top5_unicos:
                recomendaciones_unicas.append({
                    'Probabilidad estimada': round(fila.probabilidad, 4),
                    'Día recomendado': label_encoders['Dia_Semana'].inverse_transform([int(fila.Dia_Semana)])[0],
                    'Rango horario recomendado': label_encoders['Rango_Hora'].inverse_transform([int(fila.Rango_Hora)])[0],
                    'Voz recomendada': label_encoders['Voz'].inverse_transform([int(fila.Voz)])[0],
                    'Intensidad recomendada': label_encoders['Intensidad'].inverse_transform([int(fila.Intensidad)])[0],
                    'Prompt recomendado': label_encoders['Prompt'].inverse_transform([int(fila.Prompt)])[0]
                })

            df_recomendaciones_unicas = pd.DataFrame(recomendaciones_unicas)
            st.dataframe(df_recomendaciones_unicas)

    else:
        st.info("Por favor selecciona o ingresa un ID de cliente para ver recomendaciones.")
# Tabla de recomendaciones generales --- ###
        