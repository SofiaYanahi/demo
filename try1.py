import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from itertools import product
from collections import defaultdict
import plotly.graph_objects as go
import numpy as np


st.set_page_config(page_title="Dashboard de Llamadas", layout="wide", initial_sidebar_state="expanded")

seccion = st.sidebar.selectbox("Selecciona una sección:", ["Dashboard de Llamadas", "Sistema de recomendación"])


try:
    df = pd.read_csv("mi_archivo().csv")
    dfDias = df.copy()  
    dfCosto = df.copy()
    df.columns = df.columns.str.strip()  
    st.sidebar.success("Datos cargados correctamente.")
except Exception as e:
    st.sidebar.error(f"Error al cargar CSV: {e}")
    st.stop()

# Sección: Dashboard
if seccion == "Dashboard de Llamadas":
    st.title("Dashboard de Llamadas")

    df["Promesa_de_Pago"] = df["Promesa_de_Pago"].fillna(0)
    df["Llamada_Contestada"] = df["Llamada_Contestada"].fillna(0)
    df["Engagement"] = df["Llamada_Contestada"] + df["Promesa_de_Pago"]


    with st.expander("Ver primeras filas del DataFrame:"):
        st.dataframe(df.head(), use_container_width=True)


    with st.popover("Filtros"):
            dias_tabla= st.multiselect(
                "Dia(s):", df["Dia_Semana"].dropna().unique(), default=df["Dia_Semana"].dropna().unique(), key="dias_tabla"
        )
            rangos_tabla = st.multiselect(
            "Rango(s) horario:", df["Rango_Hora"].dropna().unique(), default=df["Rango_Hora"].dropna().unique(), key="rangos_tabla"
        )
    df_tabla = df[(df["Dia_Semana"].isin(dias_tabla)) & (df["Rango_Hora"].isin(rangos_tabla))]

    #TABLA RESUMEN
    st.markdown("Tabla Resumen: Día de la Semana y Rango Horario")
    with st.container():
        grouped_tabla = df_tabla.groupby(["Dia_Semana", "Rango_Hora"]).agg({
            "Promesa_de_Pago": "sum",
            "Llamada_Contestada": "sum",
            "ID Ally": "count"
        }).rename(columns={"ID Ally": "Total Llamadas"}).reset_index()

        grouped_tabla["Tasa Promesa"] = grouped_tabla["Promesa_de_Pago"] / grouped_tabla["Total Llamadas"]
        grouped_tabla_vista = grouped_tabla.drop(columns=["Llamada_Contestada", "Tasa Promesa"])
                                                         
        st.dataframe(grouped_tabla.head(20), use_container_width=True)
        st.info("Ejemplo:Lunes + Mañana → 21 llamadas, 4 promesas de pago (19% de efectividad), en todo el período los lunes por la mañana.")

    with st.popover("Filtros"):
        dias_comunes = st.multiselect(
            "Día(s):", df["Dia_Semana"].dropna().unique(),
            default=df["Dia_Semana"].dropna().unique(),
            key="dias_comunes"
        )
        rangos_comunes = st.multiselect(
            "Rango(s):", df["Rango_Hora"].dropna().unique(),
            default=df["Rango_Hora"].dropna().unique(),
            key="rangos_comunes"
        )

    df_filtros_tabs = df[
        (df["Dia_Semana"].isin(dias_comunes)) &
        (df["Rango_Hora"].isin(rangos_comunes))
    ]

    tab1, tab2 = st.tabs(["Heatmap", "Conteo por Día y Rango"])
       
    with tab1:
        st.markdown("### Heatmap: Tasa de Promesa de Pago (filtrado)")

        grouped_filtrado = df_filtros_tabs.groupby(["Dia_Semana", "Rango_Hora"]).agg({
            "Promesa_de_Pago": "sum",
            "ID Ally": "count"
        }).rename(columns={"ID Ally": "Total Llamadas"}).reset_index()

        grouped_filtrado["Tasa Promesa"] = grouped_filtrado["Promesa_de_Pago"] / grouped_filtrado["Total Llamadas"]
        pivot_promesa = grouped_filtrado.pivot(index="Dia_Semana", columns="Rango_Hora", values="Tasa Promesa")

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(pivot_promesa.values, cmap="YlGnBu", aspect="auto")

        # Etiquetas de ejes
        ax.set_xticks(np.arange(len(pivot_promesa.columns)))
        ax.set_xticklabels(pivot_promesa.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot_promesa.index)))
        ax.set_yticklabels(pivot_promesa.index)

        ax.set_title("Tasa de Promesa de Pago por Día de la Semana y Rango Horario (Filtrado)")
        ax.set_xlabel("Rango Horario")
        ax.set_ylabel("Día de la Semana")

        # Anotaciones en cada celda
        for i in range(len(pivot_promesa.index)):
            for j in range(len(pivot_promesa.columns)):
                val = pivot_promesa.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")

        fig.colorbar(im, ax=ax)
        st.pyplot(fig)

    with tab2:
        st.markdown("### Conteo de llamadas por Día de la Semana y Rango Horario")

        # Crear tabla resumen para el gráfico
        conteo = df_filtros_tabs.groupby(['Dia_Semana', 'Rango_Hora']).size().unstack(fill_value=0)

        # Orden de las categorías si se desea (ajusta según tus datos)
        if isinstance(conteo.index[0], str):
            dias_orden = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
            conteo = conteo.reindex(index=[d for d in dias_orden if d in conteo.index])

        # Gráfico
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        bottom = np.zeros(len(conteo))

        for rango in conteo.columns:
            valores = conteo[rango].values
            ax2.bar(conteo.index, valores, bottom=bottom, label=rango, edgecolor='black')
            bottom += valores

        ax2.set_title('Conteo de llamadas por Día de la Semana y Rango Horario (Filtrado)')
        ax2.set_xlabel('Día de la Semana')
        ax2.set_ylabel('Conteo de Llamadas')
        plt.xticks(rotation=45)
        ax2.legend(title="Rango Horario")

        st.pyplot(fig2)
        
    st.subheader("Data: Promesas de Pago")
    tab3, tab4 = st.tabs(["Distribución Promesas", "Tasa por Intensidad"])
    with tab3:
        conteo = df['Promesa_de_Pago'].value_counts().reset_index()
        conteo.columns = ['Promesa_de_Pago', 'Count']
        conteo['Promesa_de_Pago'] = conteo['Promesa_de_Pago'].map({1: 'Promesa de Pago', 0: 'Sin Promesa'})

        fig3 = px.pie(conteo, values='Count', names='Promesa_de_Pago', title='Distribución de Promesas de Pago', hole=0.3)
        fig3.update_traces(textinfo='percent+label')
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        intensity_group = df.groupby("Intensidad").agg(
            promesas=("Promesa_de_Pago", "sum"),
            llamadas=("ID Ally", "count")
        ).reset_index()

        intensity_group["Tasa Promesa"] = intensity_group["promesas"] / intensity_group["llamadas"]

        fig4 = px.bar(
            intensity_group,
            x="Intensidad",
            y="Tasa Promesa",
            text=intensity_group["Tasa Promesa"].apply(lambda x: f"{x:.2%}"),
            title="Tasa de Promesa de Pago por Intensidad de Llamada",
            labels={"Tasa Promesa": "Tasa de Promesa de Pago"}
        )
        fig4.update_traces(textposition='outside')
        fig4.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig4, use_container_width=True)

    #  HISTOGRAMA

    with st.container():
        st.subheader("Histograma: Días de Atraso con Tasa de Promesa de Pago")
        df['Dias_de_atraso_bin'] = pd.cut(df['Dias_de_atraso'], bins=20)
        grupo = df.groupby('Dias_de_atraso_bin').agg(
            total_llamadas=('Promesa_de_Pago', 'count'),
            promesas=('Promesa_de_Pago', 'sum')
        ).reset_index()

        grupo['tasa_promesa'] = grupo['promesas'] / grupo['total_llamadas']
        grupo['bin_centro'] = grupo['Dias_de_atraso_bin'].apply(lambda x: x.mid)
        grupo_filtrado = grupo[grupo['total_llamadas'] >= 10]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.hist(df['Dias_de_atraso'].dropna(), bins=20, color='steelblue', edgecolor='black', alpha=0.6)
        ax1.set_xlabel('Días de Atraso')
        ax1.set_ylabel('Frecuencia', color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')

        ax2 = ax1.twinx()
        ax2.plot(grupo_filtrado['bin_centro'], grupo_filtrado['tasa_promesa'], color='red', marker='o', label='Tasa de Promesa')
        ax2.set_ylabel('Tasa de Promesa de Pago', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 0.5)  # escalar para evitar picos falsos

        fig.suptitle('Frecuencia de Días de Atraso y Tasa de Promesa de Pago')
        fig.legend(loc='upper right')
        st.pyplot(fig)

        tasa_max = grupo_filtrado['tasa_promesa'].max()
        tasa_min = grupo_filtrado['tasa_promesa'].min()
    with st.popover("Interpretación"):
            st.markdown(
            f"""
            La tasa de promesa alcanza hasta **{tasa_max:.1%}** y puede caer a valores tan bajos como **{tasa_min:.1%}**.  
            Esto sugiere que cuanto más pronto se contacta al cliente, **mayor es la efectividad de la llamada**.
            """)

############### SECCIÓN 2: RECOMENDADOR DE INTERACCIONES ###############
elif seccion == "Sistema de recomendación":
    st.title("Sistema de recomendación")

    df.columns = df.columns.str.strip()
    df_model_original = df.copy()
    df_model = df.dropna(subset=['Promesa_de_Pago']).copy()

    for col in ['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora']:
        df_model[col] = df_model[col].fillna("desconocido")

    label_encoders = {}
    for col in ['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora']:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le

    df_model['Saldo_vencido'] = df_model['Saldo_vencido'].fillna(0)
    df_model['Dias_de_atraso'] = df_model['Dias_de_atraso'].fillna(0)

    X = df_model[['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora', 'Saldo_vencido', 'Dias_de_atraso']]
    y = df_model['Promesa_de_Pago']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # RECOMENDACIÓN POR CLIENTE 

    col1, col2 = st.columns([1, 2])

    with col1:
        opcion = st.radio("Tipo de entrada:", ["Seleccionar de lista", "Escribir manualmente"])

    with col2:
        cliente_ids = df_model_original['ID Ally'].dropna().astype(str).unique()
        if opcion == "Seleccionar de lista":
            cliente_id = st.selectbox("Selecciona un ID de cliente", cliente_ids)
        else:
            cliente_id = st.text_input("Escribe el ID del cliente")

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
                "*Probabilidad estimada*": round(mejor['prob'], 4),
                "*Día recomendado*": label_encoders['Dia_Semana'].inverse_transform([int(mejor['Dia_Semana'])])[0],
                "*Rango horario recomendado*": label_encoders['Rango_Hora'].inverse_transform([int(mejor['Rango_Hora'])])[0],
                "*Voz recomendada*": label_encoders['Voz'].inverse_transform([int(mejor['Voz'])])[0],
                "*Intensidad recomendada*": label_encoders['Intensidad'].inverse_transform([int(mejor['Intensidad'])])[0],
                "*Prompt recomendado*": label_encoders['Prompt'].inverse_transform([int(mejor['Prompt'])])[0]
            }

        resultado = recomendar_para_cliente(cliente_id)
        st.subheader(" Recomendación Individual")
        if isinstance(resultado, dict):
            for k, v in resultado.items():
                st.write(f"**{k}**: {v}")
        else:
            st.warning(resultado)

        # TOP 5 RECOMENDACIONES UNICAS
        st.subheader(" Top 5 Recomendaciones Únicas")
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
            atraso = cliente.get('Dias_de_atraso', 0)

            cliente_df = estrategias_df.copy()
            cliente_df['Saldo_vencido'] = saldo
            cliente_df['Dias_de_atraso'] = atraso
            cliente_df = cliente_df[X_train.columns]
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

    # BLOQUE: RECOMENDACIONES AGREGADAS
    from collections import defaultdict

    with st.expander(" Recomendaciones agregadas por día (niveladas con hasta 3 por cliente)"):
        estrategias = list(product(
            df_model_original['Voz'].dropna().unique(),
            df_model_original['Intensidad'].dropna().unique(),
            df_model_original['Prompt'].dropna().unique(),
            df_model_original['Dia_Semana'].dropna().unique(),
            df_model_original['Rango_Hora'].dropna().unique()
        ))
        estrategias_df = pd.DataFrame(estrategias, columns=['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora'])
        
        for col in estrategias_df.columns:
            estrategias_df[col] = label_encoders[col].transform(estrategias_df[col])

        recomendaciones_expandida = []

        for client_id in df_model_original['ID Ally'].dropna().unique():
            cliente = df_model_original[df_model_original['ID Ally'] == client_id].iloc[-1]
            saldo = cliente.get('Saldo_vencido', 0)
            atraso = cliente.get('Dias_de_atraso', 0)

            cliente_df = estrategias_df.copy()
            cliente_df['Saldo_vencido'] = saldo
            cliente_df['Dias_de_atraso'] = atraso
            cliente_df = cliente_df[X_train.columns]
            cliente_df['probabilidad'] = model.predict_proba(cliente_df)[:, 1]
            cliente_df['ID Ally'] = client_id

            top5 = cliente_df.sort_values(by='probabilidad', ascending=False).head(5)

            # Tomar hasta 3 del top 5
            recomendaciones_expandida.extend(top5.head(3).to_dict('records'))

        df_recomendaciones = pd.DataFrame(recomendaciones_expandida)

        # Invertir codificación
        for col in ['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora']:
            df_recomendaciones[col] = label_encoders[col].inverse_transform(df_recomendaciones[col].astype(int))

        # Contar por día
        llamadas_por_dia = df_recomendaciones['Dia_Semana'].value_counts().sort_index()

        st.subheader(" Cantidad de llamadas niveladas por día")
        st.bar_chart(llamadas_por_dia)

        # Mostrar tabla de IDs por día
        ids_por_dia = df_recomendaciones.groupby('Dia_Semana')['ID Ally'].apply(list).to_dict()
        tabla_ids = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in ids_por_dia.items()]))
        st.subheader(" Tabla de IDs por día")
        st.dataframe(tabla_ids)

            # Nivelar exactamente la misma cantidad por día
        llamadas_por_dia = df_recomendaciones['Dia_Semana'].value_counts()
        minimo_por_dia = llamadas_por_dia.min()

        # Muestreo nivelado
        df_nivelado = (
            df_recomendaciones.groupby('Dia_Semana')
            .apply(lambda x: x.sample(minimo_por_dia, random_state=42))
            .reset_index(drop=True)
        )

        st.subheader("Llamadas niveladas: mismo número por día")
        st.bar_chart(df_nivelado['Dia_Semana'].value_counts().sort_index())

        ids_nivelados = df_nivelado.groupby('Dia_Semana')['ID Ally'].apply(list).to_dict()
        tabla_ids_nivelada = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in ids_nivelados.items()]))
        st.dataframe(tabla_ids_nivelada)



    # BLOQUE: EFECTIVIDAD POR ALIADA 
    with st.expander(" Efectividad de llamadas por aliada"):
        try:
            df_contacto_filtrado = pd.read_csv("df_contacto_filtrado.csv")
        except Exception as e:
            st.error(f"Error al cargar df_contacto_filtrado.csv: {e}.")
            st.stop()

        df_contacto_filtrado.rename(columns={"ID Ally": "ID Aliada"}, inplace=True)
        df_contacto_filtrado.columns = df_contacto_filtrado.columns.str.strip()
        df_contacto_filtrado["Promesa_de_Pago"] = df_contacto_filtrado["Tipificaciones"].apply(lambda x: 1 if x == "1. Promesa de pago" else 0)
        df_contacto_filtrado["Ya_Pagaron"] = df_contacto_filtrado["Tipificaciones"].apply(lambda x: 1 if x == "4. Cliente Ya Pago" else 0)
        df_contacto_filtrado["Buzon"] = df_contacto_filtrado["Tipificaciones"].apply(lambda x: 1 if x == "2. Usuario colgo la llamada" else 0)

        df_llamadas_por_aliada = df_contacto_filtrado.groupby("ID Aliada").agg(
            Llamadas=('Tipificaciones', 'count'),
            Promesas=('Promesa_de_Pago', 'sum'),
            Ya_Pagaron=('Ya_Pagaron', 'sum'),
            Buzon=('Buzon', 'sum')
        ).reset_index()

        df_agrupado_por_llamadas = df_llamadas_por_aliada.groupby("Llamadas").agg(
            Promesas=('Promesas', 'sum'),
            Ya_Pagaron=('Ya_Pagaron', 'sum'),
            Buzon=('Buzon', 'sum')
        ).sort_index()

        min_llamadas = int(df_agrupado_por_llamadas.index.min())
        max_llamadas = int(df_agrupado_por_llamadas.index.max())
        rango = st.slider("Selecciona el rango de llamadas por aliada:", min_llamadas, max_llamadas, (min_llamadas, max_llamadas))
        df_filtrado_efectividad = df_agrupado_por_llamadas.loc[rango[0]:rango[1]]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_filtrado_efectividad.index, y=df_filtrado_efectividad["Promesas"], name="Promesa de pago", marker_color="#0B1793"))
        fig.add_trace(go.Bar(x=df_filtrado_efectividad.index, y=df_filtrado_efectividad["Ya_Pagaron"], name="Ya pagó", marker_color="#577BD5"))
        fig.add_trace(go.Bar(x=df_filtrado_efectividad.index, y=df_filtrado_efectividad["Buzon"], name="Buzón", marker_color="#479AFF"))

        fig.update_layout(
            barmode='stack',
            title="Efectividad de llamadas agrupada por número de llamadas por aliada",
            xaxis_title="Número de llamadas hechas por aliada",
            yaxis_title="Cantidad total por resultado",
            legend_title="Resultado",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander(" Análisis de Costo por Llamada"):
        try:
            df_llamadas_costo = pd.read_csv("df_llamadas_costo.csv")
            df_llamadas_costo.columns = df_llamadas_costo.columns.str.strip()
        except Exception as e:
            st.error(f"Error al cargar df_llamadas_costo.csv: {e}.")
            st.stop()

        # Calcular el costo por llamada según la tipificación
        def costo_por_llamada_fijo(row):
            if row["Tipificaciones"] == "3. Contacto no definido":
                return 0.024
            else:
                return 0.049

        df_llamadas_costo["Costo por llamada"] = df_llamadas_costo.apply(costo_por_llamada_fijo, axis=1)
        df_llamadas_costo["Costo Total"] = df_llamadas_costo["Costo por llamada"]
        df_llamadas_costo["Buzon"] = df_llamadas_costo["Tipificaciones"].apply(
            lambda x: 1 if x == "3. Contacto no definido" else 0
        )
        df_llamadas_costo["Otras"] = df_llamadas_costo["Tipificaciones"].apply(
            lambda x: 1 if x not in ["3. Contacto no definido"] else 0
        )                        
        # Agrupar por ID Ally y Campo 2
        detalle_por_aliada = df_llamadas_costo.groupby(["ID Ally", "Campo 2"]).agg({
            "Minutos": "sum",
            "Costo Total": "sum",
            "Tipificaciones": "count",
            "Buzon": "sum",
            "Otras": "sum"
        }).rename(columns={"Tipificaciones": "Número de llamadas"})

        # Agrupar por Campo 2
        resumen_por_campo2 = df_llamadas_costo.groupby("Campo 2").agg(
            Costo_Total=('Costo Total', 'sum'),
            Numero_llamadas=('ID Ally', 'count')
        ).reset_index()

        # Mostrar ambas tablas con pestañas
        tab1, tab2 = st.tabs(["Detalle por aliada", "Resumen por Campo 2"])

        with tab1:
            st.dataframe(detalle_por_aliada.style.format({
                "Costo_Total": "${:,.2f}",
                "Minutos": "{:.2f}",
                "Numero_llamadas": "{:,}"
            }), use_container_width=True)

        with tab2:
            st.dataframe(resumen_por_campo2.style.format({
                "Costo_Total": "${:,.2f}",
                "Numero_llamadas": "{:,}"
            }), use_container_width=True)