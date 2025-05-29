

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from itertools import product

df = pd.read_csv("mi_archivo (2).csv")

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

X = df_model[['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora', 'Saldo_vencido', 'Dias _de_atraso']]
y = df_model['Promesa_de_Pago']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

valores = {}
for col in ['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora']:
    valores[col] = label_encoders[col].transform(df_model_original[col].dropna().unique())

combinaciones = list(product(valores['Voz'], valores['Intensidad'], valores['Prompt'],
                             valores['Dia_Semana'], valores['Rango_Hora']))

def recomendar_para_cliente(cliente_id):
    cliente = df_model_original[df_model_original['ID Ally'] == cliente_id]

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

recomendaciones = []
sample_client_ids = df_model_original['ID Ally'].unique()[:11]

for client_id in sample_client_ids:
    rec = recomendar_para_cliente(client_id)
    recomendaciones.append(rec)

recomendaciones_df = pd.DataFrame(recomendaciones)
print(recomendaciones_df)

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from itertools import product

st.title("Recomendador de Interacciones con Clientes")

# Cargar datos
df = pd.read_csv("mi_archivo (2).csv")

df_model_original = df.copy()
df_model = df.dropna(subset=['Promesa_de_Pago'])

# Rellenar valores nulos
for col in ['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora']:
    df_model[col] = df_model[col].fillna("desconocido")

label_encoders = {}
for col in ['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora']:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

df_model['Saldo_vencido'] = df_model['Saldo_vencido'].fillna(0)
df_model['Dias _de_atraso'] = df_model['Dias _de_atraso'].fillna(0)

# Asegurarse de que los nombres de columna no tengan espacios accidentales
df_model.rename(columns=lambda x: x.strip(), inplace=True)
df_model_original.rename(columns=lambda x: x.strip(), inplace=True)

X = df_model[['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora', 'Saldo_vencido', 'Dias _de_atraso']]
y = df_model['Promesa_de_Pago']


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


valores = {}
for col in ['Voz', 'Intensidad', 'Prompt', 'Dia_Semana', 'Rango_Hora']:
    valores[col] = label_encoders[col].transform(df_model_original[col].dropna().unique())

combinaciones = list(product(valores['Voz'], valores['Intensidad'], valores['Prompt'],
                             valores['Dia_Semana'], valores['Rango_Hora']))

# Función principal de recomendación
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

# Interfaz de selección o entrada manual
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
