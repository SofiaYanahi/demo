import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Simple Data Dashboard")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.write(df.head())

    st.subheader("Data Summary")
    st.write(df.describe())

    st.subheader("Filter Data")
    columns = df.columns.tolist()
    selected_column = st.selectbox("Select column to filter by", columns)
    unique_values = df[selected_column].unique()
    selected_value = st.selectbox("Select value", unique_values)

    filtered_df = df[df[selected_column] == selected_value]
    st.write(filtered_df)

    st.subheader("Plot Data")
    x_column = st.selectbox("Select x-axis column", columns)
    y_column = st.selectbox("Select y-axis column", columns)

if st.button("Generate Plot"):
    filtered_df = filtered_df.sort_values(by=x_column)

    st.markdown(f"### {y_column} vs {x_column}")
    st.line_chart(filtered_df.set_index(x_column)[y_column])

    st.markdown(f"**Promedio de {y_column}:** {filtered_df[y_column].mean():,.2f}")
    st.markdown(f"**Máximo de {y_column}:** {filtered_df[y_column].max():,.2f}")
    st.markdown(f"**Mínimo de {y_column}:** {filtered_df[y_column].min():,.2f}")
else:
    with st.spinner("Waiting for file upload..."):
        st.empty()
   #  st.write("Waiting on file upload...")