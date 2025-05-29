#Import the libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# With this we set the title of the Streamlit app
st.title("Simple Data Dashboard")
# To upload a CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# If a file is uploaded
if uploaded_file is not None:
    # To read the CSV into a DataFrame
    df = pd.read_csv(uploaded_file)
#With this line we can see few rows of the data
    st.subheader("Data Preview")
    st.write(df.head())
#This shows the  statistics summary
    st.subheader("Data Summary")
    st.write(df.describe())

#Filter the data by a selected column and value
    st.subheader("Filter Data")
    columns = df.columns.tolist()
    selected_column = st.selectbox("Select column to filter by", columns)
    unique_values = df[selected_column].unique()
    selected_value = st.selectbox("Select value", unique_values)

#On this we select the value
    filtered_df = df[df[selected_column] == selected_value]
    st.write(filtered_df)

#Select columns to plot on x and y axes
    st.subheader("Plot Data")
    x_column = st.selectbox("Select x-axis column", columns)
    y_column = st.selectbox("Select y-axis column", columns)

#This generate the chart from the filtered data
    if st.button("Generate Plot"):
        st.line_chart(filtered_df.set_index(x_column)[y_column])
else:
    with st.spinner("Waiting for file upload..."):
        st.empty()
   #  st.write("Waiting on file upload...")