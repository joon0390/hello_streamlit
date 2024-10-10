import streamlit as st
import pandas as pd

data = st.file_uploader("upload file", type={"csv", "txt"})
if data is not None:
    df = pd.read_csv(data)
    df = df[::-1]
    df['time'] = df.index
    st.write(df)
    st.line_chart(df, x="time", y="Close")
