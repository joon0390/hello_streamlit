import streamlit as st
import pandas as pd

data = st.file_uploader("upload file", type={"csv", "txt"})
if data is not None:
    df = pd.read_csv(data)
    df = df[::-1]
    df['number'] = df.index
    st.write(df)
    st.line_chart(df, x="number", y="Close")
