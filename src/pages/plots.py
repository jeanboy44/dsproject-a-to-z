import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.datasets import load_iris

st.title("Plots")
data = {
    "lang": [
        "BASIC",
        "COBOL",
        "Ruby",
        "Rust",
        "R",
        "Objective-C",
        "Nim",
        "C#",
        "Python",
        "JavaScript",
        "Java",
        "Julia",
        "Go",
    ],
    "Sum": [
        2456,
        616,
        3217,
        518,
        12020,
        813,
        260,
        21268,
        11352,
        11450,
        17355,
        260,
        1276,
    ],
}

df = pd.DataFrame(data)

st.markdown("---")
st.subheader("plotly")
fig = px.pie(df, values="Sum", names="lang", title="Pie Chart of Languages")
st.plotly_chart(fig)

fig2 = px.bar(df, x="lang", y="Sum")
st.plotly_chart(fig2)


st.markdown("---")
st.subheader("matplotlib(pyplot)")

df = load_iris(as_frame=True).data
df["species"] = load_iris().target_names[load_iris().target]
st.dataframe(df.head())


# Recommended Method
st.markdown("### 01. Recommended Method")
fig, ax = plt.subplots()
ax.scatter(*np.random.random(size=(2, 100)))
st.pyplot(fig)

# Method 3: Simple Method
st.markdown("### 02. Simple Method 1")
fig = plt.figure()
df["species"].value_counts().plot(kind="bar")
st.pyplot(fig)

# Method 3
st.markdown("### 03. Simple Method 2")
fig, ax = plt.subplots()
df["species"].value_counts().plot(kind="bar")
st.pyplot(fig)
