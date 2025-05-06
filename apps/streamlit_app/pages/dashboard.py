import pandas as pd
import streamlit as st

# 화면
data = st.file_uploader("판정 결과 파일을 업로드해주세요.", type="csv")
if not data:
    st.error("판정 결과 파일을 업로드해주세요.")
    st.stop()

data = pd.read_csv(data)
st.dataframe(data)

# 그래프 추가하기
