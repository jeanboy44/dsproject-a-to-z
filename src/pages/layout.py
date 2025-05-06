import streamlit as st

st.title("Layout")
tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])

with tab1:
    st.subheader("Tab 1")

with tab2:
    st.subheader("Tab 2")

st.markdown("---")

st.title("Columns")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Col 1")

with col2:
    st.subheader("Col 2")
st.markdown("---")
