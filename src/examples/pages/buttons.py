import streamlit as st

st.title("Butttons")
if st.button("Submit"):
    st.write("Submitted")

if st.button("Submit", key="new02"):
    st.write("Submitted")

# Working with RadioButtons
status = st.radio("What is your status", ("Active", "Inactive"))
if status == "Active":
    st.success("You are active")
elif status == "Inactive":
    st.warning("Inactive")

# Working with Checkbox
if st.checkbox("Show/Hide"):
    st.text("Showing something")

with st.expander("Julia"):
    st.text("hello Julia")

# Upload
st.file_uploader("Upload a file")

# Download
st.download_button("Download", "Hello")

# Link
st.link_button("Google", "https://www.google.com")
