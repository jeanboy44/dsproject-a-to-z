import streamlit as st

st.title("Input")
# Text Input
fname = st.text_input("Enter Firstname")
# Text Input Hide Password
password = st.text_input("Enter Password", type="password")
# Text Area
message = st.text_area("Enter Message", height=100)
# Numbers
number = st.number_input("Enter Number", 1.0, 25.0)
# Date Input
myappointment = st.date_input("Appointment")
# Time Input
mytime = st.time_input("My Time")
# Color Picker
color = st.color_picker("Select Color")

st.markdown("---")
st.title("Output")
st.write(fname)
st.write(password)
st.write(message)
st.write(number)
st.write(myappointment)
st.write(mytime)
st.write(color)
