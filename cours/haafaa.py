import streamlit as st
import time
st.title("This is my first Streamlit app")
st.write("JUKK")

sex = st.selectbox("Choose",["F","M|"])
name = st.text_input("Please enter your name:")
age = st.slider("Select your age:", 0, 100, 25)
button = st.button("Welcome message!")
if button:
    while True:
        st.balloons()
        time.sleep(2)
    if name:
        st.write(f"Hello, {name}!")
    if age < 18:
        st.write("You are a minor.")
    elif age < 65:
        st.write("You are an adult.")
    else:        
        st.write("You are a senior.")

