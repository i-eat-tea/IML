import streamlit as st
st.title("This is my first Streamlit app")
name = st.text_input("Please enter your name:")
age = st.slider("Select your age:", 0, 100, 25)
if name:
    st.write(f"Hello, {name}!")
    if age < 18:
        st.write("You are a minor.")
    elif age < 65:
        st.write("You are an adult.")
    else:        
        st.write("You are a senior.")
