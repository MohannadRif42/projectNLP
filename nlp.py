from typing import Container
import streamlit as st
import joblib,os
import base64
from streamlit_option_menu import option_menu
import pickle

with open('model_pickle','rb') as f:
    mod = pickle.load(f)

import pickle

with open('vectorizer.pk','rb') as fin:
    vec = pickle.load(fin)

def get_base64_of_bin_file(bin_file):
    """
    function to read png file
    ----------
    bin_file: png -> the background image in local folder
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('cs11.png')  



def main():
    with open('style.css') as f:

        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    st.title("Complaints Classifer NLP App")
    st.markdown("---")

    selected = option_menu(

        menu_title="NLP",
        options=["About the Project","Predict"],
        icons=["book","gear"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
        "container": {"padding": "5!important", "background-color": "#333236"},
        "icon": {"color": "#0e1047", "font-size": "20px"}, 
        "nav-link": {"font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": "#6e2951"},
        "nav-link-selected": {"background-color": "#804065"},
            }
    )
    
    if selected=="Predict":

        container = st.container()
        with container:
            new_text = st.text_area("Enter Complaint","Type Here")
            if st.button("Classify"):

                vec_text = vec.transform([new_text]).toarray()
                prediction = mod.predict(vec_text)
                st.write(prediction)
                st.balloons()

##########

    if selected=="About the Project":

        st.subheader("The main aim from this project is to leverage the power of machine learning to make our lives easier. Instead of manually having to sort the customer comments, the model does it now for us. This not only saves us time, it also allows us to accept more comments hassle free. Now we can focus on doing the more important tasks, and leaving the monotonous and boring ones to the model.")


if __name__ == '__main__':
    main()