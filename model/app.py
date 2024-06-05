# Libraries
import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from eda import eda_page
from prediction import model_page


# Header
st.header('Graded Challenge 7')
st.write("""
Created by albert novanto p j - HCK015 """)

# Description
st.write("Dataset : [Brian Tumor dataset](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset/data)")

# Main menu function
def main():
    # Define menu options
    menu_options = ["Data Analysis", "Model Prediction"]

    # Create sidebar menu
    selected_option = st.sidebar.radio("Menu", menu_options)

    # Display selected page
    if selected_option == "Data Analysis":
        eda_page()
    elif selected_option == "Model Prediction":
        model_page()

if __name__ == "__main__":
    main()