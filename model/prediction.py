import base64
import numpy as np
from PIL import Image
import streamlit as st
import joblib
from keras.preprocessing.image import load_img, img_to_array

def model_page():

    model = joblib.load("model.joblib")

    # Call the function to upload and preprocess the user's image
    uploaded_data = upload_image()

    if uploaded_data is not None:
        image_data, image_resized, _ = uploaded_data

        # Display the uploaded image
        st.write("Uploaded image:")
        st.image(image_resized)

        # Make a prediction using the uploaded image
        if image_data is not None:
            image_resized_expanded = np.expand_dims(image_resized, axis=0)
            prediction = model.predict(image_resized_expanded)

            # Interpret the prediction
            if np.argmax(prediction) > 0.1:
                st.write("Prediction: Cancer")
            else:
                st.write("Prediction: Healthy")

def upload_image():
    image_file = st.file_uploader("Upload an image", type="jpg")
    if image_file is not None:
        image_data = np.array(Image.open(image_file))
        image_resized = Image.fromarray(image_data).resize((244, 244))
        image_resized = np.array(image_resized) / 255.0

        # Encode the uploaded image as a Base64 string
        uploaded_image = base64.b64encode(image_data).decode()

        return image_data, image_resized, uploaded_image
    else:
        st.write("Please upload an image")