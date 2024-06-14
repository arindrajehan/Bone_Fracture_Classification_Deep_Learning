import base64
import numpy as np
from PIL import Image
import streamlit as st
from joblib import load
from keras.preprocessing import image

def model_page():
    st.title("Model Prediction of Bone Fractures")
    st.write("The model predicts from the image given if the bone is fractured or not")
    st.header('User Input Features')

    model = load("cnn_model_final.joblib")

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
            if prediction > 0.5:
                st.write("Prediction: Not fractured")
            else:
                st.write("Prediction: Fractured")

def upload_image():
    image_file = st.file_uploader("Upload an image", type="jpg")
    if image_file is not None:
        image_data = np.array(Image.open(image_file))
        image_resized = image.img_to_array(image.load_img(image_file, target_size=(150, 150))) / 255.0

        # Encode the uploaded image as a Base64 string
        uploaded_image = base64.b64encode(image_data).decode()

        return image_data, image_resized, uploaded_image
    else:
        st.write("Please upload an image")