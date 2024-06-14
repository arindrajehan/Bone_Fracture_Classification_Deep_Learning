# Libraries
import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from eda import eda_page
from prediction import model_page

# Load data
train_datagen = ImageDataGenerator(rescale=1./255.)

train_gen = train_datagen.flow_from_directory(
	"Bone_Fracture_Binary_Classification/train",
	target_size=(150,150),
	class_mode='binary',
  batch_size=126,
	shuffle=True,
    seed=2
)

# Header
st.header('Graded Challenge 7')
st.write("""
Created by Arindra Jehan - HCK015 """)

# Description
st.write("This program is made to predict bone fracture based on the Bone Fracture Binary Classification Data by using the Improved Functional ANN Model")
st.write("Dataset : Bone Fracture Binary Classification")

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