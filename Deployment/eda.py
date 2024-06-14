import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Load data
train_datagen = ImageDataGenerator(rescale=1./255.)
test_datagen = ImageDataGenerator(rescale=1./255.)
val_datagen = ImageDataGenerator(rescale=1./255.)

# Make train data from directory
train_gen = train_datagen.flow_from_directory(
	"Bone_Fracture_Binary_Classification/train",
	target_size=(150,150),
	class_mode='binary',
  batch_size=126,
	shuffle=True,
    seed=2
)

# Make test data from directory
test_gen = test_datagen.flow_from_directory(
	"Bone_Fracture_Binary_Classification/test",
	target_size=(150,150),
	class_mode='binary',
  batch_size=126,
	shuffle=True,
    seed=2
)

# Make validation data from directory
val_gen = val_datagen.flow_from_directory(
	"Bone_Fracture_Binary_Classification/val",
	target_size=(150,150),
	class_mode='binary',
  batch_size=126,
	shuffle=True,
    seed=2
)

def eda_page():

    st.title("Exploratory Data Analysis")
    st.write('Data exploration is made to better understand the dataset')
    st.subheader("Distribution of Fractured and Non-Fractured Bone")

    # Extract the class labels from the generator
    class_labels = train_gen.classes

    # Count the number of occurrences of each class label
    class_counts = np.bincount(class_labels)

    # Create a pie chart of the class counts
    plt.pie(class_counts, labels=["Fractured", "Not Fractured"], autopct='%1.1f%%', startangle=140)
    plt.axis("equal")
    plt.title("Bone Fracture Distribution")

    # Display the pie chart in the Streamlit app
    st.pyplot(plt)
    st.write("**Description**:")
    st.write('The data is `balanced`, because the images in the data shows **50.2%** of non-fractured bone images and **49.8%** of fractured bone images')
    st.write()
    
    st.subheader("Random Bone Images From The Train, Test, and Validation Data")

    # Plot the images
    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(25,10))
    n = 0

    for i in range(3):
        for j in range(5):
            img = train_gen[0][0][n]
            ax[i][j].imshow(img)
            ax[i][j].set_title('Class - ' + str(train_gen[0][1][n]))
            n += 1

    # Display the plot in the Streamlit app
    st.pyplot(fig)

    # Plot the images
    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(25,10))
    n = 0

    for i in range(3):
        for j in range(5):
            img = test_gen[0][0][n]
            ax[i][j].imshow(img)
            ax[i][j].set_title('Class - ' + str(test_gen[0][1][n]))
            n += 1

    # Display the plot using Streamlit
    st.pyplot(fig)

    # Plot the images
    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(25,10))
    n = 0

    for i in range(3):
        for j in range(5):
            img = val_gen[0][0][n]
            ax[i][j].imshow(img)
            ax[i][j].set_title('Class - ' + str(val_gen[0][1][n]))
            n += 1

    # Display the plot using Streamlit
    st.pyplot(fig)

    st.subheader("**Summary** :")
    st.write('Based on the figures above, we can see the images from each generator :')
    st.write('There are 2 classes in the dataset : `fractured`(0) and `not fractured`(1)')
    st.write('There are many different images from each one of train, test, and validation data. This may help to evaluate the model better in predicting the images')