import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split

# Set up file paths
tumor_dir = r'D:/hacktiv/fase2/gc7/p2-ftds015-hck-g7-albertn15/brian/Brain Tumor'
healthy_dir = r'D:/hacktiv/fase2/gc7/p2-ftds015-hck-g7-albertn15/brian/Healthy'

# Create filepaths and labels lists
filepaths = []
labels = []
dict_list = [tumor_dir, healthy_dir]
for i, j in enumerate(dict_list):
    flist = os.listdir(j)
    for f in flist:
        fpath = os.path.join(j, f)
        filepaths.append(fpath)
        if i == 0:
            labels.append('cancer')
        else:
            labels.append('healthy')

# Create pandas dataframe
Fseries = pd.Series(filepaths, name="filepaths")
Lseries = pd.Series(labels, name="labels")
tumor_data = pd.concat([Fseries, Lseries], axis=1)
tumor_df = pd.DataFrame(tumor_data)

# Split data into training, testing, and validation sets
train_set, val_test_set = train_test_split(tumor_df, test_size=0.4, random_state=42)
val_set, test_images = train_test_split(val_test_set, test_size=0.5, random_state=42)
# Create image data generator
image_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

# Create data flows for training, testing, and validation
train = image_gen.flow_from_dataframe(dataframe=train_set, x_col="filepaths", y_col="labels", target_size=(244, 244), color_mode='rgb', class_mode="categorical", batch_size=32, shuffle=False)
test = image_gen.flow_from_dataframe(dataframe=test_images, x_col="filepaths", y_col="labels", target_size=(244, 244), color_mode='rgb', class_mode="categorical", batch_size=32, shuffle=False)
val = image_gen.flow_from_dataframe(dataframe=val_set, x_col="filepaths", y_col="labels", target_size=(244, 244), color_mode='rgb', class_mode="categorical", batch_size=32, shuffle=False)

def show_brain_images(data_generator):
    # Generate a batch of images
    images, labels = next(data_generator)

    # Plot the images
    fig, axes = plt.subplots(nrows=1, ncols=len(images), figsize=(20, 20))
    for i in range(len(images)):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        axes[i].set_title(labels[i])

    # Close the figure
    plt.close(fig)

    # Return the figure
    return fig

def eda_page():
    st.title("Exploratory Data Analysis")
    st.write('Data exploration is made to better understand the dataset')
    st.subheader("The images with different source image")

    # Load the metadata dataframe
    df_metadata = pd.read_csv('metadata.csv')

    # Filter the dataframe to only include tumor class
    df_tumor = df_metadata[df_metadata['class']=='tumor']

    # Calculate the value counts of the format column
    value_counts = df_tumor['format'].value_counts(normalize=True)

    # Set up explode values
    explode_values = (0.2, 0.4, 0.4)

    # Plot the value counts as a pie chart
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.0f%%', explode=explode_values, startangle=30)
    ax.axis('equal')

    # Display the plot
    st.pyplot(fig)

    # Load the metadata dataframe
    df_metadata = pd.read_csv('metadata.csv')

    # Filter the dataframe to only include normal class
    df_normal = df_metadata[df_metadata['class']=='normal']

    # Calculate the value counts of the format column
    value_counts = df_normal['format'].value_counts(normalize=True)

    # Set up explode values
    explode_values = (0.2, 0.4)

    # Plot the value counts as a pie chart
    fig, aw = plt.subplots(figsize=(5, 5))
    aw.pie(value_counts.values, labels=value_counts.index, autopct='%1.0f%%', explode=(0.2, 0.4), startangle=30, colors=['teal', 'orange'])
    aw.axis('equal')

    # Display the plot
    st.pyplot(fig)

    st.write("Labels distribution:")
    st.write(tumor_df["labels"].value_counts())

    # Show brain images
    st.write("Brain Images:")
    st.write(show_brain_images(train))
        