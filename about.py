import streamlit as st

def app():

    st.header("Brain Tumor Detection")
    st.write("Welcome! Our Brain Tumor Detection web app uses CNN technology to help identify brain tumors swiftly and accurately.")

    st.divider()

    st.header("The Technology Behind Our Model")
    st.write("Our brain tumor detection model is built on the robust foundations of TensorFlow and Keras, two leading frameworks in the field of artificial intelligence and deep learning. Leveraging Convolutional Neural Networks (CNN), a specialized architecture designed for image classification tasks, we have trained our model to accurately analyze MRI images.")

    st.divider()

    st.header("Data and Methodology")
    st.write("We have sourced our data from Kaggle. The dataset comprises MRI images categorized into two classes: tumorous and non-tumorous. Our model is designed for binary classification, providing users with immediate feedback on whether an uploaded MRI image shows signs of a tumor.")
    
    st.divider()

    st.header("How It Works")
    st.write("Our model processes the image through layers of convolution and pooling, extracting meaningful features that distinguish between tumor and healthy tissue. Using TensorFlow for efficient computation and Keras for streamlined model building, we ensure both accuracy and speed in our predictions.")
