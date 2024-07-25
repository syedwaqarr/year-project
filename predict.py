# import streamlit as st
# import numpy as np
# from PIL import Image
# import cv2
# import os
# from tensorflow.keras.applications.vgg19 import VGG19
# from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
# from tensorflow.keras.models import Model


# def app():

#     st.write('')

#     # Create directory for uploads if it doesn't exist
#     UPLOAD_DIR = 'uploads'
#     os.makedirs(UPLOAD_DIR, exist_ok=True)

#     file = st.file_uploader(label='Upload image', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False, key=None)
#     IMAGE_SIZE = 240 

#     # Load VGG19 base model without top layers
#     base_model = VGG19(include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

#     # Add your custom classification layers
#     x = base_model.output
#     x = Flatten()(x)
#     x = Dense(4608, activation='relu')(x)
#     x = Dropout(0.2)(x)
#     x = Dense(1152, activation='relu')(x)
#     predictions = Dense(2, activation='softmax')(x)

#     # Create model
#     model_03 = Model(inputs=base_model.input, outputs=predictions)

#     # Load weights
#     model_03.load_weights('vgg_unfrozen.h5')

#     if file is not None:
#         image = Image.open(file)
#         image = np.array(image)
#         image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))  # Resize image to match model input shape
#         st.image(image)
        
#         # Save the uploaded image to the uploads directory
#         image_filename = os.path.join(UPLOAD_DIR, file.name)
#         cv2.imwrite(image_filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR before saving
        
#         with st.spinner('Predicting...'):
#             images = np.expand_dims(image, axis=0)  # Add batch dimension
#             predictions1 = model_03.predict(images)
#             predictions1 = np.argmax(predictions1, axis=1)
#             labels = ['No Tumor', 'Tumor']
#             st.write('Prediction over the uploaded image:')
#             st.title(labels[predictions1[0]])



import numpy as np
from PIL import Image
from skimage import io, transform, color
import os
import streamlit as st

def app():
    st.write('')

    # Create directory for uploads if it doesn't exist
    UPLOAD_DIR = 'uploads'
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file = st.file_uploader(label='Upload image', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False, key=None)
    IMAGE_SIZE = 240 

    # Load VGG19 base model without top layers
    from tensorflow.keras.applications.vgg19 import VGG19
    from tensorflow.keras.layers import Flatten, Dense, Dropout
    from tensorflow.keras.models import Model

    base_model = VGG19(include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    x = base_model.output
    x = Flatten()(x)
    x = Dense(4608, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1152, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)

    model_03 = Model(inputs=base_model.input, outputs=predictions)
    model_03.load_weights('vgg_unfrozen.h5')

    if file is not None:
        # Load the image with PIL
        image = Image.open(file)
        
        # Convert PIL image to numpy array
        image = np.array(image)
        
        # Resize image using scikit-image
        image_resized = transform.resize(image, (IMAGE_SIZE, IMAGE_SIZE), anti_aliasing=True)

        # Display the image
        st.image(image_resized)
        
        # Save the uploaded image to the uploads directory
        image_filename = os.path.join(UPLOAD_DIR, file.name)

        # Convert to grayscale for saving if needed, or handle it based on your requirement
        # For saving as an RGB image, use the same `image_resized` as it is already in RGB format
        io.imsave(image_filename, (image_resized * 255).astype(np.uint8))  # Convert from float [0,1] to uint8 [0,255]

        with st.spinner('Predicting...'):
            images = np.expand_dims(image_resized, axis=0)  # Add batch dimension
            predictions1 = model_03.predict(images)
            predictions1 = np.argmax(predictions1, axis=1)
            labels = ['No Tumor', 'Tumor']
            st.write('Prediction over the uploaded image:')
            st.title(labels[predictions1[0]])





