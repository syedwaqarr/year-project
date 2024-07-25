
# import streamlit as st
# import numpy as np
# from PIL import Image
# import cv2
# from keras.models import load_model
# from tensorflow.keras.applications.vgg19 import VGG19
# from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout



# st.title('Brain Tumor Classifier')
# st.write('')
# st.write('This is a python app which classifies a brain MRI into one of the four classes : ')
# st.write(' No tumor, Pituitary tumor,Meningioma tumor or Glioma tumor')
# file = st.file_uploader(label='Upload image', type=['jpg','jpeg','png'], accept_multiple_files=False, key=None)
# IMAGE_SIZE = 150


# base_model = VGG19(include_top=False, input_shape=(240,240,3))
# x = base_model.output
# flat=Flatten()(x)
# class_1 = Dense(4608, activation='relu')(flat)
# drop_out = Dropout(0.2)(class_1)
# class_2 = Dense(1152, activation='relu')(drop_out)
# output = Dense(2, activation='softmax')(class_2)
# model_03 = base_model(base_model.inputs, output)
# model_03.load_weights('vgg_unfrozen.h5')


# if file is not None:
#     image = Image.open(file)
#     image = np.array(image)
#     # image = image[:,:,::-1].copy()
#     image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
#     # plt.imshow(image)
#     st.image(image)
#     images = image.reshape(1,150,150,3)
#     predictions1 = model_03.predict(images)
#     predictions1 = np.argmax(predictions1, axis=1)
#     labels = ['No Tumor', 'Tumor']
#     st.write('Prediction over the uploaded image:')
#     st.title(labels[predictions1[0]])













# import streamlit as st
# import numpy as np
# from PIL import Image
# import cv2
# from tensorflow.keras.applications.vgg19 import VGG19
# from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
# from tensorflow.keras.models import Model

# st.title('Brain Tumor Classifier')
# st.write('')
# st.write('This is a python app which classifies a brain MRI into one of the four classes : ')
# st.write(' No tumor, Pituitary tumor,Meningioma tumor or Glioma tumor')
# file = st.file_uploader(label='Upload image', type=['jpg','jpeg','png'], accept_multiple_files=False, key=None)
# IMAGE_SIZE = 240  # Adjusted to match VGG19 input shape

# # Load VGG19 base model without top layers
# base_model = VGG19(include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

# # Add your custom classification layers
# x = base_model.output
# x = Flatten()(x)
# x = Dense(4608, activation='relu')(x)
# x = Dropout(0.2)(x)
# x = Dense(1152, activation='relu')(x)
# predictions = Dense(2, activation='softmax')(x)

# # Create model
# model_03 = Model(inputs=base_model.input, outputs=predictions)

# # Load weights
# model_03.load_weights('vgg_unfrozen.h5')

# if file is not None:
#     image = Image.open(file)
#     image = np.array(image)
#     image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))  # Resize image to match model input shape
#     st.image(image)
#     images = np.expand_dims(image, axis=0)  # Add batch dimension
#     predictions1 = model_03.predict(images)
#     predictions1 = np.argmax(predictions1, axis=1)
#     labels = ['No Tumor', 'Tumor']
#     st.write('Prediction over the uploaded image:')
#     st.title(labels[predictions1[0]])






























import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model


def app():

    st.write('')

    # Create directory for uploads if it doesn't exist
    UPLOAD_DIR = 'uploads'
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file = st.file_uploader(label='Upload image', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False, key=None)
    IMAGE_SIZE = 240 

    # Load VGG19 base model without top layers
    base_model = VGG19(include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    # Add your custom classification layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(4608, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1152, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)

    # Create model
    model_03 = Model(inputs=base_model.input, outputs=predictions)

    # Load weights
    model_03.load_weights('vgg_unfrozen.h5')

    if file is not None:
        image = Image.open(file)
        image = np.array(image)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))  # Resize image to match model input shape
        st.image(image)
        
        # Save the uploaded image to the uploads directory
        image_filename = os.path.join(UPLOAD_DIR, file.name)
        cv2.imwrite(image_filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR before saving
        
        with st.spinner('Predicting...'):
            images = np.expand_dims(image, axis=0)  # Add batch dimension
            predictions1 = model_03.predict(images)
            predictions1 = np.argmax(predictions1, axis=1)
            labels = ['No Tumor', 'Tumor']
            st.write('Prediction over the uploaded image:')
            st.title(labels[predictions1[0]])




