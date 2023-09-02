import streamlit as st
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Load the pre-trained MobileNetV2 model from TensorFlow Hub
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = hub.load(model_url)

# Streamlit UI
st.title("Fruit Class Size Detection")

# Upload an image
st.write("Upload an image of a fruit to detect its class size:")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Define class thresholds based on the predicted values
thresholds = [300, 600]  # You can adjust these thresholds based on your needs

def get_class_name(predicted_class):
    if predicted_class < thresholds[0]:
        return "Small"
    elif predicted_class < thresholds[1]:
        return "Medium"
    else:
        return "Large"

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = image.resize((224, 224))
    image = np.array(image)
    image = image.astype(np.float32) / 255.0  # Convert to float32 and normalize image values to [0, 1]
    image = np.expand_dims(image, axis=0)

    # Make predictions
    predictions = model(image)
    predicted_class = np.argmax(predictions[0])
    predicted_size = str(predicted_class)
    predicted_label = get_class_name(predicted_class)

    # Display the result
    st.write(f"Predicted Class Size: {predicted_size}")
    st.write(f"Predicted Class Size: {predicted_label}")
