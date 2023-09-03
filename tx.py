# Python In-built packages
from pathlib import Path
import PIL
import os

# External packages
import streamlit as st
import numpy as np
from deep_sort_realtime.deepsort_tracker import Tracker

# Local Modules
import settings
import helper

# Image dimensions
image_width = 480  # Replace with the actual width of your image
image_height = 360  # Replace with the actual height of your image

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes

                count = 0
                res_plotted = res[0].plot()[:, :, ::-1]

                 # Convert the relevant attributes to NumPy arrays
                xyxy = boxes.xyxy.numpy()
                conf = boxes.conf.numpy()
                cls = boxes.cls.numpy()

                # Convert the NumPy arrays to Python lists
                xyxy_list = xyxy.tolist()
                conf_list = conf.tolist()
                cls_list = cls.tolist()

                # Create a folder to save images and annotations
                output_folder = "train"
                os.makedirs(output_folder, exist_ok=True)

                # Generate a unique filename for the detected image
                image_filename = f"detected_{source_img.name}"

                # Save the detected image
                detected_image_path = os.path.join(output_folder, image_filename)
                PIL.Image.fromarray(res_plotted).save(detected_image_path, format="PNG")

                # Extract and save annotations to a text file
                annotation_filename = f"annotations_{source_img.name}.txt"
                annotation_filepath = os.path.join(output_folder, annotation_filename)

                with open(annotation_filepath, "w") as annotation_file:
                    boxes = res[0].boxes
                    for i, box in enumerate(boxes):
                        annotation_file.write(f"{cls_list[i]} {xyxy_list[i] }\n")

                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                
                # Display the converted lists
                st.write("Bounding Boxes (xyxy format):")
                st.write(xyxy_list)

                st.write("Confidence Values:")
                st.write(conf_list)

                st.write("Class Values:")
                st.write(cls_list)

                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            count=count+1
                            st.write(box.data)

                        count1 = count - 1

                        st.write(f"Total Object Detected: {count1}")
                
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.PICTURE:
    helper.play_capture_image(confidence, model)

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
