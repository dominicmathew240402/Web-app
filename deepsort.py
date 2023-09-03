# Python In-built packages
import os
from pathlib import Path
import PIL
import cv2
import numpy as np

# External packages
import streamlit as st

# Local Modules
import settings
import helper
from deep_sort_realtime.deepsort_tracker import DeepSort, Detection

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
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes

                # Tracking objects with DeepSORT
                frame = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
                
                # Convert boxes to the required format (raw_detections)
                raw_detections = [(bbox, 1.0) for bbox in boxes]  # Use a constant confidence of 1.0

                # Update tracks using the raw detections
                trackers = DeepSort.update_tracks(frame, raw_detections)

                # Create the "train" folder if it doesn't exist
                train_folder = 'train'
                os.makedirs(train_folder, exist_ok=True)

                # Extract unique IDs for tracking
                unique_ids = set()
                for track in trackers:
                    unique_ids.add(track.track_id)

                # Save up to 10 snapshots with unique IDs
                saved_snapshots = 0
                for track in trackers:
                    if saved_snapshots >= 10:
                        break
                    if track.track_id in unique_ids:
                        unique_ids.remove(track.track_id)
                        x, y, w, h = track.to_tlwh()
                        crop = frame[int(y):int(y + h), int(x):int(x + w)]

                        # Save the snapshot with a unique ID
                        snapshot_path = f"train/snapshot_{track.track_id}.jpg"
                        cv2.imwrite(snapshot_path, crop)
                        saved_snapshots += 1

                count = 0

                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image', use_column_width=True)

                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            count += 1

                            # Accessing object-level confidence is not available in the provided structure
                            # box_confidence = box.conf

                            st.write(box.data)
                            # st.write(f"Confidence : {box_confidence}")

                        count1 = count - 1
                        st.write(f"Total Object Detected: {count1}")

                    with st.expander("Tracking Results"):
                        for track in trackers:
                            x, y, w, h = track.to_tlwh()
                            track_id = track.track_id
                            st.write(f"Track ID: {track_id}")
                            st.write(f"Bounding Box: ({x:.2f}, {y:.2f}, {x + w:.2f}, {y + h:.2f})")

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
