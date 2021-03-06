# Principal packages
import argparse

import cv2
import torch
import av
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer

from mask_detection import utils as mask_utils
from mask_segmentation import utils as segmentation_utils
from ccgan import generate as gan_utils
from ressources import (
    replace_face,
    get_face_detector_model,
    get_mask_detector_model,
    get_mask_segmentation_model,
    get_ccgan_model,
)

# read and preprocess the image
ap = argparse.ArgumentParser()
# construct the argument parser and parse the arguments
ap.add_argument("-i", "--image", type=str, help="image path")
ap.add_argument(
    "-c",
    "--confidence",
    type=float,
    default=0.5,
    help="minimum probability to filter weak detections",
)
args = vars(ap.parse_args())

# Setting custom Page Title and Icon with changed layout and sidebar state
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="centered",
    initial_sidebar_state="expanded",
)


def local_css(file_name):
    """Method for reading styles.css and applying necessary changes to HTML"""
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_models():
    st.write("Cache miss: load_models running")
    # the computation device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Device set to:", device)

    face_detector_path = "model_weights/face_detector"
    mask_detector_model_path = "model_weights/mask_detector_model.pth"
    mask_segmentation_model_path = "model_weights/model_mask_segmentation.pth"
    ccgan_path = "model_weights/ccgan-110.pth"

    try:
        get_face_detector_model()
        get_mask_detector_model()
        get_mask_segmentation_model()
        get_ccgan_model()
    except:
        print("error")
        raise ValueError("Error while loading models")

    maskModel, faceNet = mask_utils.load_models(
        device, face_detector_path, mask_detector_model_path
    )
    segmentation_model = segmentation_utils.load_model(
        device, mask_segmentation_model_path
    )
    generator_model = gan_utils.load_model(ccgan_path, device)
    print("[INFO] Models loaded")
    return faceNet, maskModel, segmentation_model, generator_model


def unmask_frame(image, faceNet, maskModel, segmentation_model, generator_model):
    if image is not None:
        (faces, locs, preds) = mask_utils.detect_and_predict_mask(
            image, faceNet, maskModel, args["confidence"]
        )

        if len(faces) != 0:
            # segment the mask on faces
            faces_mask = segmentation_utils.predict(faces, segmentation_model)

            # predict the face underneath the mask
            gan_preds = gan_utils.predict(
                generator=generator_model, images=faces, masks=faces_mask
            )

            image = replace_face(image, gan_preds, locs)

    return image


class VideoProcessor:
    def __init__(self) -> None:
        self.faceNet = None
        self.maskModel = None
        self.segmentation_model = None
        self.generator_model = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if (
            self.faceNet is not None
            and self.maskModel is not None
            and self.segmentation_model is not None
            and self.generator_model is not None
        ):
            img = unmask_frame(
                img,
                self.faceNet,
                self.maskModel,
                self.segmentation_model,
                self.generator_model,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def set_streamlit_app():
    local_css("css/styles.css")
    st.markdown(
        '<h1 align="center">😷 Face Mask Detection 😚</h1>', unsafe_allow_html=True
    )
    # activities = ["Image", "Webcam"]
    activities = ["Webcam", "Image"]
    st.set_option("deprecation.showfileUploaderEncoding", False)
    st.sidebar.markdown("# Mask Detection on?")
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    faceNet, maskModel, segmentation_model, generator_model = load_models()

    if choice == "Image":
        st.markdown(
            '<h2 align="center">Detection on Image</h2>', unsafe_allow_html=True
        )
        st.markdown("### Upload your image here ⬇")
        image_file = st.file_uploader("", type=["jpg", "jpeg"])  # upload image
        if image_file is not None:
            our_image = Image.open(image_file)  # making compatible to PIL
            our_image.save("./images/out.jpg")
            st.image(image_file, caption="", use_column_width=True)
            st.markdown(
                '<h3 align="center">Image uploaded successfully!</h3>',
                unsafe_allow_html=True,
            )
            if st.button("Process"):
                new_image = unmask_frame(
                    cv2.imread("./images/out.jpg"),
                    faceNet,
                    maskModel,
                    segmentation_model,
                    generator_model,
                )
                st.image(
                    cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB),
                    caption="cool",
                    use_column_width=True,
                )

    if choice == "Webcam":
        st.markdown(
            '<h2 align="center">Detection on Webcam</h2>', unsafe_allow_html=True
        )
        ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
        if ctx.video_processor:
            ctx.video_processor.faceNet = faceNet
            ctx.video_processor.maskModel = maskModel
            ctx.video_processor.segmentation_model = segmentation_model
            ctx.video_processor.generator_model = generator_model


set_streamlit_app()
