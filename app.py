import codecs
import os

import av
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as stc
import torch
import utils
from PIL import Image, ImageEnhance
from streamlit_webrtc import webrtc_streamer

# Setting custom Page Title and Icon with changed layout and sidebar state
st.set_page_config(page_title='Face Mask Detector', page_icon='😷', layout='centered', initial_sidebar_state='expanded')


def local_css(file_name):
    """ Method for reading styles.css and applying necessary changes to HTML"""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def mask_image(image):
    # the computation device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print("[INFO] Device set to: ", device)
	
    maskModel, faceNet = utils.load_models(device, "face_detector")

    (_, locs, preds) =  utils.detect_and_predict_mask(image, faceNet, maskModel, 0.5)

    image = utils.display_result(locs, preds, image)

    return image

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        new_image = mask_image(img)

        return av.VideoFrame.from_ndarray(new_image, format="bgr24")

def mask_detection():
    local_css("css/styles.css")
    st.markdown('<h1 align="center">😷 Face Mask Detection 😚</h1>', unsafe_allow_html=True)
    # activities = ["Image", "Webcam"]
    activities = ["Webcam","Image"]
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.sidebar.markdown("# Mask Detection on?")
    choice = st.sidebar.selectbox("Choose among the given options:", activities)

    if choice == 'Image':
        st.markdown('<h2 align="center">Detection on Image</h2>', unsafe_allow_html=True)
        st.markdown("### Upload your image here ⬇")
        image_file = st.file_uploader("", type=['jpg', 'jpeg'])  # upload image
        if image_file is not None:
            our_image = Image.open(image_file)  # making compatible to PIL
            our_image.save('./images/out.jpg')
            st.image(image_file, caption='', use_column_width=True)
            st.markdown('<h3 align="center">Image uploaded successfully!</h3>', unsafe_allow_html=True)
            if st.button('Process'):
                new_image = mask_image(cv2.imread("./images/out.jpg"))
                st.image(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB),  caption='cool',use_column_width=True)

    if choice == 'Webcam':
        st.markdown('<h2 align="center">Detection on Webcam</h2>', unsafe_allow_html=True)
        webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
mask_detection()
