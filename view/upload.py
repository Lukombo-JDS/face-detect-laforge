import os

import streamlit as st

from interface import detect_faces, path_to_images

# TODO: Refactor

# def image_uploded(output_img_dir: str):

#     uploaded_file = st.file_uploader("Choose an image", type="jpg")
#     if uploaded_file is not None:
#         # To read file as bytes:

#         detect_faces(uploaded_file.getvalue(), output_img_dir)
#         images = os.listdir(output_img_dir)
#         faces = os.listdir(output_img_dir + "/faces")
#         print("path images: ", images.remove("faces"))
#         print("path faces: ", faces)

#         st.image(path_to_images(output_img_dir + "/faces", faces))
#         st.image(path_to_images(output_img_dir, images))
