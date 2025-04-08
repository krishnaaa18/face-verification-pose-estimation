import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from py_files.utils import verify_faces, estimate_head_pose

st.title("🔍 Face Verification + Pose Detection")

st.markdown("Upload two images to verify if they are of the same person and detect their head pose.")

col1, col2 = st.columns(2)

with col1:
    uploaded_img1 = st.file_uploader("D:\\Github repoS\\Face_Verification\\data\\k1.jpg", type=['jpg', 'png'], key="img1")

with col2:
    uploaded_img2 = st.file_uploader("D:\\Github repoS\\Face_Verification\\data\\k2.jpg", type=['jpg', 'png'], key="img2")

if uploaded_img1 and uploaded_img2:
    # Save uploaded files to temp files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp1:
        tmp1.write(uploaded_img1.read())
        img1_path = tmp1.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp2:
        tmp2.write(uploaded_img2.read())
        img2_path = tmp2.name

    # Run face verification
    verified, distance = verify_faces(img1_path, img2_path)

    st.subheader("🧠 Face Verification Result")
    st.write(f"**Verified:** {verified}")
    st.write(f"**Distance:** {distance:.4f} (lower = more similar)")

    # Load images and run head pose
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    pose1, pose_img1 = estimate_head_pose(img1)
    pose2, pose_img2 = estimate_head_pose(img2)

    st.subheader("🎯 Head Pose Estimation")
    col3, col4 = st.columns(2)
    with col3:
        st.image(cv2.cvtColor(pose_img1, cv2.COLOR_BGR2RGB), caption=f"Pose: {pose1}")
    with col4:
        st.image(cv2.cvtColor(pose_img2, cv2.COLOR_BGR2RGB), caption=f"Pose: {pose2}")
