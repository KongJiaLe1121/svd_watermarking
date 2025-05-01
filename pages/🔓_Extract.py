import streamlit as st
import numpy as np
import pickle
from PIL import Image
from watermark_module import extract_watermark_svd

st.set_page_config(
    page_title="Koala SVD Watermarking – Extract",
    page_icon="🐨"
)
st.title("🔓 Extract Watermark")

up_img = st.file_uploader("Upload Watermarked Image", type=["png","jpg","jpeg"])
up_key = st.file_uploader("Upload Key File (.pkl)", type=["pkl"])
if not (up_img and up_key):
    st.stop()

img = np.array(Image.open(up_img).convert("L"))
key = pickle.load(up_key)
wm  = extract_watermark_svd(img, key)

st.subheader("Recovered Watermark")
st.image(wm, caption="Extracted Watermark", use_container_width=True, clamp=True)
st.success("Extraction complete.")
