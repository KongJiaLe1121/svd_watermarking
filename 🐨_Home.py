import streamlit as st

st.set_page_config(
    page_title="Koala SVD Watermarking – Home",
    page_icon="🐨"
)
st.title("🛡 Image Watermarking System")
st.markdown("""
**What This Watermarking System Does**  
- Embed hidden metadata into an image using SVD  
- Extract and verify the watermark (even after rotation, blur, noise…)

**Why SVD?**  
- Tiny singular-value tweaks preserve image quality  
- Robust against many common image operations

Use the sidebar on the left to jump to **Embed** or **Extract**.  
""")