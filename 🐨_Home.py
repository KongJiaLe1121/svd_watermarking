import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Koala SVD Watermarking – Home",
    page_icon="🐨"
)

# Title and overview
st.title("🐨 Koala SVD Watermarking")
st.markdown(
    """
    In modern medical workflows, preserving patient data integrity and image quality is paramount. 
    Traditional watermarking methods often compromise diagnostic detail or lack robustness against routine image manipulations. 
    The Koala SVD Watermarking System bridges this gap by using Singular Value Decomposition to embed and extract metadata seamlessly,
    ensuring both imperceptibility and resilience while reducing file sizes for efficient handling.
    """
)

# Objectives
st.subheader("🎯 What We Aim To Achieve")
st.markdown(
    """
    1. **Imperceptible Embedding**: Keep PSNR ≥ 50 dB so diagnostic features remain clear.
    2. **Accurate Extraction**: Ensure BER < 0.03 and NCC ≥ 0.70 under ideal conditions.
    3. **Robustness**: Withstand rotation, noise, blur, and cropping with BER < 0.05 and NCC ≥ 0.60.
    4. **Storage Efficiency**: Convert DICOM to JPEG/PNG formats to reduce file size by up to 80% without quality loss.
    """
)

# Algorithm Evolution
st.subheader("🔍 From Original to Improved")
st.markdown(
    """
    - **Original Approach**: Watermarks were spread across all singular values, which protected data but reduced image quality (PSNR ≈ 48 dB).
    - **Our Improvement**: By embedding exclusively in the middle singular values, we boost PSNR above 50 dB while maintaining BER < 0.03 and NCC ≥ 0.70—even after attacks.

    **Why the Middle?** 
    The middle singular values capture the image’s core structure without touching extremes that affect visibility (Top SV) or noise-sensitive bands (Bottom SV). 
    This sweet spot preserves clarity and strengthens watermark survival.
    """
)

# Core Features
st.subheader("🔧 System Features")
st.markdown(
    """
    - **Embed Module**: Upload DICOM/JPG/PNG, enter metadata, adjust strength, and preview watermarked results.
    - **Extract Module**: Recover and verify hidden data using the original image and key file.
    - **Attack Simulator**: Emulate rotation, noise, blur, and cropping to test resilience live.
    """
)

# Benefits to Users
st.subheader("🌟 Benefits at a Glance")
st.markdown(
    """
    - **Secure Data Handling**: Safeguard patient details within images without visual compromise.
    - **Clinical Confidence**: Retain diagnostic fidelity while embedding robust, recoverable watermarks.
    - **User-Friendly Interface**: Real-time feedback and simple controls reduce learning curves.
    - **Efficiency Boost**: Converting DICOM to lightweight formats (JPEG/PNG) easier for storage and sharing.
    """
)

# Navigation Hint
st.markdown(
    "Use the sidebar to switch between **Embed**, and **Extract** modules and explore the system."
)
