import streamlit as st
import numpy as np
from PIL import Image
from watermark_module import (
    embed_watermark_svd, extract_watermark_svd,
    calculate_psnr, calculate_ber_and_ncc, damage_image,
    load_dicom_grayscale, generate_watermark_image_2col
)

st.set_page_config(
    page_title="Koala SVD Watermarking – Embed",
    page_icon="🐨"
)
st.title("🔒 Embed Watermark")

# Choose input format
input_fmt = st.radio("Input Format", ["PNG / JPG", "DICOM"], horizontal=True)
up = (st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
      if input_fmt == "PNG / JPG"
      else st.file_uploader("Upload DICOM", type=["dcm"]))
if not up:
    st.stop()

# Load and preview
if input_fmt == "PNG / JPG":
    base_img = np.array(Image.open(up).convert("L"))
else:
    base_img, _ = load_dicom_grayscale(up)
st.image(base_img, caption="Preview", clamp=True, use_container_width=True)

# Watermark parameters
strength = st.slider("Embedding Strength", 0.01, 0.1, 0.02, step=0.01)
pid = st.text_input("Patient ID")
did = st.text_input("Doctor ID")
hid = st.text_input("Hospital ID")
vdat = st.date_input("Visit Date").strftime("%Y-%m-%d")

# Generate watermark
ready = all(x.strip() for x in [pid, did, hid])
watermark = generate_watermark_image_2col(
    [(f"P:{pid}", f"D:{did}"), (f"H:{hid}", vdat)], 0.9
) if ready else None
if watermark is not None:
    st.image(watermark, caption="Generated Watermark", use_container_width=True)

# Embed action
if st.button("Embed Watermark"):
    if watermark is None:
        st.error("Please fill in Patient, Doctor, and Hospital IDs.")
        st.stop()

    watermarked_img, key = embed_watermark_svd(base_img, watermark, strength)
    extracted_wm = extract_watermark_svd(watermarked_img, key)
    psnr = calculate_psnr(base_img, watermarked_img)
    ber, ncc = calculate_ber_and_ncc(watermark, extracted_wm)

    st.subheader("📊 Original vs. Watermarked")
    col_orig, col_wm = st.columns(2)
    col_orig.image(base_img, caption="Original Image", use_container_width=True)
    col_wm.image(watermarked_img, caption="Watermarked Image", use_container_width=True)

    st.subheader("✅ Embedding Quality Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("PSNR (dB)", f"{psnr:.2f}")
    m2.metric("BER",       f"{ber:.4f}")
    m3.metric("NCC",       f"{ncc:.3f}")

    st.subheader("🛡️ Damage Robustness")
    tabs = st.tabs(["Rotation","Salt & Pepper","Blur","Crop"])
    for tab, mode in zip(tabs, ["Rotation","SaltPepper","Blur","Crop"]):
        with tab:
            dmg = damage_image(watermarked_img, mode).astype(np.uint8)
            rec = extract_watermark_svd(dmg, key)
            dber, dncc = calculate_ber_and_ncc(watermark, rec)

            c1, c2 = st.columns([2,1])
            c1.image(dmg, caption=f"{mode} Image",        use_container_width=True)
            c2.image(rec, caption="Recovered Watermark", use_container_width=True)

            st.write("")  # small spacer
            dm1, dm2 = st.columns(2)
            dm1.metric("BER", f"{dber:.4f}")
            dm2.metric("NCC", f"{dncc:.3f}")

    # Download ZIP of results (PNG + key)
    import io, zipfile, pickle
    buf_img, buf_key = io.BytesIO(), io.BytesIO()
    Image.fromarray(watermarked_img).save(buf_img, format="PNG")
    pickle.dump(key, buf_key)
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as z:
        z.writestr("watermarked.png", buf_img.getvalue())
        z.writestr("key.pkl", buf_key.getvalue())
    st.download_button("Download Results (ZIP)", zbuf.getvalue(), "watermarked_results.zip")
