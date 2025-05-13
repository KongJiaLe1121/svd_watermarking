import streamlit as st
import numpy as np
import cv2
from PIL import Image
from watermark_module import (
    embed_watermark_svd,
    extract_watermark_svd,
    calculate_psnr,
    calculate_ber_and_ncc,
    damage_image,
    load_dicom_grayscale,
    generate_watermark_image_2col,
    generate_watermark_multi_col,
    extract_all_metadata_rows
)

st.set_page_config(
    page_title="Koala SVD Watermarking – Embed",
    page_icon="🐨"
)
st.title("🔒 Embed Watermark")

input_fmt = st.radio("Input Format", ["PNG / JPG", "DICOM"], horizontal=True)
up = (
    st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if input_fmt == "PNG / JPG"
    else st.file_uploader("Upload DICOM", type=None)
)
if not up:
    st.stop()

if input_fmt == "PNG / JPG":
    base_img = np.array(Image.open(up).convert("L"))
    pid = st.text_input("Patient ID")
    did = st.text_input("Doctor ID")
    hid = st.text_input("Hospital ID")
    vdat = st.date_input("Visit Date").strftime("%Y-%m-%d")
    rows = [
        (f"P:{pid}", f"D:{did}"),
        (f"H:{hid}", vdat)
    ] if all(x.strip() for x in (pid, did, hid)) else None

else:
    base_img, ds = load_dicom_grayscale(up)
    st.write("### Extracted DICOM Metadata")
    rows = extract_all_metadata_rows(ds)  # now excludes PixelData

st.image(base_img, caption="Preview", clamp=True, use_container_width=True)

strength = st.slider("Embedding Strength", 0.01, 0.1, 0.02, step=0.01)

if rows:
    if input_fmt == "PNG / JPG":
        wm = generate_watermark_image_2col(rows, font_scale=0.9, pad=8, line_gap=6)
    else:
        wm = generate_watermark_multi_col(rows, cols=3, font_scale=0.8, pad=10, line_gap=8)
        wm = cv2.resize(wm, (wm.shape[1]*2, wm.shape[0]*2), interpolation=cv2.INTER_NEAREST)
    st.image(wm, caption="Generated Watermark", use_container_width=True)
else:
    wm = None

if st.button("Embed Watermark"):
    if wm is None:
        st.error("No watermark data available.")
        st.stop()

    watermarked_img, key = embed_watermark_svd(
        base_img, wm, strength=strength, embed_low=0.2, embed_high=0.8
    )

    extracted_wm = extract_watermark_svd(watermarked_img, key)
    psnr = calculate_psnr(base_img, watermarked_img)
    ber, ncc = calculate_ber_and_ncc(wm, extracted_wm)

    st.subheader("📊 Original vs. Watermarked")
    c1, c2 = st.columns(2)
    c1.image(base_img, caption="Original", use_container_width=True)
    c2.image(watermarked_img, caption="Watermarked", use_container_width=True)

    st.subheader("✅ Embedding Quality Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("PSNR (dB)", f"{psnr:.2f}")
    m2.metric("BER", f"{ber:.4f}")
    m3.metric("NCC", f"{ncc:.3f}")

    st.subheader("🛡️ Damage Robustness")
    tabs = st.tabs(["Rotation", "Salt & Pepper", "Blur", "Crop"])
    for tab, mode in zip(tabs, ["Rotation", "SaltPepper", "Blur", "Crop"]):
        with tab:
            dmg = damage_image(watermarked_img, mode).astype(np.uint8)
            rec = extract_watermark_svd(dmg, key)
            dber, dncc = calculate_ber_and_ncc(wm, rec)

            cc1, cc2 = st.columns([2,1])
            cc1.image(dmg, caption=mode, use_container_width=True)
            cc2.image(rec, caption="Recovered WM", use_container_width=True)

            st.write("")  # spacer
            dm1, dm2 = st.columns(2)
            dm1.metric("BER", f"{dber:.4f}")
            dm2.metric("NCC", f"{dncc:.3f}")

    import io, zipfile, pickle
    buf_img, buf_key = io.BytesIO(), io.BytesIO()
    Image.fromarray(watermarked_img).save(buf_img, format="PNG")
    pickle.dump(key, buf_key)
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as z:
        z.writestr("watermarked.png", buf_img.getvalue())
        z.writestr("key.pkl", buf_key.getvalue())
    st.download_button("Download Results (ZIP)", zbuf.getvalue(), "watermarked_results.zip")