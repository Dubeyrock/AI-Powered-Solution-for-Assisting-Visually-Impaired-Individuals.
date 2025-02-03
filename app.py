import streamlit as st
from PIL import Image, ImageDraw
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage
import pytesseract
import base64
import io
import os
import cv2
import numpy as np
from pyzbar.pyzbar import decode

# Ensure correct Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Configure Google Gemini API Key
GOOGLE_API_KEY = "AIzaSyAHQUzwDADFhIueFxYB2AYT5ekTf8ssQLw" 

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# Function to convert an image to Base64 format
def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

# Function to run OCR on an image
def run_ocr(image, is_handwritten=False):
    custom_config = "--psm 6" if is_handwritten else ""
    return pytesseract.image_to_string(image, config=custom_config).strip()

# Function to analyze the image using Gemini
def analyze_image(image, prompt):
    try:
        image_base64 = image_to_base64(image)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
            ]
        )
        response = llm.invoke([message])
        return response.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Function to scan QR and Barcodes
def scan_qr_barcode(image):
    img_cv = np.array(image)
    decoded_objects = decode(img_cv)
    results = []
    for obj in decoded_objects:
        results.append(obj.data.decode('utf-8'))
    return results

# Function to recognize dominant colors in the image
def recognize_colors(image, k=5):
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    img_cv = img_cv.reshape((-1, 3))
    kmeans = cv2.KMeans(n_clusters=k, n_init=10)
    kmeans.fit(img_cv)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

# Main app function
def main():
    st.set_page_config(page_title="AI Assistive Tool", layout="wide", page_icon="🤖")
    st.title('AI Assistive Tool for Visually Impaired 👁️ 🤖')

    st.sidebar.header("📂 Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image (jpg, jpeg, png)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("📝 Extract Handwritten Text"):
            with st.spinner("Extracting handwritten text..."):
                extracted_text = run_ocr(image, is_handwritten=True)
                st.subheader("Extracted Handwritten Text")
                st.info(extracted_text if extracted_text else "No handwritten text detected.")

        if st.button("🖼️ AI-Based Image Captioning"):
            with st.spinner("Generating caption..."):
                caption_prompt = "Generate a meaningful caption for this image."
                caption = analyze_image(image, caption_prompt)
                st.subheader("Generated Caption")
                st.success(caption)

        if st.button("📌 Scan QR & Barcodes"):
            with st.spinner("Scanning..."):
                results = scan_qr_barcode(image)
                st.subheader("QR & Barcode Data")
                if results:
                    for res in results:
                        st.success(res)
                else:
                    st.warning("No QR or Barcodes detected.")

        if st.button("🎨 Recognize Colors"):
            with st.spinner("Analyzing colors..."):
                colors = recognize_colors(image)
                st.subheader("Recognized Colors")
                for i, color in enumerate(colors):
                    color_hex = '#%02x%02x%02x' % tuple(color)
                    st.markdown(f"<div style='background-color:{color_hex}; padding:10px; border-radius:5px;'> Color {i+1}: {color_hex}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
