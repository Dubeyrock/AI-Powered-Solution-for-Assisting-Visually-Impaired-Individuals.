import os
import streamlit as st
from PIL import Image, ImageDraw
import pytesseract
from gtts import gTTS
import io
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import torch
from torchvision.transforms import functional as F

# Set Tesseract OCR path (use environment variable for portability)
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# Configure Google Gemini API Key
GOOGLE_API_KEY = "AIzaSyCpybaH_MC6gItkK5Sn2fZ4FpU_HDxoVbQ"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# Load YOLOv5 Model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to convert an image to Base64 format
def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

# Function to run OCR on an image
def run_ocr(image):
    try:
        return pytesseract.image_to_string(image).strip()
    except Exception as e:
        return f"Error: Unable to process OCR - {str(e)}"

# Function to analyze the image using Gemini
def analyze_image(image, prompt):
    try:
        image_base64 = image_to_base64(image)
        message = HumanMessage(
            content=[{"type": "text", "text": prompt},
                     {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}]
        )
        response = llm.invoke([message])
        return response.content.strip()
    except Exception as e:
        return f"Error: Unable to analyze image - {str(e)}"

# Function to detect objects using YOLOv5
def detect_objects(image):
    try:
        img_tensor = F.to_tensor(image).unsqueeze(0)
        results = yolo_model(img_tensor)
        return results.pandas().xyxy[0]
    except Exception as e:
        return f"Error: Unable to detect objects - {str(e)}"

# Function to highlight detected objects
def highlight_objects(image, detections):
    draw = ImageDraw.Draw(image)
    for _, row in detections.iterrows():
        bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
        label = f"{row['name']} ({row['confidence']:.2f})"
        draw.rectangle(bbox, outline="red", width=3)
        draw.text((bbox[0], bbox[1] - 10), label, fill="red")
    return image

# Function to convert text to speech with language support
def text_to_speech(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes.getvalue()
    except Exception as e:
        return f"Error: Unable to convert text to speech - {str(e)}"

# Main app function
def main():
    st.set_page_config(page_title="AI Assistive Tool", layout="wide", page_icon="🤖")

    st.title('AI Assistive Tool for Visually Impaired 👁️ 🤖')

    st.sidebar.header("📂 Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image (jpg, jpeg, png)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        option = st.selectbox("Choose an Action", ["🖼️ Describe Scene", "📜 Extract Text", "🚧 Detect Objects & Obstacles", "🛠️ Personalized Assistance"])

        if st.button("Run"):
            if option == "🖼️ Describe Scene":
                with st.spinner("Generating scene description..."):
                    prompt = "Describe this image briefly."
                    result = analyze_image(image, prompt)
                    st.subheader("Scene Description")
                    st.success(result)
                    st.audio(text_to_speech(result), format='audio/mp3')

            elif option == "📜 Extract Text":
                with st.spinner("Extracting text..."):
                    text = run_ocr(image)
                    st.subheader("Extracted Text")
                    st.info(text)
                    st.audio(text_to_speech(text), format='audio/mp3')

            elif option == "🚧 Detect Objects & Obstacles":
                with st.spinner("Detecting objects..."):
                    detections = detect_objects(image)
                    if isinstance(detections, str):  # Error case
                        st.error(detections)
                    else:
                        highlighted_image = highlight_objects(image.copy(), detections)
                        st.image(highlighted_image, caption="Detected Objects", use_column_width=True)
                        detection_summary = "\n".join(
                            [f"{row['name']} - Confidence: {row['confidence']:.2f}" for _, row in detections.iterrows()]
                        )
                        st.success(detection_summary)
                        st.audio(text_to_speech(detection_summary), format='audio/mp3')

            elif option == "🛠️ Personalized Assistance":
                with st.spinner("Providing personalized guidance..."):
                    prompt = "Provide task-specific guidance based on this image."
                    result = analyze_image(image, prompt)
                    st.subheader("Personalized Assistance")
                    st.success(result)
                    st.audio(text_to_speech(result), format='audio/mp3')

if __name__ == "__main__":
    main()
