import streamlit as st
from PIL import Image, ImageDraw
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage
import pytesseract
from gtts import gTTS
import io
import base64
import os

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configure Google Gemini API Key
GOOGLE_API_KEY = "AIzaSyCpybaH_MC6gItkK5Sn2fZ4FpU_HDxoVbQ"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=AIzaSyCS06mTr13YCK9FbSVRDV7tjkfR9892aRM)

# Function to convert an image to Base64 format
def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

# Function to run OCR on an image
def run_ocr(image):
    return pytesseract.image_to_string(image).strip()

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

# Function to convert text to speech (using a neutral or female voice)
def text_to_speech(text):
    tts = gTTS(text=text, lang='en', slow=False)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes.getvalue()

# Function to detect and highlight objects in the image
def detect_and_highlight_objects(image):
    # Example: Drawing rectangles as placeholders (Replace with actual object detection model)
    draw = ImageDraw.Draw(image)
    objects = [
        {"label": "Obstacle", "bbox": (50, 50, 200, 200)},
        {"label": "Object", "bbox": (300, 100, 500, 300)}
    ]
    
    for obj in objects:
        bbox = obj['bbox']
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="red", width=5)
        draw.text((bbox[0], bbox[1] - 10), obj['label'], fill="red")
    
    return image, objects

# Main app function
def main():
    st.set_page_config(page_title="AI Assistive Tool", layout="wide", page_icon="🤖")

    # Adding background image to sidebar
    sidebar_image_path = r'C:\Users\dubey\Desktop\ai_assistive_app\ai_assistive_app\back_ground.jpg'

    # Check if the image file exists
    if os.path.exists(sidebar_image_path):
        sidebar_image = Image.open(sidebar_image_path)
        st.sidebar.image(sidebar_image, use_column_width=True)
    else:
        st.sidebar.warning("Sidebar image not found. Please check the file path.")

    st.title('AI Assistive Tool for Visually Impaired 👁️ 🤖')

    # Project Overview
    st.write("""
        This AI-powered tool assists visually impaired individuals by leveraging image analysis. 
        It provides the following features:
        - **Scene Understanding**: Describes the content of uploaded images.
        - **Text-to-Speech Conversion**: Extracts and reads aloud text from images using OCR.
        - **Object & Obstacle Detection**: Identifies objects or obstacles for safe navigation.
        - **Personalized Assistance**: Offers task-specific guidance based on image content, like reading labels or recognizing items.
        
        Upload an image to get started and let AI help you understand and interact with your environment!
    """)

    st.sidebar.header("📂 Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image (jpg, jpeg, png)", type=['jpg', 'jpeg', 'png'])

    st.sidebar.header("🔧 Instructions")
    st.sidebar.write("""
    1. Upload an image.
    2. Choose an option below:
       - 🖼️ Describe Scene: Get a description of the image.
       - 📜 Extract Text: Extract text from the image.
       - 🚧 Detect Objects & Obstacles: Identify obstacles and highlight them.
       - 🛠️ Personalized Assistance: Get task-specific help.
    3. Results will be read aloud for easy understanding.
    """)

    if uploaded_file:
        if 'last_uploaded_file' in st.session_state and st.session_state.last_uploaded_file != uploaded_file:
            st.session_state.extracted_text = None
            st.session_state.summarized_text = None

        st.session_state.last_uploaded_file = uploaded_file
        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🖼️ Describe Scene"):
            with st.spinner("Generating scene description..."):
                scene_prompt = "Describe this image briefly."
                scene_description = analyze_image(image, scene_prompt)
                st.subheader("Scene Description")
                st.success(scene_description)
                st.audio(text_to_speech(scene_description), format='audio/mp3')

        if st.button("📜 Extract Text"):
            with st.spinner("Extracting text..."):
                extracted_text = run_ocr(image)
                st.subheader("Extracted Text")
                if extracted_text:
                    st.info(extracted_text)
                    st.audio(text_to_speech(extracted_text), format='audio/mp3')
                else:
                    st.warning("No text detected in the image.")

        if st.button("🚧 Detect Objects & Obstacles"):
            with st.spinner("Identifying objects and obstacles..."):
                highlighted_image, objects = detect_and_highlight_objects(image.copy())
                st.image(highlighted_image, caption="Highlighted Image with Detected Objects", use_column_width=True)
                st.success(f"Detected Objects: {[obj['label'] for obj in objects]}")

        if st.button("🛠️ Personalized Assistance"):
            with st.spinner("Providing personalized guidance..."):
                task_prompt = "Provide task-specific guidance based on the content of this image in brief. Include item recognition, label reading, and any relevant context."
                assistance_description = analyze_image(image, task_prompt)
                st.subheader("Personalized Assistance")
                st.success(assistance_description)
                st.audio(text_to_speech(assistance_description), format='audio/mp3')

if __name__ == "__main__":
    main()
