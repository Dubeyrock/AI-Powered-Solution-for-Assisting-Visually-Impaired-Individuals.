import streamlit as st
from PIL import Image, ImageDraw
import pyttsx3
import os
import pytesseract
import torch
from langchain_google_genai import GoogleGenerativeAI

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize Google Generative AI with API Key
GEMINI_API_KEY = "YOUR_API_KEY"  # Replace with your valid API key
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
llm = GoogleGenerativeAI(model="gemini-1.5-pro", api_key=GEMINI_API_KEY)

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Streamlit Page Configuration
st.set_page_config(page_title="SightBeyond", layout="wide", page_icon="👁️")

st.markdown(
    """
    <style>
     .main-title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #0662f6;
        margin-top: -20px;
     }
    .subtitle {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 20px;
    }
    .feature-header {
        font-size: 24px;
        color: #333;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">SightBeyond 👁️</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Solutions for Empowering the Visually Impaired 🗣️</div>', unsafe_allow_html=True)

# Sidebar Configuration
image = Image.open(r"C:\Users\dubey\Desktop\ai_assistive_app\ai_assistive_app\back_ground.png")
image_resized = image.resize((250, 250))
st.sidebar.image(image_resized, use_column_width=False)

st.sidebar.title("ℹ️ About")
st.sidebar.markdown(
    """
    📌 **Features**
    - 🔍 **Describe Scene**: AI insights about the image, including objects and suggestions.
    - 📝 **Extract Text**: Extract visible text using OCR.
    - 🔊 **Text-to-Speech**: Hear the extracted text aloud.
    - 🛑 **Object Detection**: Identify objects/obstacles for safe navigation.
    - 👨‍🏫 **Personalized Assistance**: Task-specific guidance for daily activities.
    
    💡 **How it helps**:
    Assists visually impaired users by providing scene descriptions, text extraction, and object detection.
    """
)

st.sidebar.text_area("📜 Instructions", "Upload an image to start. Choose a feature to interact with.")

# Functions
def extract_text_from_image(image):
    """Extracts text from the given image using OCR."""
    return pytesseract.image_to_string(image)

def text_to_speech(text):
    """Converts the given text to speech."""
    engine.say(text)
    engine.runAndWait()

def generate_scene_description(input_prompt, image_data):
    """Generates a scene description using Google Generative AI."""
    model = llm
    response = model.generate_text(input_prompt, image_data)
    return response.text

def detect_objects(image):
    """Detects objects in the image and returns annotated image with object labels."""
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    results = model(image)
    labels, coords = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    draw = ImageDraw.Draw(image)
    
    for i in range(len(labels)):
        x_min, y_min, x_max, y_max, confidence = coords[i]
        label = model.names[int(labels[i])]
        draw.rectangle([(x_min * image.width, y_min * image.height), 
                        (x_max * image.width, y_max * image.height)], outline="red", width=2)
        draw.text((x_min * image.width, y_min * image.height), f"{label}: {confidence:.2f}", fill="red")
    
    return image, results.pandas().xyxy[0]  # Annotated image and detection info

def input_image_setup(uploaded_file):
    """Prepares the uploaded image for processing."""
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data,
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded.")

# Upload Image Section
st.markdown("<h3 class='feature-header'>📤 Upload an Image</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag and drop or browse an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Buttons Section
st.markdown("<h3 class='feature-header'>⚙️ Features</h3>", unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)

scene_button = col1.button("🔍 Describe Scene")
ocr_button = col2.button("📝 Extract Text")
tts_button = col3.button("🔊 Text-to-Speech")
obj_button = col4.button("🛑 Detect Objects")
assist_button = col5.button("👨‍🏫 Personalized Assistance")

# Input Prompt for Scene Understanding
input_prompt = """
You are an AI assistant helping visually impaired individuals. For the provided image:
1. List the objects and their purposes.
2. Provide safe navigation guide based on detected obstacles.
3. Offer personalized guidance on daily tasks like identifying items, reading text, or performing a specific action.
"""

# Process user interactions
if uploaded_file:
    image_data = input_image_setup(uploaded_file)

    if scene_button:
        with st.spinner("Generating scene description..."):
            response = generate_scene_description(input_prompt, image_data)
            st.markdown("<h3 class='feature-header'>🔍 Scene Description & Guidance</h3>", unsafe_allow_html=True)
            st.write(response)

    if ocr_button:
        with st.spinner("Extracting text from the image..."):
            text = extract_text_from_image(image)
            st.markdown("<h3 class='feature-header'>📝 Extracted Text</h3>", unsafe_allow_html=True)
            st.text_area("Extracted Text", text, height=150)

    if tts_button:
        with st.spinner("Converting text to speech..."):
            text = extract_text_from_image(image)
            if text.strip():
                text_to_speech(text)
                st.success("✅ Text-to-Speech Conversion Completed!")
            else:
                st.warning("No text found to convert.")

    if obj_button:
        with st.spinner("Detecting objects..."):
            annotated_image, detection_data = detect_objects(image)
            st.image(annotated_image, caption="Detected Objects", use_column_width=True)
            st.markdown("<h3 class='feature-header'>📋 Detected Objects & Obstacles</h3>", unsafe_allow_html=True)
            st.write(detection_data)

    if assist_button:
        st.write("This feature provides task-based assistance and context.")

# Footer
st.markdown(
    """
    <hr>
    <footer style="text-align:center;">
        <p>Powered by <strong>Google Gemini API</strong> | Built with ❤️ using Streamlit</p>
    </footer>
    """,
    unsafe_allow_html=True,
)
