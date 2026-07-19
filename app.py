import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from fpdf import FPDF
import datetime
import os
import plotly.graph_objects as go

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="Leaf Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# =============================
# Load Model
# =============================
MODEL_PATH = "models/leaf_model.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

IMG_SIZE = 128
class_names = ['Bacterial_Spot', 'Healthy', 'Late_Blight']

# =============================
# Treatment Info
# =============================
treatment = {
    "Healthy": """
    ✅ The plant is healthy.
    • Maintain proper watering.
    • Ensure adequate sunlight.
    • Continue regular monitoring.
    """,

    "Bacterial_Spot": """
    ⚠ Bacterial Spot Detected.
    • Apply copper-based fungicide.
    • Avoid overhead watering.
    • Remove infected leaves immediately.
    • Improve air circulation.
    """,

    "Late_Blight": """
    ⚠ Late Blight Detected.
    • Apply recommended fungicide spray.
    • Remove affected leaves.
    • Avoid excessive moisture.
    • Monitor nearby plants.
    """
}

# =============================
# Sidebar
# =============================
st.sidebar.title("🌿 Leaf Disease Detection")
st.sidebar.markdown("Upload an image to detect plant leaf disease.")

uploaded_file = st.sidebar.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "png", "jpeg"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📘 Model Information")
st.sidebar.write("Model Type: Convolutional Neural Network (CNN)")
st.sidebar.write("Input Size: 128 x 128 pixels")
st.sidebar.write("Total Classes: 3")

# =============================
# Main Title
# =============================
st.title("🌿 Plant Leaf Disease Detection Dashboard")
st.markdown("---")

# =============================
# Prediction Section
# =============================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    resized_image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(resized_image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    with st.spinner("Analyzing image using CNN model..."):
        prediction = model.predict(image_array)

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction))
    probabilities = prediction[0] * 100

    # 2 COLUMN LAYOUT
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, width=300)

    with col2:
        st.subheader("🔍 Prediction Result")
        st.success(f"Prediction: {predicted_class}")
        st.progress(confidence)
        st.info(f"Confidence: {confidence*100:.2f}%")

    st.markdown("---")
    # Treatment UI
    st.subheader("💊 Treatment Recommendation")
    st.warning(treatment[predicted_class])

    # =============================
    # PDF Section (MUST BE HERE)
    # =============================
    st.subheader("📄 Download Prediction Report (PDF)")

    current_time = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    clean_treatment = treatment[predicted_class]
    clean_treatment = clean_treatment.replace("✅", "")
    clean_treatment = clean_treatment.replace("⚠", "")
    clean_treatment = clean_treatment.replace("•", "-")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Plant Leaf Disease Detection Report", ln=True, align='C')
    pdf.ln(10)

    pdf.cell(200, 10, txt=f"Prediction: {predicted_class}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence*100:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Date & Time: {current_time}", ln=True)

    pdf.ln(10)
    pdf.multi_cell(0, 8, txt="Treatment Recommendation:")
    pdf.multi_cell(0, 8, txt=clean_treatment)

    pdf_output = pdf.output(dest='S').encode('latin-1')

    st.download_button(
        label="⬇ Download PDF Report",
        data=pdf_output,
        file_name="Leaf_Disease_Report.pdf",
        mime="application/pdf"
    )

else:
    st.info("Please upload a leaf image from the sidebar to begin detection.")

# =============================
# Performance Section
# =============================
with st.expander("📊 View Model Performance Metrics"):

    st.subheader("Training Accuracy Graph")
    if os.path.exists("results/accuracy_graph.png"):
        st.image("results/accuracy_graph.png", use_container_width=True)
    else:
        st.write("Accuracy graph not found.")

    st.subheader("Confusion Matrix")
    if os.path.exists("results/confusion_matrix.png"):
        st.image("results/confusion_matrix.png", use_container_width=True)
    else:
        st.write("Confusion matrix not found.")

# =============================
# Footer
# =============================
st.markdown("---")
st.caption("Developed using Deep Learning (CNN) and Streamlit | B.Sc Computer Science Project")