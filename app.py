import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Titel
st.set_page_config(page_title="Bird Classifier", layout="centered")
st.title("🐦 KI-Vogelerkennung")
st.write("Lade ein Bild hoch und die KI sagt dir, welche Vogelart es ist.")

# Modell laden (Caching wichtig!)
@st.cache_resource
def load_model():
    model_name = "chriamue/bird-species-classifier"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return processor, model

processor, model = load_model()

# Bild-Upload
uploaded_file = st.file_uploader("📷 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Dein Bild", use_column_width=True)

    # Preprocessing
    inputs = processor(images=image, return_tensors="pt")

    # Vorhersage
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_id = logits.argmax(-1).item()

    label = model.config.id2label[predicted_class_id]
    confidence = torch.softmax(logits, dim=-1)[0][predicted_class_id].item()

    st.subheader("🔍 Ergebnis")
    st.write(f"**Art:** {label}")
    st.write(f"**Sicherheit:** {confidence:.2%}")
