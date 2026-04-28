import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(
    page_title="Vogel-Erkennung",
    page_icon="🐦",
    layout="centered"
)

st.title("🐦 Vogel-Erkennung (stabile Version)")
st.write("Lade ein Bild hoch und die KI analysiert es.")

# -------------------------
# MODELL (STABIL & CACHED)
# -------------------------
@st.cache_resource
def load_model():
    model = tf.keras.applications.EfficientNetB0(weights="imagenet")
    return model

try:
    model = load_model()
except Exception as e:
    st.error("❌ Modell konnte nicht geladen werden.")
    st.stop()

# -------------------------
# LABELS
# -------------------------
@st.cache_resource
def load_labels():
    import json
    import urllib.request

    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    with urllib.request.urlopen(url) as f:
        return json.load(f)

labels = load_labels()

# -------------------------
# PREPROCESSING (SAFE)
# -------------------------
def preprocess(image):
    image = image.convert("RGB")
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# -------------------------
# VOGEL-KLASSEN
# -------------------------
bird_keywords = [
    "bird", "hen", "cock", "rooster", "ostrich", "goldfinch",
    "robin", "jay", "magpie", "chickadee", "eagle", "vulture",
    "owl", "parrot", "penguin", "flamingo", "pelican", "toucan"
]

# -------------------------
# UPLOAD
# -------------------------
uploaded_file = st.file_uploader("📤 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:

    try:
        image = Image.open(uploaded_file)

        st.image(image, caption="Dein Bild", use_column_width=True)

        with st.spinner("🧠 Analysiere Bild..."):
            processed = preprocess(image)
            prediction = model.predict(processed)

        # Top 5
        top_indices = prediction[0].argsort()[-5:][::-1]

        results = []
        for i in top_indices:
            label = labels[str(i)][1]
            confidence = float(prediction[0][i])
            results.append((label, confidence))

        # -------------------------
        # VOGEL CHECK
        # -------------------------
        found_bird = False
        best_label = results[0][0]
        best_conf = results[0][1]

        for label, confidence in results:
            if any(word in label.lower() for word in bird_keywords):
                found_bird = True
                best_label = label
                best_conf = confidence
                break

        # -------------------------
        # OUTPUT (STABIL UI)
        # -------------------------
        st.subheader("📊 Ergebnis")

        if found_bird:
            st.success(f"🐦 Vogel erkannt: {best_label}")
        else:
            st.warning("❌ Kein Vogel erkannt")

        st.write(f"🔎 Wahrscheinlichkeit: {round(best_conf * 100, 2)}%")

        # Progress bar
        st.progress(int(best_conf * 100))

        # -------------------------
        # DETAILS
        # -------------------------
        with st.expander("🔍 Top 5 Vorhersagen"):
            for label, confidence in results:
                st.write(f"{label} – {round(confidence * 100, 2)}%")

    except Exception as e:
        st.error("❌ Fehler beim Verarbeiten des Bildes.")
        st.exception(e)
