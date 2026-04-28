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
    layout="wide"
)

# -------------------------
# STYLE
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding: 2rem 3rem;
    max-width: 1200px;
}

/* Header */
.app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 0.25rem;
}

.app-title {
    font-family: 'Lora', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #1a2e1a;
    margin: 0;
    line-height: 1.1;
}

.app-subtitle {
    color: #5a7a5a;
    font-size: 1rem;
    font-weight: 300;
    margin-top: 0.2rem;
    margin-bottom: 1.5rem;
}

/* Upload label */
.upload-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: #3d5c3d;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

/* Result cards */
.result-card {
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 16px;
}

.result-card.success {
    background: linear-gradient(135deg, #e8f5e9, #f1f8f1);
    border: 1.5px solid #a5d6a7;
}

.result-card.warning {
    background: linear-gradient(135deg, #fff8e1, #fffde7);
    border: 1.5px solid #ffe082;
}

.result-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
    opacity: 0.65;
}

.result-name {
    font-family: 'Lora', serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #1a2e1a;
    line-height: 1.2;
}

.result-conf {
    font-size: 0.9rem;
    color: #4a6a4a;
    margin-top: 4px;
    font-weight: 400;
}

/* Top predictions */
.pred-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}

.pred-name {
    font-size: 0.88rem;
    color: #2d3a2d;
    min-width: 160px;
    font-weight: 400;
}

.pred-bar-bg {
    flex: 1;
    height: 8px;
    background: #e8ede8;
    border-radius: 99px;
    overflow: hidden;
}

.pred-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #558b2f, #8bc34a);
}

.pred-pct {
    font-size: 0.82rem;
    font-weight: 500;
    color: #4a6a4a;
    min-width: 44px;
    text-align: right;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #f4f7f0;
    border-right: 1px solid #dce8dc;
}

hr {
    border: none;
    border-top: 1px solid #dce8dc;
    margin: 1.2rem 0;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.markdown("### ⚙️ Einstellungen")
    THRESHOLD = st.slider(
        "Mindest-Konfidenz",
        min_value=0.01,
        max_value=0.99,
        value=0.10,
        step=0.01,
        help="Nur Vorhersagen über diesem Wert gelten als sicher erkannt."
    )
    st.markdown("---")
    st.markdown("### ℹ️ Info")
    st.caption(
        "Diese App nutzt EfficientNetB0 (ImageNet) "
        "um Vogelarten auf Fotos zu erkennen. "
        "Lade ein Bild hoch – die KI analysiert es automatisch."
    )
    st.markdown("---")
    st.caption("🐦 Vogel-Erkennung · powered by TensorFlow")

# -------------------------
# HEADER
# -------------------------
st.markdown("""
<div class='app-header'>
    <span style='font-size:2.6rem'>🐦</span>
    <div>
        <div class='app-title'>Vogel-Erkennung</div>
    </div>
</div>
<div class='app-subtitle'>Lade ein Foto hoch – die KI erkennt die Vogelart in Sekunden.</div>
""", unsafe_allow_html=True)

# -------------------------
# MODEL
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.applications.EfficientNetB0(weights="imagenet")

@st.cache_resource
def load_labels():
    import json
    import urllib.request
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    with urllib.request.urlopen(url) as f:
        return json.load(f)

try:
    model = load_model()
    labels = load_labels()
except Exception as e:
    st.error("❌ Modell oder Labels konnten nicht geladen werden.")
    st.stop()

# -------------------------
# BIRD KEYWORDS
# -------------------------
bird_keywords = [
    "bird", "hen", "cock", "rooster", "ostrich", "goldfinch",
    "robin", "jay", "magpie", "chickadee", "eagle", "vulture",
    "owl", "parrot", "penguin", "flamingo", "pelican", "toucan"
]

# -------------------------
# PREPROCESS
# -------------------------
def preprocess(image):
    image = image.convert("RGB")
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# -------------------------
# UPLOAD
# -------------------------
st.markdown("<div class='upload-label'>📤 Bild hochladen</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    try:
        image = Image.open(uploaded_file)

        col_img, col_res = st.columns([1, 1.2], gap="large")

        with col_img:
            st.image(image, caption="📸 Dein Bild", use_container_width=True)

        with col_res:
            with st.spinner("🧠 KI analysiert..."):
                processed = preprocess(image)
                prediction = model.predict(processed)

            # Top 5
            top_indices = prediction[0].argsort()[-5:][::-1]
            results = []
            for i in top_indices:
                label = labels[str(i)][1]
                confidence = float(prediction[0][i])
                results.append((label, confidence))

            # Bird check
            found_bird = False
            best_label = results[0][0]
            best_conf = results[0][1]
            for label, confidence in results:
                if any(word in label.lower() for word in bird_keywords):
                    if confidence >= THRESHOLD:
                        found_bird = True
                        best_label = label
                        best_conf = confidence
                        break

            # Result card
            if found_bird:
                card_class = "success"
                icon = "🐦"
                status = "Vogel erkannt"
            else:
                card_class = "warning"
                icon = "❌"
                status = "Kein Vogel erkannt"

            st.markdown(f"""
            <div class='result-card {card_class}'>
                <div class='result-label'>{icon} {status}</div>
                <div class='result-name'>{best_label.replace("_", " ").title()}</div>
                <div class='result-conf'>Konfidenz: {round(best_conf * 100, 1)} %</div>
            </div>
            """, unsafe_allow_html=True)

            # Top 5 bars
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("**🔎 Top Vorhersagen**")

            for label, conf in results:
                pct = round(conf * 100, 1)
                display_label = label.replace("_", " ").title()
                st.markdown(f"""
                <div class='pred-row'>
                    <span class='pred-name'>{display_label}</span>
                    <div class='pred-bar-bg'>
                        <div class='pred-bar-fill' style='width:{pct}%'></div>
                    </div>
                    <span class='pred-pct'>{pct} %</span>
                </div>
                """, unsafe_allow_html=True)

            # All classes expander
            st.markdown("<hr>", unsafe_allow_html=True)
            with st.expander("📋 Alle Top-Klassen anzeigen"):
                all_top = prediction[0].argsort()[-20:][::-1]
                for i in all_top:
                    lbl = labels[str(i)][1].replace("_", " ").title()
                    pct = round(float(prediction[0][i]) * 100, 2)
                    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                    st.caption(f"`{bar}` {lbl} – {pct} %")

    except Exception as e:
        st.error("❌ Fehler beim Verarbeiten des Bildes.")
        st.exception(e)
