"""
VisionLock AI - AI Powered Family Recognition Smart Lock
A premium Streamlit application that uses a trained CNN model to recognize
family members (Khush, Mummy, Papa) and deny access to unknown visitors.

Developer: Khush Arora
"""

import os
import textwrap
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError


def inject_css(raw_css: str) -> None:
    """Render a CSS/HTML block via st.markdown safely.

    Streamlit's markdown renderer treats a blank line inside an HTML block
    as the end of that block, which causes any following CSS to be shown
    as literal page text. Dedenting alone does not fix this, so blank
    lines are stripped before the block is rendered.
    """
    dedented = textwrap.dedent(raw_css)
    no_blank_lines = "\n".join(line for line in dedented.splitlines() if line.strip() != "")
    st.markdown(no_blank_lines, unsafe_allow_html=True)

# TensorFlow / Keras imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "visionlock_model.keras"
IMG_SIZE = 200
CLASS_NAMES = ["Khush", "Mummy", "Papa", "Unknown"]
FAMILY_MEMBERS = {"Khush", "Mummy", "Papa"}

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VisionLock AI",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS - Premium futuristic dark / blue / cyan glassmorphism theme
# ---------------------------------------------------------------------------
inject_css(
    """\
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
    :root {
        --bg-black: #03060f;
        --bg-deep: #060b1a;
        --cyan: #00e5ff;
        --blue: #1e90ff;
        --green: #00ff9d;
        --red: #ff4d6d;
        --glass: rgba(255, 255, 255, 0.06);
        --glass-border: rgba(0, 229, 255, 0.25);
        --text-main: #e8f4ff;
        --text-muted: #8aa0c0;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(1200px 800px at 20% -10%, rgba(30,144,255,0.18), transparent 60%),
                    radial-gradient(1000px 700px at 90% 10%, rgba(0,229,255,0.14), transparent 55%),
                    linear-gradient(180deg, #03060f 0%, #060b1a 60%, #03060f 100%);
        color: var(--text-main);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(rgba(0,229,255,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,229,255,0.04) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events: none;
        z-index: 0;
        animation: gridFloat 20s linear infinite;
    }
    @keyframes gridFloat {
        0% { background-position: 0 0, 0 0; }
        100% { background-position: 40px 40px, 40px 40px; }
    }

    h1, h2, h3, .brand-title {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 1px;
    }

    .brand-title {
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #00e5ff 0%, #1e90ff 50%, #00e5ff 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0,229,255,0.35);
        animation: glowPulse 3s ease-in-out infinite;
    }
    @keyframes glowPulse {
        0%, 100% { filter: drop-shadow(0 0 10px rgba(0,229,255,0.4)); }
        50% { filter: drop-shadow(0 0 25px rgba(0,229,255,0.8)); }
    }

    .subtitle {
        text-align: center;
        font-size: 1.15rem;
        color: var(--cyan);
        margin-top: -0.5rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 500;
    }

    .glass-card {
        background: var(--glass);
        backdrop-filter: blur(18px) saturate(160%);
        -webkit-backdrop-filter: blur(18px) saturate(160%);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 8px 40px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.08);
        transition: transform 0.35s ease, box-shadow 0.35s ease;
        position: relative;
        z-index: 1;
    }
    .glass-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 18px 60px rgba(0,229,255,0.25), inset 0 1px 0 rgba(255,255,255,0.12);
    }

    .feature-card {
        text-align: center;
        padding: 1.5rem 1rem;
        border-radius: 20px;
        background: var(--glass);
        border: 1px solid rgba(0,229,255,0.18);
        backdrop-filter: blur(14px);
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        border-color: var(--cyan);
        box-shadow: 0 0 25px rgba(0,229,255,0.35);
        transform: translateY(-4px);
    }
    .feature-icon {
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    .feature-text {
        color: var(--text-main);
        font-weight: 600;
        font-size: 1rem;
        margin-top: 0.3rem;
    }

    .result-card-granted {
        background: linear-gradient(135deg, rgba(0,255,157,0.12), rgba(0,229,255,0.08));
        border: 2px solid var(--green);
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(0,255,157,0.35);
        animation: fadeInUp 0.6s ease;
    }
    .result-card-denied {
        background: linear-gradient(135deg, rgba(255,77,109,0.15), rgba(255,0,0,0.08));
        border: 2px solid var(--red);
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(255,77,109,0.4);
        animation: shake 0.6s ease;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        20% { transform: translateX(-8px); }
        40% { transform: translateX(8px); }
        60% { transform: translateX(-5px); }
        80% { transform: translateX(5px); }
    }

    .status-granted {
        font-family: 'Orbitron', sans-serif;
        font-size: 2rem;
        font-weight: 900;
        color: var(--green);
        letter-spacing: 2px;
        text-shadow: 0 0 20px rgba(0,255,157,0.6);
    }
    .status-denied {
        font-family: 'Orbitron', sans-serif;
        font-size: 2rem;
        font-weight: 900;
        color: var(--red);
        letter-spacing: 2px;
        text-shadow: 0 0 20px rgba(255,77,109,0.6);
    }
    .person-name {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--cyan);
    }
    .confidence-label {
        color: var(--text-muted);
        font-size: 0.95rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .step-card {
        text-align: center;
        padding: 1.2rem 0.8rem;
        border-radius: 16px;
        background: var(--glass);
        border: 1px solid rgba(30,144,255,0.25);
        transition: all 0.3s ease;
    }
    .step-card:hover {
        border-color: var(--cyan);
        transform: translateY(-4px);
        box-shadow: 0 0 20px rgba(0,229,255,0.3);
    }
    .step-number {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.6rem;
        font-weight: 900;
        color: var(--cyan);
    }
    .step-arrow {
        text-align: center;
        font-size: 1.8rem;
        color: var(--cyan);
        padding: 0.5rem 0;
        animation: arrowBounce 1.5s ease-in-out infinite;
    }
    @keyframes arrowBounce {
        0%, 100% { transform: translateY(0); opacity: 0.7; }
        50% { transform: translateY(6px); opacity: 1; }
    }

    .footer {
        text-align: center;
        padding: 2rem 1rem 1rem;
        color: var(--text-muted);
        font-size: 0.95rem;
        border-top: 1px solid rgba(0,229,255,0.12);
        margin-top: 2rem;
    }

    .upload-wrap {
        display: flex;
        justify-content: center;
        margin: 1.5rem 0;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #060b1a 0%, #03060f 100%);
        border-right: 1px solid rgba(0,229,255,0.18);
    }
    [data-testid="stSidebar"] * {
        color: var(--text-main) !important;
    }
    .sidebar-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--cyan) !important;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid rgba(0,229,255,0.2);
    }
    .sidebar-info-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px dashed rgba(0,229,255,0.12);
        font-size: 0.95rem;
    }
    .sidebar-info-label {
        color: var(--text-muted);
    }
    .sidebar-info-value {
        color: var(--cyan);
        font-weight: 600;
    }

    /* Buttons */
    .stButton > button {
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        letter-spacing: 1px;
        border-radius: 14px;
        padding: 0.7rem 2rem;
        background: linear-gradient(90deg, #00e5ff, #1e90ff);
        color: #03060f;
        border: none;
        box-shadow: 0 0 20px rgba(0,229,255,0.4);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 0 35px rgba(0,229,255,0.7);
        color: #03060f;
    }
    .stButton > button:active {
        transform: translateY(0) scale(0.98);
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00e5ff, #1e90ff);
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(0,229,255,0.6);
    }

    /* Uploaded image card */
    .image-card {
        border-radius: 20px;
        overflow: hidden;
        border: 2px solid var(--glass-border);
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        padding: 8px;
        background: var(--glass);
        backdrop-filter: blur(14px);
    }

    .section-heading {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        color: var(--cyan);
        margin: 2rem 0 1rem;
        letter-spacing: 2px;
    }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--cyan), transparent);
        margin: 2rem 0;
        opacity: 0.4;
    }
    </style>
    """
)


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_visionlock_model(path: str):
    """Load the trained Keras CNN model from disk."""
    if not os.path.exists(path):
        return None
    try:
        return load_model(path)
    except Exception as exc:
        st.error(f"Failed to load the CNN model: {exc}")
        return None


# ---------------------------------------------------------------------------
# Image preprocessing - exactly compatible with the training notebook
# ---------------------------------------------------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize to 200x200, convert to array, scale by 255, expand dims."""
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_image(model, image: Image.Image):
    """Run inference and return (class_name, confidence, all_probs)."""
    processed = preprocess_image(image)
    preds = model.predict(processed, verbose=0)[0]
    idx = int(np.argmax(preds))
    class_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "Unknown"
    confidence = float(preds[idx]) * 100.0
    return class_name, confidence, preds


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------
def render_sidebar():
    """Render the project information sidebar."""
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🔐 VisionLock AI</div>', unsafe_allow_html=True)
        st.markdown("### Project Information")

        rows = [
            ("CNN Model", "visionlock_model.keras"),
            ("Classes", "4 (Khush, Mummy, Papa, Unknown)"),
            ("Input Size", "200 x 200"),
            ("Framework", "TensorFlow / Keras"),
            ("Frontend", "Streamlit"),
            ("Developer", "Khush Arora"),
        ]
        for label, value in rows:
            st.markdown(
                f'<div class="sidebar-info-row">'
                f'<span class="sidebar-info-label">{label}</span>'
                f'<span class="sidebar-info-value">{value}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            '<div style="text-align:center; color:var(--text-muted); font-size:0.85rem;">'
            '🛡️ AI Powered Family Recognition Smart Lock'
            '</div>',
            unsafe_allow_html=True,
        )


def render_hero():
    """Render the top hero section with logo, title and subtitle."""
    st.markdown(
        """
        <div style="text-align:center; padding: 2rem 1rem 1rem;">
            <div style="font-size: 4rem; animation: glowPulse 3s ease-in-out infinite;">🔐</div>
            <div class="brand-title">VisionLock AI</div>
            <div class="subtitle">AI Powered Family Recognition Smart Lock</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_features():
    """Render the feature cards row."""
    features = [
        ("🧠", "CNN Powered", "Convolutional Neural Network architecture"),
        ("👁️", "Smart Recognition", "Identifies family members instantly"),
        ("🛡️", "Secure Access", "Grants or denies entry automatically"),
        ("⚡", "Real Time Prediction", "Fast inference on every upload"),
    ]
    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-text">{title}</div>
                    <div style="color:var(--text-muted); font-size:0.82rem; margin-top:0.3rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_upload_section():
    """Render the upload button and return the uploaded file."""
    st.markdown('<div class="upload-wrap">', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return uploaded


def render_image_card(image: Image.Image):
    """Display the uploaded image inside a premium rounded card."""
    st.markdown(
        f"""
        <div class="image-card" style="text-align:center;">
            <img src="data:image/png;base64,{image_to_base64(image)}"
                 style="max-width:100%; border-radius:16px;" />
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_to_base64(image: Image.Image) -> str:
    """Convert a PIL image to a base64 string for inline display."""
    import base64
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def render_result_card(class_name: str, confidence: float):
    """Render the large premium result card based on prediction."""
    is_family = class_name in FAMILY_MEMBERS

    if is_family:
        icon = "🛡️"
        status_text = "ACCESS GRANTED"
        welcome = f"Welcome {class_name}"
        card_class = "result-card-granted"
        status_class = "status-granted"
    else:
        icon = "⚠️"
        status_text = "ACCESS DENIED"
        welcome = "Unknown Visitor Detected"
        card_class = "result-card-denied"
        status_class = "status-denied"

    st.markdown(
        f"""
        <div class="{card_class}">
            <div style="font-size: 3.5rem;">{icon}</div>
            <div class="confidence-label" style="margin-top:0.5rem;">Detected Person</div>
            <div class="person-name">{class_name}</div>
            <div class="confidence-label" style="margin-top:1rem;">Confidence</div>
            <div style="font-family:'Orbitron',sans-serif; font-size:2.2rem; font-weight:900;
                        color:var(--cyan); margin:0.3rem 0;">{confidence:.2f}%</div>
            <div class="{status_class}" style="margin-top:1rem;">{status_text}</div>
            <div style="margin-top:0.8rem; font-size:1.1rem; color:var(--text-main);
                        font-weight:600;">{welcome}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confidence_bar(confidence: float, granted: bool):
    """Render a beautiful progress bar for the confidence value."""
    color_label = "Confidence Level"
    st.markdown(f'<div class="confidence-label" style="margin:1rem 0 0.4rem;">{color_label}</div>',
                unsafe_allow_html=True)
    bar = st.progress(0)
    # Animate the bar
    for pct in range(0, int(confidence) + 1, 2):
        bar.progress(min(pct, 100))
    bar.progress(int(confidence))
    if granted:
        st.success(f"Identity verified with {confidence:.2f}% confidence.")
    else:
        st.error(f"Unrecognized visitor — confidence {confidence:.2f}%.")


def render_how_it_works():
    """Render the 'How VisionLock Works' bottom section."""
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">⚙️ How VisionLock Works</div>', unsafe_allow_html=True)

    steps = [
        ("01", "📤", "Upload Image", "User uploads a face photo"),
        ("02", "🧠", "CNN Model", "200x200 image processed by the network"),
        ("03", "🔮", "Prediction", "Model predicts one of 4 classes"),
        ("04", "🛡️", "Smart Decision", "Access granted or denied"),
    ]

    cols = st.columns(4)
    for i, (num, icon, title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-number">{num}</div>
                    <div style="font-size:2rem;">{icon}</div>
                    <div style="font-weight:700; color:var(--cyan); margin:0.4rem 0;">{title}</div>
                    <div style="color:var(--text-muted); font-size:0.82rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if i < len(steps) - 1:
                st.markdown('<div class="step-arrow">↓</div>', unsafe_allow_html=True)


def render_footer():
    """Render the footer."""
    st.markdown(
        """
        <div class="footer">
            Made with ❤️ using TensorFlow + Streamlit<br/>
            <span style="color:var(--cyan);">VisionLock AI</span> &copy; 2025 — Developed by Khush Arora
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main():
    render_sidebar()
    render_hero()
    render_features()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Load model
    model = load_visionlock_model(MODEL_PATH)
    if model is None:
        st.error(
            "🔒 The CNN model file `visionlock_model.keras` was not found in the application directory. "
            "Please place the trained model next to `app.py` and restart."
        )
        render_how_it_works()
        render_footer()
        return

    # Upload section
    st.markdown('<div class="section-heading">📤 Upload Face Image</div>', unsafe_allow_html=True)
    uploaded_file = render_upload_section()

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except UnidentifiedImageError:
            st.error("Invalid image file. Please upload a valid JPG, PNG or WEBP image.")
            render_how_it_works()
            render_footer()
            return
        except Exception as exc:
            st.error(f"Could not read the uploaded image: {exc}")
            render_how_it_works()
            render_footer()
            return

        # Show image + predict button
        col_img, col_action = st.columns([2, 1])
        with col_img:
            st.markdown("### Uploaded Image")
            render_image_card(image)
        with col_action:
            st.markdown("### Action")
            st.markdown(
                '<div style="color:var(--text-muted); font-size:0.9rem; margin-bottom:0.6rem;">'
                'Review the image, then run the recognition.'
                '</div>',
                unsafe_allow_html=True,
            )
            predict_clicked = st.button("🔮 Predict", use_container_width=True)

        if predict_clicked:
            with st.spinner("Analyzing image with the CNN model..."):
                try:
                    class_name, confidence, _ = predict_image(model, image)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    render_how_it_works()
                    render_footer()
                    return

            granted = class_name in FAMILY_MEMBERS

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-heading">🎯 Recognition Result</div>', unsafe_allow_html=True)

            res_col1, res_col2 = st.columns([2, 1])
            with res_col1:
                render_result_card(class_name, confidence)
            with res_col2:
                st.markdown("### Confidence Meter")
                render_confidence_bar(confidence, granted)
                st.markdown(
                    f"""
                    <div class="glass-card" style="margin-top:1rem;">
                        <div class="confidence-label">Model Output</div>
                        <div style="font-family:'Orbitron',sans-serif; font-size:1.2rem;
                                    color:var(--cyan); margin-top:0.3rem;">{class_name}</div>
                        <div class="confidence-label" style="margin-top:0.8rem;">Status</div>
                        <div style="font-weight:700; color:
                            {'var(--green)' if granted else 'var(--red)'};
                            font-size:1.1rem; margin-top:0.2rem;">
                            {'GRANTED' if granted else 'DENIED'}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    render_how_it_works()
    render_footer()


if __name__ == "__main__":
    main()
