import streamlit as st
import numpy as np
import joblib
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="🧠",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0d0d1a;
    color: #e8e8f0;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 1200px;
    margin: auto;
}

.hero {
    text-align: center;
    padding: 1.2rem 0 1.8rem;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #38bdf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #8b8ba8;
    font-size: 0.95rem;
    margin: 0;
}

.card {
    background: #13132a;
    border: 1px solid #2a2a4a;
    border-radius: 16px;
    padding: 1.6rem;
}
.card-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 1rem;
}

div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 1.2rem;
    font-size: 1rem;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    cursor: pointer;
    transition: opacity 0.2s;
    margin-top: 0.8rem;
}
div.stButton > button:hover { opacity: 0.85; }

.result-badge {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    background: linear-gradient(135deg, #1a1a3e, #0f2a1e);
    border: 1px solid #34d399;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.2rem;
}
.result-digit {
    font-size: 5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #34d399, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    min-width: 60px;
    text-align: center;
}
.result-label {
    font-size: 0.78rem;
    color: #8b8ba8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.2rem;
}
.result-text {
    font-size: 1rem;
    color: #e8e8f0;
    font-weight: 600;
}

.placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 260px;
    border: 2px dashed #2a2a4a;
    border-radius: 14px;
    color: #4a4a6a;
    font-size: 0.9rem;
    text-align: center;
    gap: 0.6rem;
}
.placeholder span { font-size: 2.5rem; }

.stat-row {
    display: flex;
    gap: 0.8rem;
    margin-top: 1rem;
}
.stat-box {
    flex: 1;
    background: #0d0d1a;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 0.8rem;
    text-align: center;
}
.stat-val {
    font-size: 1.3rem;
    font-weight: 700;
    color: #a78bfa;
}
.stat-lbl {
    font-size: 0.7rem;
    color: #8b8ba8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.stImage img {
    border-radius: 8px;
    border: 1px solid #2a2a4a;
    image-rendering: pixelated;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("svm_digit_model.pkl")

try:
    model = load_model()
    model_ready = True
except Exception:
    model_ready = False

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧠 Digit Recognizer</h1>
    <p>Draw a digit on the left — see the result instantly on the right</p>
</div>
""", unsafe_allow_html=True)

if not model_ready:
    st.error("⚠️ Model not found. Place `svm_digit_model.pkl` in the same folder.")
    st.stop()

# ── Two-column layout ─────────────────────────────────────────────────────────
left, gap, right = st.columns([1.1, 0.05, 1])

# ── LEFT: Draw ────────────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="card"><div class="card-title">✏️ Draw a digit (0 – 9)</div>', unsafe_allow_html=True)

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=16,
        stroke_color="white",
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

    predict_btn = st.button("⚡ Predict Digit")

    st.markdown("""
        <p style='color:#4a4a6a; font-size:0.78rem; margin-top:0.5rem;'>
        Tip: Draw thick and centered for best accuracy. Refresh to clear.
        </p>
    </div>""", unsafe_allow_html=True)

# ── GAP ───────────────────────────────────────────────────────────────────────
with gap:
    st.markdown("<div style='height:100%;border-left:1px solid #2a2a4a;margin:auto;'></div>", unsafe_allow_html=True)

# ── RIGHT: Result ─────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="card"><div class="card-title">🎯 Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        if canvas_result.image_data is None or canvas_result.image_data.max() == 0:
            st.warning("✏️ Nothing drawn yet — sketch a digit first.")
        else:
            with st.spinner("Analyzing..."):
                img = Image.fromarray(canvas_result.image_data[:, :, 0].astype(np.uint8))
                img = img.convert("L").resize((28, 28))
                img_array = np.array(img) / 255.0
                img_flat = img_array.reshape(1, 784)
                prediction = model.predict(img_flat)[0]
                active_pixels = int((img_array > 0.1).sum())
                coverage = round((active_pixels / 784) * 100, 1)

            st.markdown(f"""
            <div class="result-badge">
                <div class="result-digit">{prediction}</div>
                <div>
                    <div class="result-label">Recognized digit</div>
                    <div class="result-text">Model predicts: <strong>{prediction}</strong></div>
                </div>
            </div>
            <div class="stat-row">
                <div class="stat-box">
                    <div class="stat-val">{prediction}</div>
                    <div class="stat-lbl">Digit</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val">{active_pixels}</div>
                    <div class="stat-lbl">Active px</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val">{coverage}%</div>
                    <div class="stat-lbl">Coverage</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='card-title' style='margin-top:1rem;'>🔬 28×28 input preview</div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1, 1, 1])
            with c2:
                st.image(img, width=100, caption="Model sees this")

    else:
        st.markdown("""
        <div class="placeholder">
            <span>🎯</span>
            Draw a digit on the left<br>then hit <strong>Predict Digit</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)