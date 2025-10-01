import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import time
import sqlite3
from datetime import datetime

# ======================
# Konfigurasi Halaman
# ======================
st.set_page_config(
    page_title="GAN Image Colorization",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definisikan path dan nama konstanta
MODEL_PATH = "best_generator.h5"
DB_NAME = "colorization_history.db"
MODEL_INPUT_SIZE = 256

# ======================
# Manajemen Database (SQLite)
# ======================

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                original_image BLOB NOT NULL,
                colorized_image BLOB NOT NULL
            )
        """)
        conn.commit()

def add_to_history(original_bytes, colorized_bytes):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO history (timestamp, original_image, colorized_image) VALUES (?, ?, ?)",
            (timestamp, original_bytes, colorized_bytes)
        )
        conn.commit()

def get_history():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, original_image, colorized_image FROM history ORDER BY id DESC")
        return cursor.fetchall()

def clear_history():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM history")
        conn.commit()

init_db()

# ======================
# Muat Model (dengan Caching)
# ======================
@st.cache_resource
def load_colorization_model():
    try:
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        return None

model = load_colorization_model()

# ======================
# CSS Kustom - Modern Dark Theme dengan Animasi Enhanced
# ======================
st.markdown(
    """
    <style>
    /* Import Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Hide sidebar collapse buttons */
    [data-testid="collapsedControl"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    
    button[kind="header"] {
        display: none !important;
    }
    
    [data-testid="stSidebar"] button[aria-label*="collapse"] {
        display: none !important;
    }
    
    [data-testid="stSidebar"] > div > button {
        display: none !important;
    }
    
    section[data-testid="stSidebar"] {
        width: 21rem !important;
        min-width: 21rem !important;
        transform: none !important;
    }
    
    section[data-testid="stSidebar"] > div {
        width: 21rem !important;
        min-width: 21rem !important;
    }
    
    section[data-testid="stSidebar"][aria-expanded="false"] {
        display: block !important;
        margin-left: 0 !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff;
    }
    
    /* Main Title with Animation */
    .main-title {
        text-align: center;
        font-size: 5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 75%, #4facfe 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        padding-top: 0.5rem;
        animation: titlePulse 3s ease-in-out infinite, gradientShift 4s ease infinite;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
        letter-spacing: 2px;
    }
    
    @keyframes titlePulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.03); }
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .subtitle {
        text-align: center;
        color: #b8b8d1;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
        font-weight: 400;
        animation: fadeInUp 1s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.1) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
        padding: 15px !important;
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(102, 126, 234, 0.2) !important;
        transform: translateX(5px);
    }
    
    .streamlit-expanderContent {
        background: rgba(135, 206, 250, 0.15) !important;
        border: 1px solid rgba(135, 206, 250, 0.3) !important;
        border-radius: 10px !important;
        padding: 20px !important;
        margin-top: 10px !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        animation: expandFade 0.3s ease-out;
    }
    
    @keyframes expandFade {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Card Container with Hover Effect */
    .card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: cardSlideIn 0.5s ease-out;
    }
    
    @keyframes cardSlideIn {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px 0 rgba(102, 126, 234, 0.3);
    }
    
    /* Upload Section with Enhanced Animation */
    .upload-section {
        background: rgba(102, 126, 234, 0.1);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        margin: 30px 0;
        transition: all 0.3s ease;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            border-color: rgba(102, 126, 234, 0.5);
            box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.4);
        }
        50% {
            border-color: rgba(102, 126, 234, 0.8);
            box-shadow: 0 0 20px 5px rgba(102, 126, 234, 0.2);
        }
    }
    
    .upload-section:hover {
        border-color: rgba(102, 126, 234, 0.8);
        background: rgba(102, 126, 234, 0.15);
        animation: none;
    }
    
    /* File Uploader Styling */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 12px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        transform: scale(1.01);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    [data-testid="stFileUploader"] label {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    [data-testid="stFileUploader"] section {
        border: 2px dashed #667eea !important;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.98) !important;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"] section:hover {
        border-color: #764ba2 !important;
        background: rgba(102, 126, 234, 0.05) !important;
    }
    
    [data-testid="stFileUploader"] section > div {
        color: #000000 !important;
    }
    
    [data-testid="stFileUploader"] section > div > div {
        color: #000000 !important;
    }
    
    [data-testid="stFileUploader"] section p {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stFileUploader"] section span {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stFileUploader"] small {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] span {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stFileUploader"] svg {
        fill: #667eea !important;
        stroke: #667eea !important;
    }
    
    /* Image Preview with Animation */
    .stImage > img {
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        animation: imageZoomIn 0.5s ease-out;
        transition: transform 0.3s ease;
    }
    
    .stImage > img:hover {
        transform: scale(1.02);
    }
    
    @keyframes imageZoomIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Button Styling with Enhanced Animation */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        color: white;
        border: none;
        padding: 15px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.98);
    }
    
    /* Star Animation Keyframes */
    @keyframes starBurst {
        0% {
            opacity: 1;
            transform: translate(-50%, -50%) scale(0) rotate(0deg);
        }
        50% {
            opacity: 0.8;
        }
        100% {
            opacity: 0;
            transform: translate(-50%, -50%) scale(3) rotate(720deg);
        }
    }
    
    /* Star particles */
    .star-particle {
        position: fixed;
        font-size: 30px;
        pointer-events: none;
        z-index: 9999;
        animation: starBurst 1.5s ease-out forwards;
    }
    
    /* Confetti Animation */
    @keyframes confettiFall {
        0% {
            transform: translateY(-100vh) rotate(0deg);
            opacity: 1;
        }
        100% {
            transform: translateY(100vh) rotate(720deg);
            opacity: 0;
        }
    }
    
    .confetti {
        position: fixed;
        width: 10px;
        height: 10px;
        pointer-events: none;
        z-index: 9999;
        animation: confettiFall 3s linear forwards;
    }
    
    /* Ripple Effect for Clicks */
    @keyframes ripple {
        0% {
            transform: scale(0);
            opacity: 0.8;
        }
        100% {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    .ripple-effect {
        position: fixed;
        border-radius: 50%;
        border: 2px solid rgba(102, 126, 234, 0.8);
        width: 20px;
        height: 20px;
        pointer-events: none;
        z-index: 9999;
        animation: ripple 0.6s ease-out;
    }
    
    /* Sparkle Effect */
    @keyframes sparkle {
        0%, 100% {
            opacity: 0;
            transform: scale(0) rotate(0deg);
        }
        50% {
            opacity: 1;
            transform: scale(1) rotate(180deg);
        }
    }
    
    .sparkle {
        position: fixed;
        pointer-events: none;
        z-index: 9999;
        animation: sparkle 1s ease-out;
    }
    
    /* Download Button */
    .stDownloadButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 600;
        color: #ffffff;
        background: rgba(0, 170, 255, 0.2);
        border: 2px solid #00aaff;
        padding: 12px 0;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background: #00aaff;
        color: #0f0c29;
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0, 170, 255, 0.4);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff !important;
        margin-bottom: 20px;
        padding: 15px 20px;
        background: rgba(102, 126, 234, 0.15);
        border-radius: 10px;
        border-left: 4px solid #667eea;
        backdrop-filter: blur(5px);
        animation: slideInLeft 0.5s ease-out;
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* All text white */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    p, span, div, label, .stMarkdown, .stText, .stCaption {
        color: #ffffff !important;
    }
    
    /* Info boxes */
    .stInfo {
        background: rgba(102, 126, 234, 0.15);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 15px;
        position: relative;
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .stInfo * {
        color: #ffffff !important;
        position: relative;
        z-index: 1;
    }
    
    /* Alert boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 15px;
        position: relative;
        animation: bounceIn 0.5s ease-out;
    }
    
    @keyframes bounceIn {
        0% {
            opacity: 0;
            transform: scale(0.3);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .stAlert * {
        color: #ffffff !important;
        position: relative;
        z-index: 1;
    }
    
    .stSuccess {
        background: rgba(76, 175, 80, 0.15);
        border-left: 4px solid #4CAF50;
    }
    
    .stWarning {
        background: rgba(255, 152, 0, 0.15);
        border-left: 4px solid #FF9800;
    }
    
    .stError {
        background: rgba(244, 67, 54, 0.15);
        border-left: 4px solid #F44336;
    }
    
    /* Deprecation Warning Box */
    .stException, [data-testid="stNotification"] {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px);
        border-radius: 10px !important;
        padding: 15px !important;
        border-left: 4px solid #FF9800 !important;
    }
    
    .stException *, [data-testid="stNotification"] * {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }
    
    /* History Section */
    .history-item {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s ease-out;
    }
    
    .history-item:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateX(10px);
    }
    
    /* Slider Styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Progress Bar Animation */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #667eea 100%);
        background-size: 200% 100%;
        animation: progressShine 1.5s linear infinite;
    }
    
    @keyframes progressShine {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.5);
        border-radius: 5px;
        transition: background 0.3s ease;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(102, 126, 234, 0.7);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# JavaScript untuk animasi bintang, confetti, dan efek klik
st.markdown("""
    <script>
    // Star Burst Animation
    function createStarBurst(x, y) {
        const stars = ['⭐', '✨', '🌟', '💫', '⚡', '🎆', '🎇'];
        const numStars = 15;
        
        for (let i = 0; i < numStars; i++) {
            const star = document.createElement('div');
            star.className = 'star-particle';
            star.textContent = stars[Math.floor(Math.random() * stars.length)];
            star.style.left = x + 'px';
            star.style.top = y + 'px';
            star.style.animationDelay = (i * 0.05) + 's';
            
            document.body.appendChild(star);
            
            setTimeout(() => {
                star.remove();
            }, 1500);
        }
    }
    
    // Confetti Animation
    function createConfetti(x, y) {
        const colors = ['#667eea', '#764ba2', '#00aaff', '#ff6b6b', '#4ecdc4', '#ffd93d'];
        const shapes = ['▀', '▄', '█', '▌', '▐', '■', '□', '●', '○'];
        const numConfetti = 30;
        
        for (let i = 0; i < numConfetti; i++) {
            const confetti = document.createElement('div');
            confetti.className = 'confetti';
            confetti.textContent = shapes[Math.floor(Math.random() * shapes.length)];
            confetti.style.left = (x + (Math.random() - 0.5) * 100) + 'px';
            confetti.style.top = y + 'px';
            confetti.style.color = colors[Math.floor(Math.random() * colors.length)];
            confetti.style.fontSize = (Math.random() * 20 + 10) + 'px';
            confetti.style.animationDelay = (Math.random() * 0.3) + 's';
            confetti.style.animationDuration = (Math.random() * 2 + 2) + 's';
            
            document.body.appendChild(confetti);
            
            setTimeout(() => {
                confetti.remove();
            }, 5000);
        }
    }
    
    // Ripple Effect on Click
    function createRipple(x, y) {
        const ripple = document.createElement('div');
        ripple.className = 'ripple-effect';
        ripple.style.left = (x - 10) + 'px';
        ripple.style.top = (y - 10) + 'px';
        
        document.body.appendChild(ripple);
        
        setTimeout(() => {
            ripple.remove();
        }, 600);
    }
    
    // Sparkle Effect
    function createSparkle(x, y) {
        const sparkles = ['✨', '⭐', '💫', '🌟'];
        const numSparkles = 5;
        
        for (let i = 0; i < numSparkles; i++) {
            const sparkle = document.createElement('div');
            sparkle.className = 'sparkle';
            sparkle.textContent = sparkles[Math.floor(Math.random() * sparkles.length)];
            sparkle.style.left = (x + (Math.random() - 0.5) * 50) + 'px';
            sparkle.style.top = (y + (Math.random() - 0.5) * 50) + 'px';
            sparkle.style.fontSize = (Math.random() * 15 + 15) + 'px';
            sparkle.style.animationDelay = (i * 0.1) + 's';
            
            document.body.appendChild(sparkle);
            
            setTimeout(() => {
                sparkle.remove();
            }, 1000);
        }
    }
    
    // Global click handler for ripple effect
    document.addEventListener('click', function(e) {
        createRipple(e.clientX, e.clientY);
        
        // Random sparkle effect on some clicks
        if (Math.random() > 0.7) {
            createSparkle(e.clientX, e.clientY);
        }
    });
    
    // Trigger animations on colorize button click
    document.addEventListener('click', function(e) {
        const button = e.target.closest('button');
        if (button && button.textContent.includes('COLORIZE')) {
            const rect = button.getBoundingClientRect();
            const x = rect.left + rect.width / 2;
            const y = rect.top + rect.height / 2;
            createStarBurst(x, y);
            createConfetti(x, y);
            createSparkle(x, y);
        }
    });
    
    // File upload success animation - SAMA dengan button colorize
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                if (node.nodeType === 1) {
                    const fileUploader = node.querySelector('[data-testid="stFileUploader"]');
                    if (fileUploader) {
                        const rect = fileUploader.getBoundingClientRect();
                        const x = rect.left + rect.width / 2;
                        const y = rect.top + rect.height / 2;
                        
                        setTimeout(() => {
                            createStarBurst(x, y);
                            createConfetti(x, y);
                            createSparkle(x, y);
                        }, 500);
                    }
                }
            });
        });
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
    </script>
""", unsafe_allow_html=True)

# ======================
# Inisialisasi Session State
# ======================
if 'colorized_image' not in st.session_state:
    st.session_state.colorized_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'image_bytes' not in st.session_state:
    st.session_state.image_bytes = None
if 'output_width' not in st.session_state:
    st.session_state.output_width = 512
if 'output_height' not in st.session_state:
    st.session_state.output_height = 1287

# ======================
# Sidebar - Parameter Settings
# ======================
with st.sidebar:
    st.markdown("### ⚙️ Output Settings")
    st.markdown("---")
    
    st.markdown("#### 📐 Lebar Output")
    st.session_state.output_width = st.slider(
        "Width",
        min_value=256,
        max_value=2048,
        value=512,
        step=128,
        help="Lebar gambar output"
    )
    
    st.markdown("#### 📐 Tinggi Output")
    st.session_state.output_height = st.slider(
        "Height",
        min_value=256,
        max_value=2048,
        value=1287,
        step=128,
        help="Tinggi gambar output"
    )
    
    st.markdown("---")
    st.markdown("### 📊 History")
    
    history_data = get_history()
    history_count = len(history_data)
    
    st.metric("Total Colorizations", history_count)
    
    if history_count > 0:
        if st.button("🗑️ Clear All History", use_container_width=True):
            clear_history()
            st.success("✅ History cleared!")
            time.sleep(1)
            st.rerun()

# ======================
# Main Content - Header dipindah ke paling atas
# ======================
st.markdown('<div class="main-title">GAN Image Colorization</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Mengubah gambar hitam putih menjadi berwarna dengan model GAN</div>', unsafe_allow_html=True)

# ======================
# Upload & Settings Section - PALING ATAS
# ======================
st.markdown('''
    <div style="
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff !important;
        margin-bottom: 20px;
        padding: 15px 20px;
        background: rgba(102, 126, 234, 0.15);
        border-radius: 10px;
        border-left: 4px solid #667eea;
        backdrop-filter: blur(5px);
    ">📤 Upload & Settings</div>
''', unsafe_allow_html=True)

# Upload file
uploaded_file = st.file_uploader(
    "🖼️ Drag and Drop file here • Limit 200mb • JPG, PNG",
    type=["jpg", "jpeg", "png"],
    help="Upload gambar hitam putih untuk diwarnai"
)

if uploaded_file is not None:
    # Hanya reset jika file berbeda
    new_bytes = uploaded_file.getvalue()
    if st.session_state.image_bytes != new_bytes:
        st.session_state.image_bytes = new_bytes
        st.session_state.original_image = Image.open(io.BytesIO(st.session_state.image_bytes)).convert("RGB")
        st.session_state.colorized_image = None
        st.success("✅ Gambar berhasil diupload!")
        st.balloons()

# Preview Settings
st.markdown('''
    <div style="
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff !important;
        margin: 20px 0 15px 0;
        padding: 12px 15px;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 8px;
        border-left: 3px solid #667eea;
    ">📏 Atur Ukuran Preview (PX)</div>
''', unsafe_allow_html=True)

col_slider1, col_slider2 = st.columns(2)
with col_slider1:
    preview_min = st.slider("Minimum Size", 200, 400, 200, 50)
with col_slider2:
    preview_max = st.slider("Maximum Size", 400, 800, 600, 50)

st.markdown("---")

# ======================
# Main Display Area - Input & Output
# ======================
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('''
        <div style="
            font-size: 1.5rem;
            font-weight: 600;
            color: #ffffff !important;
            margin-bottom: 20px;
            padding: 15px 20px;
            background: rgba(102, 126, 234, 0.15);
            border-radius: 10px;
            border-left: 4px solid #667eea;
            backdrop-filter: blur(5px);
        ">📥 Input</div>
    ''', unsafe_allow_html=True)
    
    if st.session_state.original_image is not None:
        st.image(st.session_state.original_image, use_container_width=True)
        
        # Show image info
        width, height = st.session_state.original_image.size
        st.caption(f"📐 Dimensi: {width} × {height} px")
    else:
        st.info("📤 Silakan upload gambar terlebih dahulu")

with col2:
    st.markdown('''
        <div style="
            font-size: 1.5rem;
            font-weight: 600;
            color: #ffffff !important;
            margin-bottom: 20px;
            padding: 15px 20px;
            background: rgba(102, 126, 234, 0.15);
            border-radius: 10px;
            border-left: 4px solid #667eea;
            backdrop-filter: blur(5px);
        ">📤 Output</div>
    ''', unsafe_allow_html=True)
    
    if st.session_state.colorized_image is not None:
        st.image(st.session_state.colorized_image, use_container_width=True)
        
        # Show output info
        width, height = st.session_state.colorized_image.size
        st.caption(f"📐 Dimensi: {width} × {height} px")
        
        # Download buttons
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            buf_png = io.BytesIO()
            st.session_state.colorized_image.save(buf_png, format="PNG")
            st.download_button(
                label="💾 Download PNG",
                data=buf_png.getvalue(),
                file_name=f"colorized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True
            )
        
        with col_download2:
            buf_jpg = io.BytesIO()
            st.session_state.colorized_image.convert("RGB").save(buf_jpg, format="JPEG", quality=95)
            st.download_button(
                label="💾 Download JPG",
                data=buf_jpg.getvalue(),
                file_name=f"colorized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
    else:
        st.info("🎨 Hasil colorization akan muncul di sini")

# Colorize Button
if st.session_state.original_image is not None:
    if st.button("✨ COLORIZE IMAGE", use_container_width=True):
        if model is not None:
            try:
                with st.spinner("🎨 AI sedang mewarnai gambar Anda..."):
                    progress_bar = st.progress(0)
                    
                    # Preprocessing
                    original_image = st.session_state.original_image
                    img_resized = original_image.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
                    progress_bar.progress(25)
                    
                    img_array = np.array(img_resized) / 255.0
                    img_array_expanded = np.expand_dims(img_array, axis=0)
                    progress_bar.progress(50)
                    
                    # Prediksi
                    pred_array = model.predict(img_array_expanded, verbose=0)
                    progress_bar.progress(75)
                    
                    # Postprocessing
                    pred_clipped = np.clip(pred_array[0], 0, 1)
                    colorized_img = Image.fromarray((pred_clipped * 255).astype(np.uint8))
                    
                    # Resize to output dimensions
                    output_size = (st.session_state.output_width, st.session_state.output_height)
                    colorized_img = colorized_img.resize(output_size, Image.LANCZOS)
                    progress_bar.progress(90)
                    
                    # Simpan ke database
                    buf_colorized = io.BytesIO()
                    colorized_img.save(buf_colorized, format="PNG")
                    add_to_history(st.session_state.image_bytes, buf_colorized.getvalue())
                    
                    # Set session state
                    st.session_state.colorized_image = colorized_img
                    progress_bar.progress(100)
                    time.sleep(0.5)
                    
                st.success('✅ Gambar berhasil diwarnai!', icon='🎉')
                st.balloons()
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error saat colorization: {str(e)}")
        else:
            st.error("❌ Model tidak dapat dimuat. Pastikan file model tersedia.")

st.markdown("---")

# ======================
# About Section - dipindah ke bawah setelah colorize
# ======================
with st.expander("ℹ️ Tentang Website Ini", expanded=False):
    st.markdown("""
    ### Apa itu Image Colorization?
    Image Colorization adalah proses mengubah gambar hitam putih (grayscale) menjadi gambar berwarna 
    menggunakan teknologi Deep Learning dengan arsitektur **GAN**.
    
    ### Fitur Aplikasi:
    - 🎨 Colorization otomatis dengan AI
    - 📊 Pengaturan resolusi output
    - 💾 Download hasil dalam format PNG/JPG
    - 📜 History otomatis tersimpan
    - 🖼️ Preview real-time
    
    ### Cara Menggunakan:
    1. Upload gambar hitam putih (max 200MB)
    2. Atur ukuran output di sidebar (opsional)
    3. Klik tombol **Colorize**
    4. Download hasil colorization
    """)

st.markdown("---")

# ======================
# History Section
# ======================
st.markdown('''
    <div style="
        font-size: 1.8rem;
        font-weight: 600;
        color: #ffffff !important;
        margin-bottom: 20px;
        padding: 15px 20px;
        background: rgba(102, 126, 234, 0.15);
        border-radius: 10px;
        border-left: 4px solid #667eea;
        backdrop-filter: blur(5px);
    ">📜 Colorization History</div>
''', unsafe_allow_html=True)

history_data = get_history()

if not history_data:
    st.info("📂 Riwayat colorization Anda akan muncul di sini")
else:
    # Show latest 5 items
    for idx, item in enumerate(history_data[:5]):
        _, timestamp, original_bytes, colorized_bytes = item
        
        with st.container():
            st.markdown('<div class="history-item">', unsafe_allow_html=True)
            st.caption(f"🕐 {timestamp}")
            
            hist_col1, hist_col2 = st.columns(2, gap="medium")
            
            with hist_col1:
                st.image(original_bytes, caption="Original", use_container_width=True)
            with hist_col2:
                st.image(colorized_bytes, caption="Colorized", use_container_width=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
        
        if idx < len(history_data[:5]) - 1:
            st.markdown("<br>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #b8b8d1; padding: 20px;'>
        <p>Made by Jeremy Nathanael Sidabutar</p>
        <p style='font-size: 0.9rem;'>GAN Image Colorization © 2025</p>
    </div>
    """,
    unsafe_allow_html=True

)
