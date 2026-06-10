import streamlit as st
import numpy as np
from PIL import Image
import os
import pandas as pd
import base64
from io import BytesIO

# ==========================================
# 1. IMPORT TFLITE
# ==========================================
try:
    import tflite_runtime.interpreter as tflite
except:
    import tensorflow as tf
    tflite = tf.lite

# ==========================================
# 2. KONFIGURASI HALAMAN & STATE TEMA
# ==========================================
st.set_page_config(
    page_title="Coral Bleaching Detection",
    page_icon="🪸",
    layout="centered"
)

# Inisialisasi state tema (Default: Dark)
if 'theme' not in st.session_state:
    st.session_state.theme = 'Dark'

# ==========================================
# 3. LOGIKA GANTI TEMA
# ==========================================
def toggle_theme():
    if st.session_state.theme == 'Dark':
        st.session_state.theme = 'Light'
    else:
        st.session_state.theme = 'Dark'
    st.rerun()

# Tentukan Warna & Gambar berdasarkan Tema
if st.session_state.theme == 'Dark':
    # --- DARK MODE ---
    bg_overlay = "rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.8)"
    bg_image_url = "https://images.unsplash.com/photo-1590424734253-d68a042f17c6?q=80&w=2000&auto=format&fit=crop"
    container_bg = "rgba(15, 23, 42, 0.90)" 
    text_color = "#e2e8f0"
    card_bg = "rgba(30, 41, 59, 0.9)"
    box_bg = "rgba(30, 41, 59, 0.6)"
    border_color = "rgba(255,255,255,0.15)"
    header_gradient = "rgba(2, 62, 138, 0.95), rgba(0, 0, 0, 0.9)"
    header_text_color = "white"
    shadow_color = "rgba(0, 180, 216, 0.3)"
    upload_bg = "rgba(30, 41, 59, 0.6)"
    upload_border = "rgba(255,255,255,0.2)"
    upload_hover_bg = "rgba(30, 41, 59, 0.8)"
    upload_text_color = "#ffffff"
    upload_subtext_color = "#cbd5e1"
    upload_button_bg = "rgba(15, 23, 42, 0.8)"
    upload_button_text = "#ffffff"
    upload_button_border = "rgba(255,255,255,0.2)"
    tab_active_bg = "linear-gradient(90deg, #0077b6, #00b4d8)"
else:
    # --- LIGHT MODE ---
    bg_overlay = "rgba(255, 255, 255, 0.4), rgba(255, 255, 255, 0.7)"
    bg_image_url = "https://images.unsplash.com/photo-1682687221038-404670f09d1c?q=80&w=2000&auto=format&fit=crop"
    container_bg = "rgba(255, 255, 255, 0.92)"
    text_color = "#1e293b"
    card_bg = "rgba(255, 255, 255, 0.95)"
    box_bg = "rgba(241, 245, 249, 0.8)"
    border_color = "rgba(0, 0, 0, 0.1)"
    header_gradient = "rgba(224, 242, 254, 0.95), rgba(255, 255, 255, 0.9)"
    header_text_color = "#0c4a6e"
    shadow_color = "rgba(0, 0, 0, 0.1)"
    upload_bg = "#ffffff"
    upload_border = "#e2e8f0"
    upload_hover_bg = "#f8fafc"
    upload_text_color = "#1e293b"
    upload_subtext_color = "#64748b"
    upload_button_bg = "#f1f5f9"
    upload_button_text = "#1e293b"
    upload_button_border = "#cbd5e1"
    tab_active_bg = "linear-gradient(90deg, #0077b6, #00b4d8)"

# ==========================================
# 4. CUSTOM CSS
# ==========================================
st.markdown(f"""
<style>
/* Sembunyikan default streamlit */
#MainMenu {{visibility:hidden;}}
footer {{visibility:hidden;}}
header {{visibility:hidden;}}

/* Animasi Background Bergerak */
@keyframes zoomEffect {{
    0% {{ background-size: 100%; }}
    100% {{ background-size: 115%; }}
}}

/* Background Utama */
[data-testid="stAppViewContainer"] {{
    background: linear-gradient({bg_overlay}),
                url('{bg_image_url}');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    animation: zoomEffect 25s infinite alternate;
}}

/* Container Utama */
.block-container {{
    max-width: 1000px;
    padding-top: 2rem;
    padding-bottom: 2rem;
    background-color: {container_bg};
    backdrop-filter: blur(20px);
    border-radius: 25px;
    box-shadow: 0 15px 50px rgba(0,0,0,0.3);
    border: 1px solid {border_color};
    position: relative;
    z-index: 10;
}}

/* Warna Teks Global */
p, h1, h2, h3, h4, li, span, div, label {{
    color: {text_color} !important;
}}

/* Header Box */
.header-box {{
    position: relative;
    background: linear-gradient(135deg, {header_gradient});
    padding: 40px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 30px;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 10px 25px {shadow_color};
}}

.theme-toggle-btn {{
    position: absolute;
    top: 15px;
    right: 15px;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    padding: 8px 16px;
    border-radius: 20px;
    cursor: pointer;
    font-size: 14px;
    backdrop-filter: blur(4px);
    transition: 0.3s;
    z-index: 20;
}}
.theme-toggle-btn:hover {{
    background: rgba(255,255,255,0.3);
}}
.header-title {{
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 10px;
    text-shadow: 0 2px 5px rgba(0,0,0,0.1);
    line-height: 1.2;
    color: {header_text_color} !important;
}}
.header-subtitle {{
    font-size: 16px;
    opacity: 0.85;
    font-weight: 400;
    color: {header_text_color} !important;
}}

/* ========== 3 KOLOM INFO CARD ========== */
.info-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.2rem;
    margin-bottom: 2rem;
}}
.info-card {{
    background-color: {box_bg};
    padding: 1.2rem;
    border-radius: 15px;
    border: 1px solid {border_color};
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    text-align: center;
}}
.info-card h3 {{
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
    letter-spacing: 0.5px;
}}
.info-card p {{
    font-size: 0.8rem;
    line-height: 1.5;
    margin-bottom: 0.8rem;
}}

.category-row {{
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    margin: 0.6rem 0;
}}
.category-badge {{
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    background: rgba(0, 119, 182, 0.12);
    border: 1px solid rgba(0, 119, 182, 0.25);
    color: {text_color};
}}

.tech-row {{
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    margin: 0.6rem 0;
    flex-wrap: wrap;
}}
.tech-badge {{
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    font-size: 0.65rem;
    font-weight: 600;
    background: rgba(0, 119, 182, 0.12);
    border: 1px solid rgba(0, 119, 182, 0.25);
    color: {text_color};
}}

@media (max-width: 700px) {{
    .info-grid {{
        grid-template-columns: 1fr;
        gap: 1rem;
    }}
}}

/* Tabs - Rata Tengah */
.stTabs {{
    width: 100%;
}}
.stTabs [data-baseweb="tab-list"] {{
    background-color: {box_bg};
    border-radius: 40px;
    padding: 6px;
    margin-bottom: 25px;
    border: 1px solid {border_color};
    display: flex;
    justify-content: center;
    width: fit-content;
    margin-left: auto;
    margin-right: auto;
    gap: 0.5rem;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent;
    color: {text_color};
    border-radius: 30px;
    padding: 10px 28px;
    font-weight: 600;
    font-size: 0.9rem;
    white-space: nowrap;
}}
.stTabs [aria-selected="true"] {{
    background: {tab_active_bg} !important;
    color: white !important;
    box-shadow: 0 2px 10px rgba(0, 119, 182, 0.4);
}}

/* Upload Area */
[data-testid="stFileUploader"] {{
    width: 100%;
}}
[data-testid="stFileUploader"] > div {{
    background: {upload_bg} !important;
    border-radius: 20px !important;
    border: 2px dashed {upload_border} !important;
    padding: 2rem !important;
    text-align: center !important;
    transition: all 0.3s !important;
}}
[data-testid="stFileUploader"] > div:hover {{
    background: {upload_hover_bg} !important;
}}
[data-testid="stFileUploader"] div p {{
    color: {upload_text_color} !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}}
[data-testid="stFileUploader"] small {{
    color: {upload_subtext_color} !important;
    font-size: 12px !important;
}}
[data-testid="stFileUploader"] button {{
    background: {upload_button_bg} !important;
    color: {upload_button_text} !important;
    border: 1px solid {upload_button_border} !important;
    border-radius: 12px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
}}

/* Camera Input */
[data-testid="stCameraInput"] > div {{
    background: {upload_bg} !important;
    border-radius: 20px !important;
    border: 2px dashed {upload_border} !important;
    padding: 1rem !important;
}}
[data-testid="stCameraInput"] button {{
    background: {upload_button_bg} !important;
    color: {upload_button_text} !important;
    border: 1px solid {upload_button_border} !important;
}}

/* Preview Card Gambar */
.preview-title {{
    text-align: center;
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1rem;
}}
.image-preview-container {{
    background-color: {card_bg};
    border: 2px solid {border_color};
    border-radius: 20px;
    padding: 20px;
    margin-bottom: 25px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    text-align: center;
}}
.image-frame {{
    width: 100%;
    height: auto;
    min-height: 300px;
    display: flex;
    justify-content: center;
    align-items: center;
    overflow: hidden;
    border-radius: 12px;
    background-color: rgba(0,0,0,0.05);
}}
.image-frame img {{
    width: 100%;
    height: auto;
    object-fit: contain;
    display: block;
}}

/* Button Styling */
.stButton > button {{
    width: 100%;
    height: 55px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 700;
    border: none;
    background: linear-gradient(90deg, #0077b6, #00b4d8);
    color: white;
    transition: all 0.3s;
    box-shadow: 0 4px 15px rgba(0, 180, 216, 0.3);
}}
.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 180, 216, 0.5);
}}

/* Result Box */
.result-box {{
    background-color: {box_bg};
    padding: 20px;
    border-radius: 15px;
    margin: 15px 0;
    border: 1px solid {border_color};
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}}

/* Metric */
[data-testid="stMetricValue"] {{
    color: #38bdf8 !important;
    font-size: 40px;
}}

/* Footer */
.footer {{
    text-align: center;
    margin-top: 50px;
    font-size: 14px;
    color: {text_color};
    opacity: 0.7;
}}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 5. HEADER (DENGAN TOMBOL TEMA)
# ==========================================
st.markdown(f"""
<div class="header-box">
    <button class="theme-toggle-btn" onclick="parent.document.querySelector('.stButton button').click()">🌓 Light/Dark</button>
    <div class="header-title">🪸 Coral Bleaching Detection</div>
    <div class="header-subtitle">Sistem Deteksi Dini Pemutihan Terumbu Karang menggunakan CNN EfficientNetB0</div>
</div>
""", unsafe_allow_html=True)

# Tombol tersembunyi untuk trigger Javascript
if st.button("toggle_theme_hidden", key="theme_trigger"):
    toggle_theme()

# ==========================================
# 6. 3 KOLOM INFO
# ==========================================
st.markdown("""
<div class="info-grid">
    <div class="info-card">
        <h3>🌊 ECOLOGICAL INTELLIGENCE</h3>
        <p>Deteksi dini pemutihan karang berbasis CNN. Akurasi tinggi & respons cepat.</p>
    </div>
    <div class="info-card">
        <h3>🎯 KLASIFIKASI PREMIUM</h3>
        <div class="category-row">
            <span class="category-badge">Healthy</span>
            <span class="category-badge">Bleached</span>
            <span class="category-badge">Non-Coral</span>
        </div>
        <p>3 kategori presisi dengan transfer learning.</p>
    </div>
    <div class="info-card">
        <h3>⚡ CORE ENGINE</h3>
        <div class="tech-row">
            <span class="tech-badge">EfficientNetB0</span>
            <span class="tech-badge">Edge AI</span>
            <span class="tech-badge">TensorFlow Lite</span>
        </div>
        <p>CNN ringan & optimal untuk citra bawah air.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 7. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    model_path = "model_terumbu_karang_224.tflite"
    if not os.path.exists(model_path):
        return None
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        return None

interpreter = load_model()
if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

# ==========================================
# 8. PREPROCESSING
# ==========================================
IMG_SIZE = 224
def preprocess_image(image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def get_image_base64(image, format="JPEG"):
    """Convert PIL Image to base64 string with 100% quality (no compression)"""
    buffered = BytesIO()
    # Simpan dengan kualitas 100% (tanpa kompresi)
    if format == "JPEG" or format == "JPG":
        image.save(buffered, format="JPEG", quality=100, optimize=False, subsampling=0)
    else:
        image.save(buffered, format="PNG", optimize=False, compress_level=0)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# ==========================================
# 9. INPUT GAMBAR
# ==========================================
tab1, tab2 = st.tabs(["📂 Upload Gambar", "📷 Kamera"])

uploaded_file = None

with tab1:
    file_input = st.file_uploader("Pilih gambar terumbu karang", type=["jpg", "jpeg", "png"])
    if file_input:
        uploaded_file = file_input

with tab2:
    camera_input = st.camera_input("Ambil Foto")
    if camera_input:
        uploaded_file = camera_input

# ==========================================
# 10. PREVIEW & PREDIKSI
# ==========================================
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Tentukan format asli gambar
    file_extension = uploaded_file.name.split('.')[-1].upper() if hasattr(uploaded_file, 'name') else "JPEG"
    if file_extension in ['JPG', 'JPEG']:
        img_format = "JPEG"
    else:
        img_format = "PNG"
    
    # Konversi gambar ke base64 dengan kualitas 100%
    img_base64 = get_image_base64(image, img_format)
    mime_type = "image/jpeg" if img_format == "JPEG" else "image/png"
    
    st.markdown('<div class="preview-title">Preview Gambar</div>', unsafe_allow_html=True)
    st.markdown(f'''
    <div class="image-preview-container">
        <div class="image-frame">
            <img src="data:{mime_type};base64,{img_base64}" alt="Preview gambar" style="max-width:100%; height:auto;">
        </div>
    </div>
    ''', unsafe_allow_html=True)

    if st.button("Analisis Sekarang", type="primary"):
        if interpreter is None:
            st.error("Model tidak ditemukan.")
        else:
            with st.spinner("Sedang menganalisis citra..."):
                img_tensor = preprocess_image(image)
                interpreter.set_tensor(input_details[0]["index"], img_tensor.astype(input_details[0]["dtype"]))
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]["index"])
                
                classes = ['Bleached', 'Healthy', 'Not_Coral'] 
                predicted_index = np.argmax(output[0])
                label = classes[predicted_index]
                confidence = float(np.max(output[0])) * 100

                st.markdown("## Hasil Analisis")
                
                col_a, col_b = st.columns(2)
                col_a.metric("Status Karang", label)
                col_b.metric("Confidence", f"{confidence:.2f}%")

                if label == 'Healthy':
                    st.success(f"Status: **{label} Coral**")
                    penjelasan = (
                    "Sistem mengidentifikasi objek sebagai Healthy Coral. "
                    "Hasil ini menunjukkan bahwa citra memiliki karakteristik visual karang sehat, "
                    "seperti warna yang masih normal, tekstur yang jelas, dan struktur karang yang terjaga."
                    )

                elif label == 'Bleached':
                    st.warning(f"Status: **{label} Coral**")
                    penjelasan = (
                    "Sistem mengidentifikasi objek sebagai Bleached Coral. "
                    "Karang yang mengalami pemutihan umumnya ditandai dengan dominasi warna putih atau pucat "
                    "akibat berkurangnya alga simbiotik (zooxanthellae) yang berperan penting dalam menjaga kesehatan karang."
                    )

                else:
                    st.info("Status: **Not Coral**")
                    penjelasan = (
                    "Sistem mengidentifikasi objek sebagai Not Coral. "
                    "Objek pada citra tidak memiliki karakteristik visual yang sesuai dengan kategori "
                    "Healthy Coral maupun Bleached Coral sehingga diklasifikasikan sebagai bukan terumbu karang."
                    )

                st.write(penjelasan)
                
                st.markdown("#### Distribusi Probabilitas Model")
                df_prob = pd.DataFrame({'Kategori': classes, 'Probabilitas': output[0]})
                st.bar_chart(df_prob.set_index('Kategori'))

                html_result = f"""
                <div class="result-box">
                    <p><b>Analisis Teknis:</b><br>{penjelasan}</p>
                </div>
                """
                st.markdown(html_result, unsafe_allow_html=True)
                
                with st.expander("Mengapa hasil ini muncul?"):
                    st.write("""
                    Hasil prediksi diperoleh berdasarkan proses ekstraksi fitur visual oleh model Deep Learning yang telah dilatih untuk membedakan kategori Healthy Coral, Bleached Coral, dan Not Coral.
                    Jika hasil klasifikasi kurang sesuai, beberapa faktor berikut dapat memengaruhi akurasi prediksi:

                    - Kualitas Citra : Gambar yang buram (blur), memiliki resolusi rendah, atau objek karang yang tidak terlihat jelas dapat mengurangi kemampuan model dalam mengenali fitur penting.
                    - Pencahayaan    : Kondisi pencahayaan yang terlalu gelap maupun terlalu terang (overexposure) dapat mengubah warna dan tekstur objek sehingga berpotensi menyebabkan kesalahan klasifikasi.
                    - Kemiripan Objek: Beberapa objek non-karang memiliki karakteristik visual yang menyerupai terumbu karang sehingga dapat memengaruhi hasil prediksi.
                    - Kondisi Lingkungan Bawah Air: Dominasi warna biru dari air laut, keberadaan ikan atau biota laut lainnya, serta gangguan visual di sekitar objek dapat memengaruhi proses identifikasi yang dilakukan oleh model.
                     """)


# ==========================================
# 11. FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Coral Bleaching Detection System</p>
    <p>Universitas Maritim Raja Ali Haji (UMRAH) | Barokah United</p>
</div>
""", unsafe_allow_html=True)