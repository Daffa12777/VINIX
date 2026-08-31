"""
🛣️  Deteksi Kerusakan Jalan — Streamlit App
Implementasi Model Computer Vision (YOLO) untuk Deteksi Kerusakan
Permukaan Jalan pada Infrastruktur Pintar — SDG 9.
"""
from pathlib import Path
import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ----------------------------------------------------------------------
# Konfigurasi halaman
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Deteksi Kerusakan Jalan",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("models/best.pt")
CLASS_NAMES = {0: "Pothole", 1: "Crack", 2: "Manhole"}
CLASS_LABEL_ID = {0: "Lubang", 1: "Retakan", 2: "Manhole"}
CLASS_COLORS = {0: "#e63946", 1: "#f4a261", 2: "#2a9d8f"}  # merah / oranye / teal

# ----------------------------------------------------------------------
# Styling minimalis
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1200px;}
      #MainMenu, footer {visibility: hidden;}
      h1, h2, h3 {letter-spacing: -0.02em;}
      .hero {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b4965 100%);
        padding: 1.6rem 1.8rem; border-radius: 16px; color: #fff; margin-bottom: 1.4rem;
      }
      .hero h1 {margin: 0; font-size: 1.7rem; color: #fff;}
      .hero p  {margin: .35rem 0 0; opacity: .85; font-size: .95rem;}
      .stat-card {
        background: #fff; border: 1px solid #e8ecef; border-radius: 14px;
        padding: 1rem 1.2rem; text-align: center;
      }
      .stat-num {font-size: 1.9rem; font-weight: 700; line-height: 1;}
      .stat-lbl {font-size: .8rem; color: #6b7280; margin-top: .3rem;}
      .badge {display:inline-block; padding:.15rem .6rem; border-radius:999px;
              font-size:.78rem; color:#fff; font-weight:600;}
      div[data-testid="stFileUploader"] label {font-weight:600;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Muat model (cached)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model YOLO…")
def load_model(path: str):
    from ultralytics import YOLO
    return YOLO(path)


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🛣️ Deteksi Kerusakan Jalan</h1>
      <p>Computer Vision (YOLO) untuk deteksi lubang, retakan & manhole —
         mendukung infrastruktur pintar (SDG 9).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Cek ketersediaan model
if not MODEL_PATH.exists():
    st.error(
        "⚠️ Model **`models/best.pt`** belum ditemukan.\n\n"
        "Latih model lewat notebook `road_damage_yolo_training.ipynb`, "
        "lalu letakkan file `best.pt` ke dalam folder `models/`."
    )
    st.stop()

model = load_model(str(MODEL_PATH))

# ----------------------------------------------------------------------
# Sidebar — pengaturan
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Pengaturan")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                     help="Ambang keyakinan minimum untuk menampilkan deteksi.")
    iou = st.slider("IoU threshold", 0.30, 0.90, 0.45, 0.05,
                    help="Ambang tumpang-tindih untuk Non-Max Suppression.")

    st.subheader("Filter kelas")
    active = {
        cid: st.checkbox(f"{CLASS_NAMES[cid]} ({CLASS_LABEL_ID[cid]})", value=True)
        for cid in CLASS_NAMES
    }

    st.divider()
    st.caption("Kelas terdeteksi:")
    for cid, name in CLASS_NAMES.items():
        st.markdown(
            f'<span class="badge" style="background:{CLASS_COLORS[cid]}">{name}</span>',
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("VINIX7 — Divisi AI · SDG 9")

# ----------------------------------------------------------------------
# Input gambar: upload atau contoh
# ----------------------------------------------------------------------
st.subheader("1 · Pilih gambar")

col_up, col_sample = st.columns([2, 1])
image = None

with col_up:
    uploaded = st.file_uploader(
        "Unggah gambar jalan (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")

with col_sample:
    st.write("atau coba contoh:")
    sample_dir = Path("sample_images")
    samples = sorted(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
    for sp in samples:
        if st.button(f"📷 {sp.stem}", use_container_width=True):
            image = Image.open(sp).convert("RGB")

# ----------------------------------------------------------------------
# Inferensi + hasil
# ----------------------------------------------------------------------
if image is not None:
    st.subheader("2 · Hasil deteksi")

    classes_to_show = [cid for cid, on in active.items() if on]
    with st.spinner("Menganalisis gambar…"):
        result = model.predict(
            np.array(image),
            conf=conf,
            iou=iou,
            classes=classes_to_show if classes_to_show else None,
            verbose=False,
        )[0]

    annotated = result.plot()[..., ::-1]  # BGR -> RGB

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Gambar asli")
        st.image(image, use_container_width=True)
    with c2:
        st.caption("Hasil deteksi")
        st.image(annotated, use_container_width=True)

    # --- ringkasan deteksi ---
    rows = []
    for box in result.boxes:
        cid = int(box.cls)
        rows.append(
            {
                "Kelas": CLASS_NAMES.get(cid, str(cid)),
                "Keterangan": CLASS_LABEL_ID.get(cid, "-"),
                "Confidence": round(float(box.conf), 3),
            }
        )
    df = pd.DataFrame(rows)

    st.subheader("3 · Ringkasan")
    total = len(df)
    counts = df["Kelas"].value_counts().to_dict() if total else {}

    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{total}</div>'
            f'<div class="stat-lbl">Total deteksi</div></div>',
            unsafe_allow_html=True,
        )
    for i, cid in enumerate(CLASS_NAMES):
        name = CLASS_NAMES[cid]
        with cols[i + 1]:
            st.markdown(
                f'<div class="stat-card"><div class="stat-num" '
                f'style="color:{CLASS_COLORS[cid]}">{counts.get(name, 0)}</div>'
                f'<div class="stat-lbl">{name}</div></div>',
                unsafe_allow_html=True,
            )

    if total:
        st.dataframe(df, use_container_width=True, hide_index=True)

        # tombol unduh gambar hasil
        buf = io.BytesIO()
        Image.fromarray(annotated).save(buf, format="JPEG")
        st.download_button(
            "⬇️ Unduh gambar hasil",
            data=buf.getvalue(),
            file_name="hasil_deteksi.jpg",
            mime="image/jpeg",
        )
    else:
        st.info("Tidak ada kerusakan terdeteksi pada ambang saat ini. "
                "Coba turunkan *confidence threshold*.")
else:
    st.info("⬆️ Unggah gambar atau pilih salah satu contoh untuk memulai.")
