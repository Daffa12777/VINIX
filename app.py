"""
Deteksi Kerusakan Jalan - Streamlit App
Computer Vision (YOLO) untuk deteksi Pothole / Crack / Manhole,
dilengkapi analisis & rekomendasi otomatis dari LLM (Gemini).
Infrastruktur pintar - SDG 9.
"""
import os
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ----------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Deteksi Kerusakan Jalan",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("models/best.pt")
CLASS_NAMES = {0: "Pothole", 1: "Crack", 2: "Manhole"}
CLASS_LABEL_ID = {0: "Lubang", 1: "Retakan", 2: "Tutup gorong-gorong"}
CLASS_COLORS = {0: "#C0392B", 1: "#D98A2B", 2: "#7A7A72"}

# Model Gemini. Ganti bila akun tak punya akses:
#   "gemini-2.5-flash" (stabil), "gemini-flash-latest", "gemini-3.6-flash"
GEMINI_MODEL = "gemini-2.5-flash"

# Nama manual acuan (dipakai di rekomendasi & catatan)
MANUAL_REF = "Manual Bina Marga No. 001-02/M/BM/2011"

# ----------------------------------------------------------------------
# Style (Inter, palet merah/cream/abu, 3D + animasi)
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
:root{
  --cream:#f6efe4; --cream2:#efe6d6; --paper:#fffdf9; --card:#ffffff;
  --red:#c0392b; --red-deep:#9c2a20; --red-soft:#e06055;
  --ink:#2c2622; --muted:#8a8178; --muted2:#b3aa9d;
  --line:#e7ddcc; --line2:#f0e8da; --gray:#7a7a72; --graybg:#eceae4;
  --eo:cubic-bezier(.22,1,.36,1);
}
.stApp{
  background:
    radial-gradient(120% 90% at 88% -8%, #f9f2e7 0%, transparent 45%),
    radial-gradient(120% 90% at -8% 8%, #f3e3df 0%, transparent 42%),
    var(--cream);
}
html,body,[class*="css"],.stApp,input,button,select,textarea{
  font-family:'Inter',system-ui,sans-serif !important; color:var(--ink);
}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{visibility:hidden; height:0;}
header[data-testid="stHeader"]{background:transparent;}
/* sidebar selalu tampil + tombol buka/tutup terlihat */
section[data-testid="stSidebar"]{visibility:visible !important;}
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"]{
  visibility:visible !important; opacity:1 !important; z-index:1000 !important;}
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stExpandSidebarButton"] button{
  background:var(--red) !important; color:#fff !important; border:none !important;
  box-shadow:0 8px 18px -8px rgba(140,32,24,.6) !important;}
[data-testid="stSidebarCollapsedControl"] button *,
[data-testid="stExpandSidebarButton"] button *{color:#fff !important; fill:#fff !important;}
::selection{background:var(--red); color:#fff;}
.block-container{padding-top:1.4rem; padding-bottom:3rem; max-width:1180px;
  animation:pageUp .8s var(--eo) both;}
@keyframes pageUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}
h1,h2,h3,h4{color:var(--ink); letter-spacing:-.02em;}
p,span,label,li,.stMarkdown{color:var(--ink);}
hr{border:none; border-top:1px solid var(--line) !important; margin:1.5rem 0;}
@keyframes rise{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}
@keyframes pop{from{opacity:0;transform:scale(.9)}to{opacity:1;transform:scale(1)}}
@keyframes floaty{0%,100%{transform:translateY(0)}50%{transform:translateY(-9px)}}
@keyframes shimmer{0%{transform:translateX(-120%)}100%{transform:translateX(220%)}}

/* ---------------- HERO (3D) ---------------- */
.hero{
  position:relative; overflow:hidden; border-radius:26px; color:#fff;
  padding:2.6rem 2.4rem 2.4rem; margin-bottom:1.8rem;
  background:linear-gradient(135deg,#a8281d 0%, var(--red) 42%, #cf5647 100%);
  box-shadow:0 40px 70px -34px rgba(140,32,24,.65),
             0 8px 22px -12px rgba(140,32,24,.5),
             inset 0 1px 0 rgba(255,255,255,.22);
  transform:translateZ(0); animation:rise .9s var(--eo) both;
}
.hero::before{content:""; position:absolute; top:-40%; right:-10%; width:420px; height:420px;
  border-radius:50%; background:radial-gradient(circle, rgba(255,255,255,.22), transparent 62%);
  animation:floaty 7s ease-in-out infinite; pointer-events:none;}
.hero::after{content:""; position:absolute; top:0; left:0; width:60%; height:100%;
  background:linear-gradient(105deg, transparent, rgba(255,255,255,.16), transparent);
  transform:skewX(-18deg); animation:shimmer 6.5s var(--eo) infinite; pointer-events:none;}
.hero .eyebrow{position:relative; font-size:.76rem; font-weight:700; letter-spacing:.16em;
  text-transform:uppercase; color:rgba(255,236,230,.92);}
.hero h1{position:relative; margin:.55rem 0 0; font-size:2.15rem; font-weight:900;
  color:#fff; line-height:1.05; text-shadow:0 2px 14px rgba(120,24,18,.4);}
.hero p{position:relative; margin:.7rem 0 0; max-width:62ch; font-size:1rem; line-height:1.6;
  color:rgba(255,244,240,.9);}
.hbadges{position:relative; display:flex; gap:9px; margin-top:1.3rem; flex-wrap:wrap;}
.hbadge{background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.28);
  color:#fff; font-size:.76rem; font-weight:600; padding:7px 14px; border-radius:999px;
  backdrop-filter:blur(4px); box-shadow:0 6px 14px -8px rgba(120,24,18,.5);}

/* ---------------- section label ---------------- */
.kick{color:var(--red); font-size:.78rem; font-weight:800; letter-spacing:.12em;
  text-transform:uppercase; margin:.2rem 0 .1rem;}
.sub{color:var(--muted); font-size:.95rem; margin:0 0 .7rem;}

/* ---------------- stat cards (3D) ---------------- */
.stat-card{position:relative; background:linear-gradient(180deg,var(--paper),#fbf6ee);
  border:1px solid var(--line); border-radius:18px; padding:1.15rem 1rem; text-align:center;
  box-shadow:0 22px 40px -30px rgba(120,60,30,.4), inset 0 1px 0 #fff;
  transition:transform .25s var(--eo), box-shadow .25s var(--eo);
  animation:rise .6s var(--eo) both;}
.stat-card:hover{transform:translateY(-5px);
  box-shadow:0 30px 52px -28px rgba(120,60,30,.5), inset 0 1px 0 #fff;}
.stat-num{font-size:2.1rem; font-weight:900; line-height:1; letter-spacing:-.03em;}
.stat-lbl{font-size:.76rem; color:var(--muted); margin-top:.5rem; font-weight:600;
  text-transform:uppercase; letter-spacing:.04em;}

.badge{display:inline-block; padding:.2rem .7rem; border-radius:999px;
  font-size:.75rem; color:#fff; font-weight:700; box-shadow:0 4px 10px -5px rgba(0,0,0,.4);}

/* ---------------- AI card (3D) ---------------- */
.ai-card{position:relative; background:var(--card); border:1px solid var(--line);
  border-radius:20px; padding:1.3rem 1.5rem; overflow:hidden;
  box-shadow:0 30px 56px -34px rgba(120,60,30,.45), inset 0 1px 0 #fff;
  animation:rise .6s var(--eo) both;}
.ai-card::before{content:""; position:absolute; left:0; top:0; bottom:0; width:5px;
  background:linear-gradient(180deg,var(--red),var(--red-soft));}
.ai-badge{display:inline-block; background:linear-gradient(135deg,var(--red),var(--red-deep));
  color:#fff; font-size:.72rem; font-weight:800; padding:5px 13px; border-radius:999px;
  letter-spacing:.03em; box-shadow:0 8px 16px -8px rgba(140,32,24,.6);}
.ai-badge.auto{background:linear-gradient(135deg,#9a938a,#7c766e);}
.ai-body{margin:.8rem 0 0; font-size:1rem; line-height:1.75; color:var(--ink); white-space:pre-wrap;}

/* analisis terstruktur */
.ins-head{display:flex; align-items:center; gap:12px; margin-bottom:1rem; flex-wrap:wrap;}
.ins-title{font-weight:900; font-size:1.2rem; letter-spacing:-.01em;}
.ins-row{display:flex; gap:14px; padding:.8rem 0; border-top:1px solid var(--line2);
  animation:rise .5s var(--eo) both;}
.ins-num{width:30px; height:30px; flex:0 0 30px; border-radius:9px; display:grid;
  place-items:center; font-weight:800; font-size:.85rem; color:#fff;
  background:linear-gradient(135deg,var(--red),var(--red-deep));
  box-shadow:0 8px 16px -8px rgba(140,32,24,.55), inset 0 1px 0 rgba(255,255,255,.35);}
.ins-num.g{background:linear-gradient(135deg,#8a857c,#635f58);
  box-shadow:0 8px 16px -8px rgba(70,66,60,.5), inset 0 1px 0 rgba(255,255,255,.3);}
.ins-label{font-weight:800; color:var(--red); font-size:.95rem; margin-bottom:.15rem;
  letter-spacing:.01em;}
.ins-label.g{color:#6c665e;}
.ins-text{color:var(--ink); font-size:.95rem; line-height:1.6;}

.note{background:linear-gradient(180deg,#fbf4e9,#f6ecdb); border:1px solid var(--line);
  border-radius:14px; padding:.9rem 1.1rem; font-size:.9rem; color:#7c6a52; line-height:1.6;
  box-shadow:inset 0 1px 0 #fff;}
.disc{margin-top:1rem; font-size:.8rem; color:var(--muted2); line-height:1.6;}

/* ---------------- edu / benefit cards ---------------- */
.grid3{display:grid; grid-template-columns:repeat(3,1fr); gap:18px;}
.grid4{display:grid; grid-template-columns:repeat(4,1fr); gap:16px;}
@media(max-width:820px){.grid3,.grid4{grid-template-columns:1fr 1fr}}
@media(max-width:520px){.grid3,.grid4{grid-template-columns:1fr}}
.ecard{position:relative; background:var(--card); border:1px solid var(--line);
  border-radius:18px; padding:1.3rem 1.25rem; overflow:hidden;
  box-shadow:0 24px 44px -32px rgba(120,60,30,.42), inset 0 1px 0 #fff;
  transition:transform .28s var(--eo), box-shadow .28s var(--eo);
  animation:rise .6s var(--eo) both;}
.ecard:hover{transform:translateY(-6px);
  box-shadow:0 36px 60px -30px rgba(120,60,30,.55), inset 0 1px 0 #fff;}
.enum{width:42px; height:42px; border-radius:13px; display:grid; place-items:center;
  font-weight:900; font-size:1.05rem; color:#fff; margin-bottom:.9rem;
  background:linear-gradient(135deg,var(--red),var(--red-deep));
  box-shadow:0 12px 22px -10px rgba(140,32,24,.6), inset 0 1px 0 rgba(255,255,255,.35);}
.enum.g{background:linear-gradient(135deg,#8a857c,#635f58);
  box-shadow:0 12px 22px -10px rgba(70,66,60,.55), inset 0 1px 0 rgba(255,255,255,.3);}
.etitle{font-size:1.05rem; font-weight:800; margin:0 0 .35rem; letter-spacing:-.01em;}
.etext{font-size:.92rem; color:var(--muted); line-height:1.6; margin:0;}
.chip-legend{display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:7px;
  vertical-align:middle; box-shadow:0 2px 5px -2px rgba(0,0,0,.4);}

/* ---------------- buttons (3D, teks kontras) ---------------- */
.stButton>button, .stDownloadButton>button{
  width:100%; border-radius:14px; font-weight:700; padding:.72rem 1rem; font-size:.96rem;
  border:none !important;
  background:linear-gradient(135deg,var(--red) 0%, var(--red-deep) 100%) !important;
  box-shadow:0 16px 28px -14px rgba(140,32,24,.7), inset 0 1px 0 rgba(255,255,255,.28) !important;
  transition:transform .15s var(--eo), box-shadow .22s var(--eo), filter .22s ease;
}
/* paksa SEMUA teks di dalam tombol jadi putih (fix teks tak terlihat) */
.stButton>button, .stButton>button *,
.stButton>button p, .stButton>button div, .stButton>button span,
.stDownloadButton>button, .stDownloadButton>button *,
.stDownloadButton>button p, .stDownloadButton>button div, .stDownloadButton>button span{
  color:#ffffff !important; fill:#ffffff !important;
}
.stButton>button:hover, .stDownloadButton>button:hover{
  transform:translateY(-2px); filter:brightness(1.05);
  box-shadow:0 22px 34px -14px rgba(140,32,24,.8), inset 0 1px 0 rgba(255,255,255,.3) !important;
}
.stButton>button:active, .stDownloadButton>button:active{transform:translateY(0) scale(.99);}

/* inputs & media */
div[data-testid="stFileUploader"] label{font-weight:600;}
[data-testid="stImage"] img{border-radius:16px;
  box-shadow:0 26px 48px -30px rgba(120,60,30,.5); animation:pop .6s var(--eo) both;}
[data-testid="stDataFrame"]{border-radius:14px; overflow:hidden;
  box-shadow:0 22px 44px -34px rgba(120,60,30,.4);}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#fbf5ea,#f3e9d8);
  border-right:1px solid var(--line);}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Model YOLO (cached)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model YOLO...")
def load_model(path: str):
    from ultralytics import YOLO
    return YOLO(path)


# ----------------------------------------------------------------------
# Integrasi LLM (Gemini) - aman, selalu ada fallback
# ----------------------------------------------------------------------
def _get_api_key():
    """Ambil API key dari st.secrets atau environment variable."""
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


def _fallback_insight(total, counts, avg_conf):
    """Ringkasan berbasis aturan bila LLM tidak tersedia."""
    if total == 0:
        return ("Tidak ada kerusakan terdeteksi pada ambang keyakinan saat ini. "
                "Ini belum tentu berarti jalan mulus - coba turunkan confidence "
                "threshold, atau gunakan gambar dengan sudut dan pencahayaan yang "
                "lebih baik untuk verifikasi manual.")

    pot = counts.get("Pothole", 0)
    cra = counts.get("Crack", 0)
    man = counts.get("Manhole", 0)

    if pot >= 3:
        sev = "tinggi"
    elif pot >= 1 or cra >= 4:
        sev = "sedang"
    else:
        sev = "rendah"

    rincian = []
    if pot:
        rincian.append(f"{pot} lubang (pothole)")
    if cra:
        rincian.append(f"{cra} retakan (crack)")
    if man:
        rincian.append(f"{man} tutup gorong-gorong (manhole)")
    rincian_txt = ", ".join(rincian)

    # prioritas
    if pot:
        prioritas = ("Lubang (pothole) harus didahulukan karena berdampak langsung pada "
                     "keselamatan; retakan menyusul untuk mencegah pelebaran.")
    elif cra:
        prioritas = ("Retakan (crack) menjadi fokus utama agar tidak berkembang menjadi "
                     "lubang; tangani lebih awal selagi masih ringan.")
    else:
        prioritas = ("Tidak ada kerusakan struktural mendesak; fokus pada pemantauan rutin "
                     "dan pemeriksaan kelengkapan manhole.")

    # rekomendasi per jenis (mengacu Manual Bina Marga)
    rek = []
    if pot:
        rek.append("Lubang: lakukan penambalan lubang sesuai Metode P5 - gali hingga "
                   "lapisan keras, isi agregat kelas A berlapis, laburkan prime coat, "
                   "tebar campuran aspal dingin, lalu padatkan hingga rata.")
    if cra:
        rek.append("Retakan: untuk retak halus (<2 mm) gunakan Metode P3 (penutupan retak) "
                   "dengan campuran aspal emulsi dan pasir kasar; untuk retak lebih lebar "
                   "(>2 mm) gunakan Metode P4 (pengisian retak).")
    if man:
        rek.append("Manhole: pastikan elevasi tutup rata dengan permukaan aspal; sesuaikan "
                   "bila menonjol atau ambles. Manhole bukan kerusakan perkerasan.")
    if not rek:
        rek.append("Lakukan inspeksi lanjutan dan dokumentasi berkala untuk memantau kondisi.")
    rek_txt = " ".join(rek)

    # urgensi
    if sev == "tinggi":
        urg = "Sebaiknya ditangani segera (prioritas cepat) mengingat jumlah lubang cukup banyak."
    elif sev == "sedang":
        urg = "Penanganan dalam beberapa minggu ke depan disarankan sebelum kerusakan meluas."
    else:
        urg = "Cukup dipantau berkala; belum memerlukan tindakan darurat."

    return (
        f"Kondisi Umum: Terdeteksi {total} objek pada gambar - {rincian_txt}.\n"
        f"Tingkat Keparahan: Diperkirakan {sev}, dengan rata-rata keyakinan model "
        f"{avg_conf*100:.0f}%.\n"
        f"Prioritas Penanganan: {prioritas}\n"
        f"Rekomendasi Tindakan: {rek_txt}\n"
        f"Estimasi Urgensi: {urg}\n"
        f"Catatan Keselamatan: Pasang rambu atau marka sementara di sekitar kerusakan parah "
        f"untuk melindungi pengguna jalan hingga perbaikan selesai.\n"
        f"Keterbatasan: Ini ringkasan otomatis berbasis aturan. Akurasi model masih terbatas, "
        f"sehingga hasil perlu diverifikasi petugas sebelum dijadikan dasar perbaikan.\n"
        f"Rujukan: Metode perbaikan mengacu pada {MANUAL_REF}."
    )


# ----------------------------------------------------------------------
# Referensi dari PDF (dibaca langsung saat runtime, verbatim/lengkap)
# ----------------------------------------------------------------------
# Letakkan file PDF manual di salah satu lokasi berikut di dalam repo.
def _find_pdf():
    candidates = [
        Path("reference/manual_binamarga.pdf"),
        Path("manual_binamarga.pdf"),
        Path("reference/no-001-02mbm2011-manual-perbaikan-standar-untuk-pemeliharaan-rutin-jalan.pdf"),
        Path("no-001-02mbm2011-manual-perbaikan-standar-untuk-pemeliharaan-rutin-jalan.pdf"),
    ]
    for c in candidates:
        if c.exists():
            return c
    refdir = Path("reference")
    if refdir.exists():
        pdfs = sorted(refdir.glob("*.pdf"))
        if pdfs:
            return pdfs[0]
    return None


# Penanda bagian metode di manual, dipetakan ke kelas deteksi.
METODE_MARKERS = {
    "Crack": ["Metode Perbaikan P3", "Metode Perbaikan P4"],
    "Pothole": ["Metode Perbaikan P5"],
    "Manhole": [],  # tidak ada metode P-standar khusus pada manual
}


@st.cache_data(show_spinner=False)
def _load_pdf_text(path_str):
    try:
        from pypdf import PdfReader
        reader = PdfReader(path_str)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        return ""


def _extract_section(text, marker, maxlen=4000):
    """Ambil isi bagian metode (verbatim) dari penanda sampai metode berikutnya."""
    positions = [m.start() for m in re.finditer(re.escape(marker), text)]
    if not positions:
        return ""
    idx = positions[-1]  # kemunculan terakhir = isi metode (bukan daftar isi)
    rest = text[idx:]
    nxt = re.search(r"Metode Perbaikan [PUK]\s?\d", rest[len(marker):])
    end = len(marker) + nxt.start() if nxt else min(len(rest), maxlen)
    return rest[:min(end, maxlen)].strip()


def referensi_pdf(counts):
    """Kumpulkan kutipan LENGKAP metode dari PDF sesuai kelas yang terdeteksi."""
    pdf = _find_pdf()
    if pdf is None:
        return ""
    full = _load_pdf_text(str(pdf))
    if not full:
        return ""
    blocks, seen = [], set()
    for kelas in ("Pothole", "Crack", "Manhole"):
        if counts.get(kelas, 0) <= 0:
            continue
        for marker in METODE_MARKERS.get(kelas, []):
            if marker in seen:
                continue
            seen.add(marker)
            sec = _extract_section(full, marker)
            if sec:
                blocks.append(sec)
    return "\n\n----------\n\n".join(blocks)


def get_ai_insight(total, counts, avg_conf, per_class_conf):
    """
    Hasilkan analisis & rekomendasi.
    Return: (teks, sumber)  ->  sumber "ai" | "auto".
    """
    key = _get_api_key()
    if not key:
        return _fallback_insight(total, counts, avg_conf), "auto"

    rincian = []
    for name in ("Pothole", "Crack", "Manhole"):
        c = counts.get(name, 0)
        if c:
            pc = per_class_conf.get(name)
            pc_txt = f", conf rata-rata {pc*100:.0f}%" if pc is not None else ""
            rincian.append(f"- {name}: {c} objek{pc_txt}")
    rincian_txt = "\n".join(rincian) if rincian else "- Tidak ada kerusakan terdeteksi"

    # Referensi verbatim (lengkap) dari PDF manual, sesuai kelas terdeteksi
    ref = referensi_pdf(counts)
    ref_blok = ""
    if ref:
        ref_blok = (
            f"\n\n=== REFERENSI RESMI (kutipan LENGKAP dari {MANUAL_REF}) ===\n"
            f"Gunakan kutipan berikut sebagai dasar utama rekomendasi. Pertahankan detail "
            f"penting (kode kerusakan, bahan, komposisi, dan langkah kerja) dan sebutkan "
            f"kode metodenya (mis. P3/P4/P5):\n\n{ref}\n=== AKHIR REFERENSI ==="
        )

    prompt = f"""Kamu adalah insinyur perkerasan jalan (pavement engineer) sekaligus
asisten analisis untuk sistem deteksi kerusakan jalan berbasis computer vision (model YOLO)
yang mendukung dinas infrastruktur / Dinas PU.
Model mendeteksi tiga kelas: Pothole (lubang), Crack (retakan), Manhole (tutup gorong-gorong).

Hasil deteksi pada satu gambar jalan:
Total objek terdeteksi: {total}
Rincian:
{rincian_txt}
Rata-rata keyakinan model keseluruhan: {avg_conf*100:.0f}%{ref_blok}

Tulis analisis LENGKAP dan MENDALAM dalam Bahasa Indonesia. Gunakan struktur berlabel
persis seperti di bawah ini, setiap bagian diawali labelnya sendiri di baris baru.
Setiap bagian konkret (sebutkan angka deteksi bila relevan):

Kondisi Umum: ringkas kondisi jalan secara keseluruhan berdasarkan jumlah dan jenis kerusakan.
Tingkat Keparahan: nyatakan rendah/sedang/tinggi beserta ALASAN teknisnya, dan kaitkan dengan tingkat keyakinan model.
Prioritas Penanganan: urutkan kerusakan mana yang harus didahulukan dan mengapa.
Rekomendasi Tindakan: jelaskan langkah perbaikan secara RINCI per jenis kerusakan. Bila tersedia REFERENSI RESMI di atas, WAJIB mengacu padanya, sebutkan kode metode (P3/P4/P5) dan kode kerusakan, serta uraikan bahan dan langkah kerjanya secara lengkap (jangan diringkas berlebihan).
Estimasi Urgensi: perkiraan seberapa cepat harus ditangani (mis. segera, dalam beberapa minggu, atau pemantauan berkala).
Catatan Keselamatan: risiko bagi pengguna jalan dan langkah pengamanan sementara (mis. rambu, marka, penutupan jalur).
Keterbatasan: ingatkan bahwa hasil model perlu diverifikasi petugas dan akurasi masih terbatas.
Rujukan: sebutkan bahwa metode mengacu pada {MANUAL_REF} (bila referensi tersedia).

Aturan penulisan: gaya profesional dan edukatif, boleh panjang dan rinci pada bagian
Rekomendasi Tindakan, tanpa emoji, JANGAN gunakan tanda markdown seperti bintang atau pagar,
pisahkan tiap bagian dengan baris baru."""

    try:
        from google import genai
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = (getattr(resp, "text", "") or "").strip()
        if text:
            return text, "ai"
    except Exception as e:
        st.session_state["_ai_error"] = str(e)

    return _fallback_insight(total, counts, avg_conf), "auto"


# ----------------------------------------------------------------------
# Render analisis jadi kartu terstruktur (label tebal + isi rapi)
# ----------------------------------------------------------------------
SECTION_LABELS = [
    "Kondisi Umum", "Tingkat Keparahan", "Prioritas Penanganan",
    "Rekomendasi Tindakan", "Estimasi Urgensi", "Catatan Keselamatan",
    "Keterbatasan", "Rujukan",
]


def _esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _parse_sections(text):
    """Pisah teks berlabel 'Label: isi' menjadi daftar (label, isi)."""
    pat = re.compile(r"(" + "|".join(re.escape(l) for l in SECTION_LABELS) + r")\s*:\s*",
                     re.IGNORECASE)
    matches = list(pat.finditer(text))
    if not matches:
        return []
    out = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip().strip("\n").strip()
        out.append((m.group(1), body))
    return out


def build_insight_html(text, src):
    is_ai = (src == "ai")
    badge = "Dijelaskan AI (Gemini)" if is_ai else "Ringkasan otomatis"
    bcls = "ai-badge" if is_ai else "ai-badge auto"
    g = "" if is_ai else " g"

    sections = _parse_sections(text)
    head = (f'<div class="ins-head"><span class="{bcls}">{badge}</span>'
            f'<span class="ins-title">Analisis Kondisi Jalan</span></div>')

    if not sections:
        return f'<div class="ai-card">{head}<div class="ai-body">{_esc(text)}</div></div>'

    rows = ""
    for idx, (label, body) in enumerate(sections, 1):
        rows += (
            f'<div class="ins-row" style="animation-delay:{(idx-1)*0.05:.2f}s">'
            f'<div class="ins-num{g}">{idx}</div>'
            f'<div><div class="ins-label{g}">{_esc(label)}</div>'
            f'<div class="ins-text">{_esc(body)}</div></div></div>'
        )
    return f'<div class="ai-card">{head}{rows}</div>'


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">VINIX7 Divisi AI &nbsp;|&nbsp; SDG 9 - Infrastruktur</div>
  <h1>Deteksi Kerusakan Jalan Cerdas</h1>
  <p>Sistem computer vision (YOLO) yang mengenali lubang, retakan, dan tutup gorong-gorong
     dari foto jalan, lalu memberi analisis serta rekomendasi tindakan secara otomatis untuk
     mendukung perawatan infrastruktur yang lebih cepat, aman, dan hemat biaya.</p>
  <div class="hbadges">
    <span class="hbadge">Model YOLO11n</span>
    <span class="hbadge">3 kelas kerusakan</span>
    <span class="hbadge">Analisis AI (Gemini)</span>
    <span class="hbadge">Real-time</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Cek model
if not MODEL_PATH.exists():
    st.error(
        "Model `models/best.pt` belum ditemukan. Latih model lewat notebook "
        "`road_damage_yolo_training.ipynb`, lalu letakkan file `best.pt` di folder `models/`."
    )
    st.stop()

model = load_model(str(MODEL_PATH))

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Pengaturan")
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
    st.caption("Legenda kelas")
    for cid, name in CLASS_NAMES.items():
        st.markdown(
            f'<span class="badge" style="background:{CLASS_COLORS[cid]}">{name}</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    ai_on = _get_api_key() is not None
    st.caption("Analisis AI (Gemini): " + ("aktif" if ai_on else "mode ringkasan otomatis"))
    if not ai_on:
        st.caption("Tambahkan GEMINI_API_KEY di secrets untuk mengaktifkan analisis AI.")

# ----------------------------------------------------------------------
# Input gambar
# ----------------------------------------------------------------------
st.markdown('<div class="kick">Langkah 1</div>', unsafe_allow_html=True)
st.markdown('<p class="sub">Pilih gambar jalan yang ingin dianalisis.</p>', unsafe_allow_html=True)

if "image" not in st.session_state:
    st.session_state.image = None

col_up, col_sample = st.columns([2, 1])

with col_up:
    uploaded = st.file_uploader("Unggah gambar (JPG / PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        st.session_state.image = Image.open(uploaded).convert("RGB")

with col_sample:
    st.write("Atau gunakan contoh:")
    sample_dir = Path("sample_images")
    samples = sorted(sample_dir.glob("*.jpg")) if sample_dir.exists() else []
    for sp in samples:
        if st.button(sp.stem, use_container_width=True):
            st.session_state.image = Image.open(sp).convert("RGB")

# gambar bertahan antar-rerun (mis. saat menekan tombol analisis)
image = st.session_state.image

# ----------------------------------------------------------------------
# Inferensi + hasil
# ----------------------------------------------------------------------
if image is not None:
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="kick">Langkah 2</div>', unsafe_allow_html=True)
    st.markdown('<p class="sub">Hasil deteksi model.</p>', unsafe_allow_html=True)

    classes_to_show = [cid for cid, on in active.items() if on]
    with st.spinner("Menganalisis gambar..."):
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

    # ringkasan deteksi
    rows = []
    for box in result.boxes:
        cid = int(box.cls)
        rows.append({
            "Kelas": CLASS_NAMES.get(cid, str(cid)),
            "Keterangan": CLASS_LABEL_ID.get(cid, "-"),
            "Confidence": round(float(box.conf), 3),
        })
    df = pd.DataFrame(rows)
    total = len(df)
    counts = df["Kelas"].value_counts().to_dict() if total else {}
    avg_conf = float(df["Confidence"].mean()) if total else 0.0
    per_class_conf = (
        df.groupby("Kelas")["Confidence"].mean().to_dict() if total else {}
    )

    st.markdown('<div class="kick">Langkah 3</div>', unsafe_allow_html=True)
    st.markdown('<p class="sub">Ringkasan jumlah deteksi.</p>', unsafe_allow_html=True)

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
        st.write("")
        st.dataframe(df, use_container_width=True, hide_index=True)

        buf = io.BytesIO()
        Image.fromarray(annotated).save(buf, format="JPEG")
        st.download_button(
            "Unduh gambar hasil",
            data=buf.getvalue(),
            file_name="hasil_deteksi.jpg",
            mime="image/jpeg",
        )
    else:
        st.info("Tidak ada kerusakan terdeteksi pada ambang saat ini. "
                "Coba turunkan confidence threshold.")

    # ------------------------------------------------------------------
    # Langkah 4 - Analisis & rekomendasi AI
    # ------------------------------------------------------------------
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="kick">Langkah 4</div>', unsafe_allow_html=True)
    st.markdown('<p class="sub">Analisis kondisi jalan dan rekomendasi tindakan.</p>',
                unsafe_allow_html=True)

    sig = f"{total}|" + "|".join(f"{k}:{counts.get(k,0)}:{per_class_conf.get(k,0):.3f}"
                                 for k in CLASS_NAMES.values())

    if st.button("Buat analisis & rekomendasi AI", type="primary"):
        with st.spinner("Menyusun analisis..."):
            text, src = get_ai_insight(total, counts, avg_conf, per_class_conf)
        st.session_state["ai_insight"] = {"sig": sig, "text": text, "src": src}

    data = st.session_state.get("ai_insight")
    if data and data["sig"] == sig:
        st.markdown(build_insight_html(data["text"], data["src"]), unsafe_allow_html=True)
        st.markdown(
            '<p class="disc">Analisis di atas bersifat pendukung dan tidak menggantikan '
            'pemeriksaan teknis lapangan. Akurasi model masih terbatas (baseline mAP@50 '
            '0,474), sehingga keputusan akhir tetap berada pada petugas.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="note">Klik tombol di atas untuk menghasilkan analisis kondisi '
            'jalan beserta rekomendasi tindakan berdasarkan hasil deteksi.</div>',
            unsafe_allow_html=True,
        )
else:
    st.info("Unggah gambar atau pilih salah satu contoh untuk memulai.")

# ======================================================================
# EDUKASI & MANFAAT
# ======================================================================
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div class="kick">Edukasi</div>', unsafe_allow_html=True)
st.markdown('<h2 style="margin:.1rem 0 .2rem; font-weight:900;">Cara Kerja Sistem</h2>',
            unsafe_allow_html=True)
st.markdown('<p class="sub">Tiga langkah sederhana dari foto menjadi keputusan.</p>',
            unsafe_allow_html=True)

st.markdown("""
<div class="grid3">
  <div class="ecard">
    <div class="enum">1</div>
    <p class="etitle">Unggah Foto Jalan</p>
    <p class="etext">Cukup foto permukaan jalan dari kamera ponsel atau kendaraan.
       Tidak perlu alat khusus - satu gambar sudah bisa dianalisis.</p>
  </div>
  <div class="ecard" style="animation-delay:.08s">
    <div class="enum">2</div>
    <p class="etitle">Model Mendeteksi</p>
    <p class="etext">Model YOLO memindai gambar, menandai setiap kerusakan dengan kotak,
       dan mengklasifikasikannya menjadi lubang, retakan, atau manhole.</p>
  </div>
  <div class="ecard" style="animation-delay:.16s">
    <div class="enum">3</div>
    <p class="etitle">AI Memberi Rekomendasi</p>
    <p class="etext">Hasil deteksi diringkas menjadi tingkat keparahan, prioritas,
       dan saran tindakan yang mudah dipahami petugas maupun warga.</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="height:1.6rem"></div>', unsafe_allow_html=True)
st.markdown('<div class="kick">Manfaat</div>', unsafe_allow_html=True)
st.markdown('<h2 style="margin:.1rem 0 .2rem; font-weight:900;">Mengapa Ini Penting</h2>',
            unsafe_allow_html=True)
st.markdown('<p class="sub">Nilai nyata sistem deteksi kerusakan jalan otomatis.</p>',
            unsafe_allow_html=True)

st.markdown("""
<div class="grid4">
  <div class="ecard">
    <div class="enum">A</div>
    <p class="etitle">Keselamatan</p>
    <p class="etext">Lubang terdeteksi lebih dini sehingga risiko kecelakaan dan
       kerusakan kendaraan dapat ditekan.</p>
  </div>
  <div class="ecard" style="animation-delay:.06s">
    <div class="enum">B</div>
    <p class="etitle">Hemat Biaya</p>
    <p class="etext">Retakan yang ditangani sejak awal jauh lebih murah daripada
       menunggu jalan rusak parah dan perlu rekonstruksi.</p>
  </div>
  <div class="ecard" style="animation-delay:.12s">
    <div class="enum">C</div>
    <p class="etitle">Cepat & Skalabel</p>
    <p class="etext">Inspeksi otomatis memproses banyak gambar dalam hitungan detik,
       jauh lebih cepat dibanding survei manual.</p>
  </div>
  <div class="ecard" style="animation-delay:.18s">
    <div class="enum g">D</div>
    <p class="etitle">Berbasis Data</p>
    <p class="etext">Prioritas perbaikan jadi objektif dan terukur, mendukung
       tata kelola infrastruktur pintar (SDG 9).</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="height:1.6rem"></div>', unsafe_allow_html=True)
st.markdown('<div class="kick">Kenali Kelasnya</div>', unsafe_allow_html=True)
st.markdown('<h2 style="margin:.1rem 0 .2rem; font-weight:900;">Tiga Jenis Kerusakan</h2>',
            unsafe_allow_html=True)

st.markdown(f"""
<div class="grid3">
  <div class="ecard">
    <p class="etitle"><span class="chip-legend" style="background:{CLASS_COLORS[0]}"></span>Pothole (Lubang)</p>
    <p class="etext">Cekungan pada permukaan jalan. Paling berbahaya bagi pengendara
       dan menjadi prioritas penanganan tertinggi - biasanya perlu ditambal segera.</p>
  </div>
  <div class="ecard" style="animation-delay:.08s">
    <p class="etitle"><span class="chip-legend" style="background:{CLASS_COLORS[1]}"></span>Crack (Retakan)</p>
    <p class="etext">Garis retak halus yang menjadi tanda awal kerusakan. Jika dibiarkan
       akan melebar menjadi lubang, sehingga ideal ditangani dengan sealing lebih dulu.</p>
  </div>
  <div class="ecard" style="animation-delay:.16s">
    <p class="etitle"><span class="chip-legend" style="background:{CLASS_COLORS[2]}"></span>Manhole (Gorong-gorong)</p>
    <p class="etext">Tutup saluran di badan jalan. Bukan kerusakan, tetapi posisinya perlu
       rata dengan aspal; bila menonjol atau ambles bisa membahayakan.</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="disc">Catatan: sistem ini adalah alat bantu penyaring awal, bukan pengganti '
    'inspeksi teknik. Akurasi model masih terbatas (baseline mAP@50 0,474), jadi keputusan '
    'perbaikan akhir tetap memerlukan verifikasi petugas di lapangan. Rekomendasi metode '
    'perbaikan merujuk pada Manual Bina Marga No. 001-02/M/BM/2011.</p>',
    unsafe_allow_html=True,
)