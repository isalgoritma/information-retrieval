import html
import json
import random
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from modules.preprocessing import preprocess_query
from modules.ir_engine import rank_documents


# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="Recipe Finder - Information Retrieval",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# 2. CUSTOM CSS + ANIMASI (aman untuk mode terang & gelap)
# ==========================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown, button, input, label {
        font-family: 'Poppins', sans-serif;
    }

    /* ---------- Angka berhitung naik (count-up) ---------- */
    @property --n {
        syntax: "<integer>";
        inherits: false;
        initial-value: 0;
    }
    .count {
        --n: var(--target);
        counter-reset: num var(--n);
        animation: countUp 1.6s cubic-bezier(.2, .8, .2, 1);
    }
    .count::after { content: counter(num); }
    @keyframes countUp { from { --n: 0; } }

    /* ---------- Keyframes umum ---------- */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(26px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    @keyframes grow { from { width: 0; } }
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes floatY {
        0%, 100% { transform: translateY(0) rotate(-6deg); }
        50%      { transform: translateY(-18px) rotate(6deg); }
    }
    @keyframes reveal {
        from { clip-path: inset(0 100% 0 0); }
        to   { clip-path: inset(0 0 0 0); }
    }
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 0 0 rgba(250, 204, 21, .65); }
        50%      { box-shadow: 0 0 0 11px rgba(250, 204, 21, 0); }
    }
    @keyframes popIn {
        0%   { opacity: 0; transform: scale(.6); }
        70%  { transform: scale(1.08); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* ---------- Hero ---------- */
    .hero {
        background: linear-gradient(120deg, #f97316, #ef4444, #be123c, #f97316);
        background-size: 300% 300%;
        animation: gradientShift 10s ease infinite, fadeUp .8s ease both;
        border-radius: 26px;
        padding: 52px 32px;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 20px 44px rgba(239, 68, 68, 0.30);
        position: relative;
        overflow: hidden;
        margin-bottom: 26px;
    }
    .hero::before, .hero::after {
        content: "";
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.12);
    }
    .hero::before { width: 240px; height: 240px; top: -90px; left: -70px; }
    .hero::after  { width: 300px; height: 300px; bottom: -130px; right: -80px; }
    .hero h1 {
        font-size: 50px;
        font-weight: 800;
        margin: 0;
        color: #ffffff;
        position: relative;
        z-index: 2;
        letter-spacing: -.5px;
    }
    .hero p {
        font-size: 16px;
        margin: 12px 0 0 0;
        opacity: .96;
        position: relative;
        z-index: 2;
        animation: reveal 2.4s steps(42) .5s both;
    }
    .float {
        position: absolute;
        font-size: 34px;
        z-index: 1;
        opacity: .85;
        animation: floatY 5s ease-in-out infinite;
        user-select: none;
    }

    /* ---------- Ikon SVG kustom ---------- */
    .ico {
        display: inline-flex;
        vertical-align: middle;
        width: var(--s, 24px);
        height: var(--s, 24px);
        line-height: 1;
    }
    .ico svg { width: 100%; height: 100%; }
    .float .ico { filter: drop-shadow(0 6px 10px rgba(0, 0, 0, .22)); }

    /* ---------- Stat cards ---------- */
    .stat-card {
        background: rgba(148, 163, 184, 0.10);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-left: 5px solid #f97316;
        border-radius: 16px;
        padding: 16px 20px;
        animation: fadeUp .7s ease both;
        transition: transform .25s ease, box-shadow .25s ease;
    }
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 26px rgba(0, 0, 0, 0.18);
    }
    .stat-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: .08em;
        opacity: .7;
    }
    .stat-value {
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(90deg, #f97316, #ef4444);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ---------- Judul seksi ---------- */
    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin: 28px 0 10px 0;
        animation: fadeIn .6s ease both;
    }

    /* ---------- Chips ---------- */
    .chip {
        display: inline-block;
        padding: 4px 12px;
        margin: 2px 4px 2px 0;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 500;
        background: rgba(249, 115, 22, 0.15);
        color: #f97316;
        border: 1px solid rgba(249, 115, 22, 0.35);
        animation: popIn .45s ease both;
        transition: transform .2s ease;
    }
    .chip:hover { transform: translateY(-2px) scale(1.05); }
    .chip-green {
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border-color: rgba(34, 197, 94, 0.40);
    }
    .chip-muted {
        background: rgba(148, 163, 184, 0.12);
        color: #94a3b8;
        border-color: rgba(148, 163, 184, 0.30);
        text-decoration: line-through;
    }

    /* ---------- Highlight kata yang cocok ---------- */
    mark {
        background: linear-gradient(120deg, rgba(250, 204, 21, .60), rgba(251, 146, 60, .55));
        color: inherit;
        padding: 0 4px;
        border-radius: 5px;
    }

    /* ---------- Info ringkas ---------- */
    .info-box {
        background: rgba(148, 163, 184, 0.10);
        border: 1px dashed rgba(148, 163, 184, 0.45);
        border-radius: 14px;
        padding: 14px 18px;
        margin: 10px 0 18px 0;
        font-size: 14px;
        line-height: 1.9;
        animation: fadeUp .6s ease both;
    }

    /* ---------- Kartu hasil ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px;
        transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
        animation: fadeUp .6s ease both;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(249, 115, 22, .55);
        box-shadow: 0 14px 30px rgba(239, 68, 68, 0.14);
    }
    .result-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        animation: fadeUp .7s ease both;
    }
    .result-left {
        display: flex;
        align-items: center;
        gap: 14px;
        min-width: 0;
    }
    .rank-badge {
        flex: 0 0 auto;
        width: 46px;
        height: 46px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 19px;
        color: #fff;
        background: linear-gradient(135deg, #f97316, #ef4444);
        box-shadow: 0 6px 14px rgba(239, 68, 68, 0.35);
    }
    .rank-1 { background: linear-gradient(135deg, #facc15, #f59e0b); animation: glow 2.2s ease-in-out infinite; }
    .rank-2 { background: linear-gradient(135deg, #cbd5e1, #94a3b8); }
    .rank-3 { background: linear-gradient(135deg, #fb923c, #c2410c); }
    .result-title {
        font-size: 19px;
        font-weight: 700;
        line-height: 1.3;
        margin-bottom: 4px;
    }
    .score-box { text-align: right; min-width: 140px; }
    .score-value {
        font-size: 27px;
        font-weight: 800;
        background: linear-gradient(90deg, #f97316, #ef4444);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .score-label {
        font-size: 11px;
        opacity: .65;
        text-transform: uppercase;
        letter-spacing: .08em;
    }
    .bar-bg {
        background: rgba(148, 163, 184, 0.25);
        border-radius: 999px;
        height: 9px;
        overflow: hidden;
        margin: 12px 0 8px 0;
    }
    .bar-fill {
        height: 100%;
        width: var(--w);
        border-radius: 999px;
        background: linear-gradient(90deg, #fbbf24, #f97316, #ef4444);
        animation: grow 1.2s cubic-bezier(.2, .8, .2, 1) both;
    }

    /* ---------- Detail resep ---------- */
    .ing-list {
        list-style: none;
        padding: 0;
        margin: 6px 0;
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
        gap: 8px 16px;
    }
    .ing-list li {
        padding: 8px 12px;
        border-radius: 10px;
        background: rgba(148, 163, 184, 0.10);
        animation: fadeUp .5s ease both;
        transition: transform .2s ease, background .2s ease;
    }
    .ing-list li:hover {
        transform: translateX(4px);
        background: rgba(249, 115, 22, 0.14);
    }
    .ing-list li .ico { margin-right: 8px; }
    .step {
        display: flex;
        gap: 12px;
        align-items: flex-start;
        margin: 10px 0;
        animation: fadeUp .5s ease both;
    }
    .step-no {
        flex: 0 0 30px;
        height: 30px;
        border-radius: 50%;
        background: linear-gradient(135deg, #f97316, #ef4444);
        color: #fff;
        text-align: center;
        line-height: 30px;
        font-weight: 700;
        font-size: 13px;
    }

    /* ---------- Tombol ---------- */
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(90deg, #f97316, #ef4444);
        color: #fff;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 10px 0;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(239, 68, 68, 0.4);
        color: #fff;
    }
    div[data-testid="stButton"] button {
        border-radius: 999px;
        font-size: 13px;
        transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
    }
    div[data-testid="stButton"] button:hover {
        transform: translateY(-2px);
        border-color: #f97316;
        box-shadow: 0 6px 14px rgba(249, 115, 22, 0.25);
    }

    .footer {
        text-align: center;
        opacity: .6;
        font-size: 13px;
        margin: 40px 0 10px 0;
    }

    /* Hormati pengguna yang mematikan animasi */
    @media (prefers-reduced-motion: reduce) {
        * { animation: none !important; transition: none !important; }
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. DATASET & FUNGSI BANTU
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "preprocessed_dataset.json"

CONTOH_QUERY = [
    "nasi goreng porang",
    "ayam bakar",
    "sayur bening",
    "tempe goreng",
    "ikan pepes",
]

SKIP_HIGHLIGHT = {"amp", "quot", "lt", "gt", "39"}


@st.cache_data
def load_dataset():
    with open(DATASET_FILE, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError("Format dataset harus berupa list JSON.")

    return dataset


def esc(value):
    """Escape teks agar aman disisipkan ke HTML."""
    return html.escape(str(value))


# ----- Ikon SVG kustom -----
# Taruh file SVG di folder: assets/icons/<nama>.svg
# Kalau file tidak ada, otomatis dipakai emoji sebagai cadangan.

ICON_DIR = BASE_DIR / "assets" / "icons"


def icon(name, fallback="", size=24):
    """Mengembalikan SVG inline dari assets/icons/<name>.svg (atau emoji cadangan)."""
    path = ICON_DIR / f"{name}.svg"

    if not path.exists():
        return fallback

    try:
        svg = path.read_text(encoding="utf-8")
    except OSError:
        return fallback

    # Bersihkan: deklarasi XML, doctype, komentar, script, dan spasi antar tag
    svg = re.sub(r"<\?xml.*?\?>", "", svg, flags=re.S)
    svg = re.sub(r"<!DOCTYPE.*?>", "", svg, flags=re.S | re.I)
    svg = re.sub(r"<!--.*?-->", "", svg, flags=re.S)
    svg = re.sub(r"<script.*?</script>", "", svg, flags=re.S | re.I)
    svg = re.sub(r">\s+<", "><", svg).strip()
    svg = " ".join(svg.split())

    if "<svg" not in svg:
        return fallback

    return f'<span class="ico" style="--s: {size}px">{svg}</span>'


def highlight(text, tokens):
    """Escape teks lalu beri <mark> pada kata yang cocok dengan token query."""
    safe = esc(text)
    toks = sorted(
        {
            esc(t) for t in tokens
            if len(t) >= 2 and t.lower() not in SKIP_HIGHLIGHT
        },
        key=len,
        reverse=True,
    )
    if not toks:
        return safe

    pattern = re.compile(
        "(" + "|".join(re.escape(t) for t in toks) + ")",
        re.IGNORECASE,
    )
    return pattern.sub(r"<mark>\1</mark>", safe)


def matched_tokens(doc, tokens):
    """Token query yang terdeteksi di judul/bahan/langkah resep."""
    bahan = doc.get("bahan", [])
    langkah = doc.get("langkah", [])
    bahan = bahan if isinstance(bahan, list) else [bahan]
    langkah = langkah if isinstance(langkah, list) else [langkah]

    teks = " ".join(
        [str(doc.get("judul", ""))]
        + [str(b) for b in bahan]
        + [str(s) for s in langkah]
    ).lower()

    return [t for t in tokens if t.lower() in teks]


# ----- Callback (dijalankan sebelum script dirender ulang) -----

def set_contoh(teks):
    st.session_state["query_input"] = teks
    st.session_state["auto_search"] = True


def toggle_fav(judul):
    favs = st.session_state["favorites"]
    if judul in favs:
        favs.remove(judul)
        st.toast("Dihapus dari favorit", icon="💔")
    else:
        favs.append(judul)
        st.toast("Disimpan ke favorit!", icon="⭐")


def shuffle_random(total):
    st.session_state["random_idx"] = random.randrange(total)


def clear_history():
    st.session_state["history"] = []


# ----- Kartu resep -----

def render_card(doc, rank, tokens, key):
    """Menampilkan satu kartu resep. rank=None untuk kartu tanpa skor."""
    judul = doc.get("judul", "Resep tanpa judul")
    komoditas = doc.get("komoditas", "-")
    bahan = doc.get("bahan", [])
    langkah = doc.get("langkah", [])

    delay = min(0.07 * ((rank or 1) - 1), 0.8)
    is_fav = judul in st.session_state["favorites"]

    with st.container(border=True):

        if rank is None:
            badge = f'<div class="rank-badge">{icon("dice", "🎲", 26)}</div>'
            score_html = ""
            bar_html = ""
        else:
            score = float(doc.get("score", 0))
            persen = max(0.0, min(score * 100, 100.0))
            rank_class = f"rank-{rank}" if rank <= 3 else ""
            badge = f'<div class="rank-badge {rank_class}">{rank}</div>'
            score_html = f"""
                <div class="score-box">
                    <div class="score-value">{persen:.2f}%</div>
                    <div class="score-label">Similarity</div>
                </div>"""
            bar_html = f"""
                <div class="bar-bg">
                    <div class="bar-fill" style="--w: {persen:.2f}%; animation-delay: {delay + 0.2:.2f}s"></div>
                </div>"""

        st.markdown(f"""
        <div class="result-head" style="animation-delay: {delay:.2f}s">
            <div class="result-left">
                {badge}
                <div>
                    <div class="result-title">{highlight(judul, tokens)}</div>
                    <span class="chip chip-green">{icon('leaf', '🌿', 14)} {esc(komoditas)}</span>
                </div>
            </div>
            {score_html}
        </div>
        {bar_html}
        """, unsafe_allow_html=True)

        col_chip, col_fav = st.columns([5, 2])

        with col_chip:
            if tokens:
                cocok = set(matched_tokens(doc, tokens))
                chips = "".join(
                    f'<span class="chip chip-green">✓ {esc(t)}</span>'
                    if t in cocok else
                    f'<span class="chip chip-muted">{esc(t)}</span>'
                    for t in tokens
                )
                st.markdown(
                    f'<div>{chips}</div>',
                    unsafe_allow_html=True,
                )

        with col_fav:
            st.button(
                "⭐ Tersimpan" if is_fav else "☆ Simpan",
                key=f"fav_{key}",
                on_click=toggle_fav,
                args=(judul,),
                use_container_width=True,
            )

        with st.expander("📖 Lihat detail resep"):

            tab_bahan, tab_langkah = st.tabs(
                ["🥘 Bahan-bahan", "👩‍🍳 Langkah memasak"]
            )

            with tab_bahan:
                if isinstance(bahan, list) and bahan:
                    items = "".join(
                        f'<li style="animation-delay: {min(k * 0.04, 0.8):.2f}s">'
                        f'{icon("spoon", "🥄", 16)}{highlight(item, tokens)}</li>'
                        for k, item in enumerate(bahan)
                    )
                    st.markdown(
                        f'<ul class="ing-list">{items}</ul>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.write("Bahan tidak tersedia.")

            with tab_langkah:
                if isinstance(langkah, list) and langkah:
                    steps = "".join(
                        f'<div class="step" style="animation-delay: {min(j * 0.06, 0.9):.2f}s">'
                        f'<div class="step-no">{j + 1}</div>'
                        f'<div>{highlight(step, tokens)}</div></div>'
                        for j, step in enumerate(langkah)
                    )
                    st.markdown(steps, unsafe_allow_html=True)
                else:
                    st.write("Langkah memasak tidak tersedia.")

            url = doc.get("url_sumber", "")

            if isinstance(url, str) and url.startswith("https://"):
                st.link_button("🔗 Lihat Resep Asli", url)


# ==========================================
# 4. STATE AWAL
# ==========================================

st.session_state.setdefault("favorites", [])
st.session_state.setdefault("history", [])
st.session_state.setdefault("last_search", None)


# ==========================================
# 5. HEADER APLIKASI
# ==========================================

st.markdown(f"""
<div class="hero">
    <span class="float" style="left: 6%;  top: 22%; animation-delay: 0s;">{icon('noodle', '🍜', 42)}</span>
    <span class="float" style="left: 16%; top: 62%; animation-delay: .8s;">{icon('chili', '🌶️', 42)}</span>
    <span class="float" style="left: 30%; top: 12%; animation-delay: 1.6s;">{icon('tomato', '🍅', 42)}</span>
    <span class="float" style="right: 30%; top: 66%; animation-delay: .4s;">{icon('garlic', '🧄', 42)}</span>
    <span class="float" style="right: 16%; top: 16%; animation-delay: 1.2s;">{icon('carrot', '🥕', 42)}</span>
    <span class="float" style="right: 6%; top: 58%; animation-delay: 2s;">{icon('pot', '🥘', 42)}</span>
    <h1>{icon('logo', '🍲', 52)} Recipe Finder</h1>
    <p>Sistem Pencarian Resep Masakan Berbasis Information Retrieval</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 6. MEMUAT DATASET
# ==========================================

try:
    dataset = load_dataset()

except FileNotFoundError:
    st.error(
        "File preprocessed_dataset.json tidak ditemukan. "
        "Jalankan preprocessing.py terlebih dahulu."
    )
    st.stop()

except (json.JSONDecodeError, ValueError) as error:
    st.error(f"Dataset tidak dapat dibaca: {error}")
    st.stop()

if "random_idx" not in st.session_state:
    st.session_state["random_idx"] = random.randrange(max(len(dataset), 1))

total_komoditas = len({
    d.get("komoditas") for d in dataset
    if isinstance(d, dict) and d.get("komoditas")
})

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="stat-card" style="animation-delay: .1s">
        <div class="stat-label">{icon('book', '📚', 16)} Total Resep</div>
        <div class="stat-value"><span class="count" style="--target: {len(dataset)}"></span></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card" style="animation-delay: .25s">
        <div class="stat-label">{icon('vegetable', '🥬', 16)} Jenis Komoditas</div>
        <div class="stat-value"><span class="count" style="--target: {total_komoditas}"></span></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-card" style="animation-delay: .4s">
        <div class="stat-label">{icon('gear', '⚙️', 16)} Metode Pencarian</div>
        <div class="stat-value"><span class="count" style="--target: 2"></span> Metode</div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 7. FORM PENCARIAN
# ==========================================

st.markdown(
    f'<div class="section-title">{icon("search", "🔎", 28)} Cari Resep Masakan</div>',
    unsafe_allow_html=True,
)

st.caption("Coba cepat:")
chip_cols = st.columns(len(CONTOH_QUERY))
for col, contoh in zip(chip_cols, CONTOH_QUERY):
    with col:
        st.button(
            contoh,
            key=f"contoh_{contoh}",
            on_click=set_contoh,
            args=(contoh,),
            use_container_width=True,
        )

with st.form("search_form"):

    query = st.text_input(
        "Masukkan nama resep atau bahan masakan",
        placeholder="Contoh: nasi goreng porang",
        key="query_input",
    )

    col_a, col_b = st.columns(2)

    with col_a:
        method_option = st.selectbox(
            "Pilih Metode Similarity",
            [
                "Jaccard Similarity",
                "Sørensen-Dice Coefficient",
            ],
        )

    with col_b:
        jumlah_hasil = st.selectbox(
            "Jumlah hasil pencarian",
            [5, 10, 20, 50, "Semua"],
            index=0,
        )

    search_button = st.form_submit_button(
        "🔍 Cari Resep",
        type="primary",
        use_container_width=True,
    )

auto_search = st.session_state.pop("auto_search", False)


# ==========================================
# 8. SIMPAN PARAMETER PENCARIAN
#    (disimpan di session_state supaya hasil tidak hilang
#     ketika tombol Simpan/filter ditekan)
# ==========================================

if search_button or auto_search:

    if not query.strip():
        st.warning("Masukkan kata kunci pencarian terlebih dahulu.")

    else:
        q = query.strip()

        st.session_state["last_search"] = {
            "query": q,
            "method_option": method_option,
            "jumlah": jumlah_hasil,
        }

        riwayat = [h for h in st.session_state["history"] if h.lower() != q.lower()]
        st.session_state["history"] = ([q] + riwayat)[:6]
        st.session_state["new_search"] = True


# ==========================================
# 9. TAMPILKAN HASIL
# ==========================================

ls = st.session_state["last_search"]

if ls:

    query_tokens = preprocess_query(ls["query"])

    if not query_tokens:
        st.warning(
            "Kata kunci tidak memiliki token setelah preprocessing. "
            "Silakan gunakan kata kunci lain."
        )

    else:
        method = "jaccard" if ls["method_option"] == "Jaccard Similarity" else "dsc"

        # Kedua metode dihitung agar bisa dibandingkan di tab Analisis
        res_jaccard = rank_documents(query_tokens, dataset, method="jaccard")
        res_dsc = rank_documents(query_tokens, dataset, method="dsc")
        results = res_jaccard if method == "jaccard" else res_dsc

        if st.session_state.pop("new_search", False):
            st.toast(f"{len(results)} resep ditemukan!", icon="🍽️")

        st.divider()

        st.markdown(
            f'<div class="section-title">{icon("list", "📋", 28)} Hasil Pencarian</div>',
            unsafe_allow_html=True,
        )

        tab_hasil, tab_analisis = st.tabs(
            ["🍽️ Daftar Resep", "📊 Analisis & Perbandingan"]
        )

        # ---------------- TAB HASIL ----------------
        with tab_hasil:

            token_chips = "".join(
                f'<span class="chip">{esc(t)}</span>' for t in query_tokens
            )

            st.markdown(f"""
            <div class="info-box">
                <b>Kata kunci:</b> {esc(ls["query"])}<br>
                <b>Token query:</b> {token_chips}<br>
                <b>Metode:</b> {esc(ls["method_option"])}
            </div>
            """, unsafe_allow_html=True)

            if not results:
                st.warning(
                    "Tidak ditemukan resep yang cocok dengan kata kunci."
                )

            else:
                kunci = f"{ls['query']}|{ls['method_option']}"
                opsi_komoditas = sorted({str(d.get("komoditas", "-")) for d in results})

                f1, f2 = st.columns([3, 2])

                with f1:
                    pilih_kom = st.multiselect(
                        "Filter komoditas",
                        opsi_komoditas,
                        key=f"flt_kom_{kunci}",
                        placeholder="Semua komoditas",
                    )

                with f2:
                    min_sim = st.slider(
                        "Similarity minimum (%)",
                        0, 100, 0,
                        key=f"flt_min_{kunci}",
                    )

                filtered = [
                    d for d in results
                    if (not pilih_kom or str(d.get("komoditas", "-")) in pilih_kom)
                    and float(d.get("score", 0)) * 100 >= min_sim
                ]

                if ls["jumlah"] == "Semua":
                    displayed_results = filtered
                else:
                    displayed_results = filtered[:ls["jumlah"]]

                info_col, dl_col = st.columns([3, 2])

                with info_col:
                    st.success(
                        f"Ditemukan {len(results)} resep yang memiliki "
                        f"kemiripan. Menampilkan "
                        f"{len(displayed_results)} hasil."
                    )

                with dl_col:
                    df_out = pd.DataFrame([
                        {
                            "Peringkat": i,
                            "Judul": d.get("judul", ""),
                            "Komoditas": d.get("komoditas", "-"),
                            "Similarity (%)": round(float(d.get("score", 0)) * 100, 2),
                        }
                        for i, d in enumerate(filtered, start=1)
                    ])
                    st.download_button(
                        "⬇️ Unduh hasil (CSV)",
                        df_out.to_csv(index=False).encode("utf-8"),
                        file_name="hasil_pencarian.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                if not displayed_results:
                    st.info("Tidak ada resep yang lolos filter. Longgarkan filter di atas.")

                for i, doc in enumerate(displayed_results, start=1):
                    render_card(doc, i, query_tokens, key=f"res_{i}")

        # ---------------- TAB ANALISIS ----------------
        with tab_analisis:

            if not results:
                st.info("Belum ada data untuk dianalisis.")

            else:
                skor_j = {str(d.get("judul")): float(d.get("score", 0)) for d in res_jaccard}
                skor_d = {str(d.get("judul")): float(d.get("score", 0)) for d in res_dsc}
                rank_j = {str(d.get("judul")): k for k, d in enumerate(res_jaccard, 1)}
                rank_d = {str(d.get("judul")): k for k, d in enumerate(res_dsc, 1)}

                top1_j = str(res_jaccard[0].get("judul")) if res_jaccard else None
                top1_d = str(res_dsc[0].get("judul")) if res_dsc else None

                m1, m2, m3 = st.columns(3)
                m1.metric("Resep cocok", len(results))
                m2.metric(
                    "Skor tertinggi",
                    f"{float(results[0].get('score', 0)) * 100:.1f}%",
                )
                m3.metric(
                    "Peringkat 1 kedua metode",
                    "Sama ✅" if top1_j == top1_d else "Berbeda ⚠️",
                )

                st.markdown("#### 🥇 Perbandingan Jaccard vs Sørensen-Dice")

                top_judul = []
                for d in list(res_jaccard[:10]) + list(res_dsc[:10]):
                    j = str(d.get("judul"))
                    if j not in top_judul:
                        top_judul.append(j)

                peringkat = rank_j if method == "jaccard" else rank_d
                top_judul.sort(key=lambda j: peringkat.get(j, 10**6))

                df_cmp = pd.DataFrame({
                    "Judul": top_judul,
                    "Rank Jaccard": [rank_j.get(j) for j in top_judul],
                    "Rank Dice": [rank_d.get(j) for j in top_judul],
                    "Jaccard": [skor_j.get(j, 0) * 100 for j in top_judul],
                    "Dice": [skor_d.get(j, 0) * 100 for j in top_judul],
                })

                st.dataframe(
                    df_cmp,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Rank Jaccard": st.column_config.NumberColumn(format="%d"),
                        "Rank Dice": st.column_config.NumberColumn(format="%d"),
                        "Jaccard": st.column_config.ProgressColumn(
                            "Jaccard", min_value=0, max_value=100, format="%.1f%%"
                        ),
                        "Dice": st.column_config.ProgressColumn(
                            "Dice", min_value=0, max_value=100, format="%.1f%%"
                        ),
                    },
                )

                st.markdown("#### 📈 Skor 8 Resep Teratas")

                top8 = top_judul[:8]
                df_chart = pd.DataFrame(
                    {
                        "Jaccard": [skor_j.get(j, 0) * 100 for j in top8],
                        "Dice": [skor_d.get(j, 0) * 100 for j in top8],
                    },
                    index=[f"{k}. {j[:24]}" for k, j in enumerate(top8, 1)],
                )
                st.bar_chart(df_chart)

                st.markdown("#### 🥬 Sebaran Komoditas pada Hasil")

                sebaran = (
                    pd.Series([str(d.get("komoditas", "-")) for d in results])
                    .value_counts()
                    .head(8)
                )
                st.bar_chart(sebaran)


# ==========================================
# 10. KEJUTKAN SAYA (RESEP ACAK)
# ==========================================

st.markdown(
    f'<div class="section-title">{icon("dice", "🎲", 28)} Kejutkan Saya</div>',
    unsafe_allow_html=True,
)

st.caption("Bingung mau masak apa? Biar kami yang pilihkan.")

st.button(
    "🔀 Acak resep lain",
    on_click=shuffle_random,
    args=(len(dataset),),
)

idx = st.session_state["random_idx"] % len(dataset)
render_card(dataset[idx], None, [], key="rand")


# ==========================================
# 11. INFORMASI METODE
# ==========================================

st.write("")

with st.expander("ℹ️ Tentang Metode Pencarian"):

    info1, info2 = st.columns(2)

    with info1:
        st.markdown("#### 1. Jaccard Similarity")
        st.write(
            "Mengukur tingkat kemiripan berdasarkan perbandingan "
            "irisan dan gabungan token query dengan dokumen."
        )
        st.latex(r"J(Q,D) = \frac{|Q \cap D|}{|Q \cup D|}")

    with info2:
        st.markdown("#### 2. Sørensen-Dice Coefficient")
        st.write(
            "Mengukur kemiripan berdasarkan dua kali jumlah "
            "irisan token dibandingkan total token kedua himpunan."
        )
        st.latex(r"DSC(Q,D) = \frac{2 \times |Q \cap D|}{|Q| + |D|}")


# ==========================================
# 12. SIDEBAR: FAVORIT & RIWAYAT
#     (diletakkan di akhir supaya selalu menampilkan data terbaru)
# ==========================================

with st.sidebar:

    st.markdown("## 🍳 Dapur Saya")

    st.markdown("### ⭐ Resep Favorit")

    favorit = st.session_state["favorites"]

    if favorit:
        for k, judul_fav in enumerate(favorit):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{k + 1}.** {judul_fav}")
            c2.button(
                "✕",
                key=f"rm_{k}",
                on_click=toggle_fav,
                args=(judul_fav,),
            )

        st.download_button(
            "⬇️ Unduh daftar favorit",
            "\n".join(f"{k + 1}. {j}" for k, j in enumerate(favorit)),
            file_name="resep_favorit.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.caption("Belum ada. Tekan ☆ Simpan pada resep yang kamu suka.")

    st.divider()

    st.markdown("### 🕘 Riwayat Pencarian")

    if st.session_state["history"]:
        for k, h in enumerate(st.session_state["history"]):
            st.button(
                f"🔁 {h}",
                key=f"hist_{k}",
                on_click=set_contoh,
                args=(h,),
                use_container_width=True,
            )
        st.button(
            "Hapus riwayat",
            key="clear_hist",
            on_click=clear_history,
            use_container_width=True,
        )
    else:
        st.caption("Pencarianmu akan muncul di sini.")


# ==========================================
# 13. FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    Information Retrieval Project &nbsp;|&nbsp;
    Recipe Finder &nbsp;|&nbsp;
    Python &amp; Streamlit
</div>
""", unsafe_allow_html=True)