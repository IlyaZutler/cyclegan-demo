import io
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms as tr

from model import load_generators

# ── страница ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CycleGAN · Apple ↔ Orange",
    page_icon="🍊",
    layout="wide",
)

# ── стили ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp { background: #0d0d0d; color: #f0ede6; }

h1 { font-size: 2.6rem !important; font-weight: 800 !important;
     letter-spacing: -0.03em; color: #f0ede6 !important; }
h2, h3 { font-size: 1.1rem !important; font-weight: 700 !important;
          text-transform: uppercase; letter-spacing: 0.08em;
          color: #ff6b2b !important; }

.domain-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.badge-a { background: #1a3a1a; color: #6ddd6d; border: 1px solid #3a6a3a; }
.badge-b { background: #3a1a00; color: #ff8c42; border: 1px solid #6a3a00; }
.badge-rec { background: #1a1a3a; color: #8080ff; border: 1px solid #3a3a6a; }

.img-card {
    background: #1a1a1a;
    border-radius: 12px;
    padding: 14px;
    border: 1px solid #2a2a2a;
    text-align: center;
}
.img-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 8px;
}
.arrow {
    font-size: 1.6rem;
    color: #ff6b2b;
    text-align: center;
    padding-top: 70px;
}
.divider { border: none; border-top: 1px solid #2a2a2a; margin: 32px 0; }
.info-box {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-left: 3px solid #ff6b2b;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.82rem;
    color: #888;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# ── константы нормализации (apple2orange, computed from trainA/trainB) ─────────
MEAN_A = [0.5921, 0.4201, 0.3659]
STD_A  = [0.3006, 0.3368, 0.3370]
MEAN_B = [0.6526, 0.4834, 0.2991]
STD_B  = [0.3025, 0.2775, 0.3116]

# ── загрузка модели ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Загружаем генераторы…")
def get_generators():
    device = torch.device("cpu")
    return load_generators("generators.pt", device)

generators = get_generators()
device = torch.device("cpu")


# ── вспомогательные функции ────────────────────────────────────────────────────
def make_transform(mean, std, size=256):
    return tr.Compose([
        tr.Resize(size),
        tr.CenterCrop(size),
        tr.ToTensor(),
        tr.Normalize(mean=mean, std=std),
    ])

def de_normalize(tensor, mean, std):
    """tensor (3,H,W) → numpy (H,W,3) uint8"""
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr  = np.array(std,  dtype=np.float32)
    t = tensor.detach().cpu().float()
    t = t.permute(1, 2, 0).numpy()
    t = t * std_arr + mean_arr
    t = np.clip(t, 0.0, 1.0)
    return (t * 255).astype(np.uint8)

def tensor_to_pil(tensor, mean, std):
    return Image.fromarray(de_normalize(tensor, mean, std))

def run_inference(pil_img, source_domain: str):
    """
    source_domain: "A" (apple) или "B" (orange)
    Возвращает (original_pil, translated_pil, reconstructed_pil)
    """
    if source_domain == "A":
        transform_src  = make_transform(MEAN_A, STD_A)
        transform_dst  = make_transform(MEAN_B, STD_B)   # для формальности
        g_forward  = generators["a_to_b"]   # A → B
        g_backward = generators["b_to_a"]   # B → A (для rec)
        mean_src, std_src = MEAN_A, STD_A
        mean_dst, std_dst = MEAN_B, STD_B
    else:
        transform_src  = make_transform(MEAN_B, STD_B)
        g_forward  = generators["b_to_a"]   # B → A
        g_backward = generators["a_to_b"]   # A → B (для rec)
        mean_src, std_src = MEAN_B, STD_B
        mean_dst, std_dst = MEAN_A, STD_A

    pil_rgb = pil_img.convert("RGB")
    src_tensor = transform_src(pil_rgb).unsqueeze(0).to(device)  # (1,3,256,256)

    with torch.no_grad():
        dst_tensor = g_forward(src_tensor)          # translated
        rec_tensor = g_backward(dst_tensor)         # reconstructed

    original_pil     = tensor_to_pil(src_tensor[0],  mean_src, std_src)
    translated_pil   = tensor_to_pil(dst_tensor[0],  mean_dst, std_dst)
    reconstructed_pil = tensor_to_pil(rec_tensor[0], mean_src, std_src)

    return original_pil, translated_pil, reconstructed_pil


def show_image_card(pil_img, badge_html, label):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    st.markdown(f'<div class="img-card">{badge_html}</div>', unsafe_allow_html=True)
    st.image(pil_img, use_container_width=True)
    st.markdown(f'<div class="img-label">{label}</div>', unsafe_allow_html=True)


# ── заголовок ──────────────────────────────────────────────────────────────────
st.markdown("# 🍎 CycleGAN · Apple ↔ Orange")
st.markdown("""
<div class="info-box">
Unpaired image-to-image translation.<br>
Загрузи фото яблока (домен A) или апельсина (домен B) — модель переведёт его
в противоположный домен и покажет реконструкцию через обратный цикл.
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── сайдбар ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Параметры")

    domain = st.radio(
        "Домен загружаемого изображения",
        options=["A — Apple 🍎", "B — Orange 🍊"],
        index=0,
    )
    source_domain = "A" if domain.startswith("A") else "B"

    uploaded = st.file_uploader(
        "Загрузи изображение",
        type=["jpg", "jpeg", "png", "webp"],
        help="Рекомендуемый размер — от 256×256. Большие изображения будут обрезаны до 256×256."
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Домены</b><br>
    <span style="color:#6ddd6d">■ A</span> — яблоки (apple2orange trainA)<br>
    <span style="color:#ff8c42">■ B</span> — апельсины (apple2orange trainB)<br><br>
    <b>Архитектура</b><br>
    ResNet-9 генераторы<br>
    PatchGAN 70×70 дискриминаторы<br>
    λ = 10 (cycle consistency)<br>
    200 эпох, Adam β₁=0.5
    </div>
    """, unsafe_allow_html=True)


# ── основной контент ───────────────────────────────────────────────────────────
if uploaded is None:
    # Placeholder
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if source_domain == "A":
            st.markdown("### Ожидаем фото яблока (домен A)")
        else:
            st.markdown("### Ожидаем фото апельсина (домен B)")
        st.markdown("""
        <div class="info-box" style="text-align:center; padding: 40px 20px;">
            <div style="font-size:4rem; margin-bottom:16px">🍎🍊</div>
            Загрузи изображение через панель слева,<br>
            чтобы запустить инференс.
        </div>
        """, unsafe_allow_html=True)

else:
    pil_img = Image.open(uploaded)

    with st.spinner("Генерируем перевод…"):
        original_pil, translated_pil, reconstructed_pil = run_inference(pil_img, source_domain)

    # ── лейблы в зависимости от направления ───────────────────────────────────
    if source_domain == "A":
        badge_orig  = '<span class="domain-badge badge-a">Domain A · Apple</span>'
        badge_trans = '<span class="domain-badge badge-b">Domain B · Orange</span>'
        badge_rec   = '<span class="domain-badge badge-rec">Reconstruction A</span>'
        label_orig  = "Оригинал (A)"
        label_trans = "Перевод A → B"
        label_rec   = "Реконструкция B → A → A"
        arrow_text  = "A → B"
    else:
        badge_orig  = '<span class="domain-badge badge-b">Domain B · Orange</span>'
        badge_trans = '<span class="domain-badge badge-a">Domain A · Apple</span>'
        badge_rec   = '<span class="domain-badge badge-rec">Reconstruction B</span>'
        label_orig  = "Оригинал (B)"
        label_trans = "Перевод B → A"
        label_rec   = "Реконструкция A → B → B"
        arrow_text  = "B → A"

    # ── три колонки: оригинал | перевод | реконструкция ────────────────────────
    col_orig, col_arrow1, col_trans, col_arrow2, col_rec = st.columns([10, 1, 10, 1, 10])

    with col_orig:
        st.markdown(f'<div class="img-card">{badge_orig}</div>', unsafe_allow_html=True)
        st.image(original_pil, use_container_width=True)
        st.markdown(f'<div class="img-label">{label_orig}</div>', unsafe_allow_html=True)

    with col_arrow1:
        st.markdown(f'<div class="arrow">→</div>', unsafe_allow_html=True)

    with col_trans:
        st.markdown(f'<div class="img-card">{badge_trans}</div>', unsafe_allow_html=True)
        st.image(translated_pil, use_container_width=True)
        st.markdown(f'<div class="img-label">{label_trans}</div>', unsafe_allow_html=True)

    with col_arrow2:
        st.markdown(f'<div class="arrow">→</div>', unsafe_allow_html=True)

    with col_rec:
        st.markdown(f'<div class="img-card">{badge_rec}</div>', unsafe_allow_html=True)
        st.image(reconstructed_pil, use_container_width=True)
        st.markdown(f'<div class="img-label">{label_rec}</div>', unsafe_allow_html=True)

    # ── скачать результаты ─────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### Скачать результаты")

    dl1, dl2, dl3 = st.columns(3)

    def pil_to_bytes(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    with dl1:
        st.download_button("⬇ Оригинал",       pil_to_bytes(original_pil),      "original.png",      "image/png", use_container_width=True)
    with dl2:
        st.download_button("⬇ Перевод",         pil_to_bytes(translated_pil),    "translated.png",    "image/png", use_container_width=True)
    with dl3:
        st.download_button("⬇ Реконструкция",   pil_to_bytes(reconstructed_pil), "reconstructed.png", "image/png", use_container_width=True)
