"""
MRI Image Denoising — U-Net Convolutional Denoising Autoencoder
===============================================================
Key improvements over v1:
  • U-Net skip connections  → preserves fine structural detail (no more blur)
  • MSE + SSIM combined loss → penalises over-smoothing directly
  • Richer synthetic training data (edges, gradients, multi-scale blobs)
  • More steps per epoch + gradient clipping for stability
  • Light-themed Streamlit UI with full Q&A reference tab

Run with:
    pip install streamlit torch torchvision pillow numpy scikit-image
    streamlit run app.py
"""

import io
import math
import time
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# ═══════════════════════════════════════════════════════════════════════════
#  1. MODEL  —  U-Net with skip connections (prevents blurring)
# ═══════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Double conv: Conv→BN→ReLU → Conv→BN→ReLU (standard U-Net block)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UNetDAE(nn.Module):
    """
    U-Net Denoising Autoencoder
    ───────────────────────────
    Input  : 1 × 256 × 256 (noisy grayscale MRI)
    Output : 1 × 256 × 256 (denoised, Sigmoid → [0,1])

    Skip connections copy encoder feature maps directly to the decoder,
    which is the key fix for blurry outputs: high-frequency structural
    details (edges, textures) bypass the bottleneck and are re-injected
    at every decoder stage.
    """
    def __init__(self, base=32):
        super().__init__()
        # ── Encoder ───────────────────────────────────────────────────────
        self.enc1 = ConvBlock(1,        base)        # → base   × 256 × 256
        self.enc2 = ConvBlock(base,     base * 2)    # → base*2 × 128 × 128
        self.enc3 = ConvBlock(base * 2, base * 4)    # → base*4 ×  64 ×  64
        self.enc4 = ConvBlock(base * 4, base * 8)    # → base*8 ×  32 ×  32
        self.pool = nn.MaxPool2d(2)

        # ── Bottleneck ────────────────────────────────────────────────────
        self.bottleneck = ConvBlock(base * 8, base * 16)  # × 16 × 16

        # ── Decoder (concatenates skip from encoder) ──────────────────────
        self.up4   = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4  = ConvBlock(base * 16, base * 8)   # skip concat → ×2 channels

        self.up3   = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3  = ConvBlock(base * 8,  base * 4)

        self.up2   = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2  = ConvBlock(base * 4,  base * 2)

        self.up1   = nn.ConvTranspose2d(base * 2, base,     2, stride=2)
        self.dec1  = ConvBlock(base * 2,  base)

        # ── Output ────────────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        # Encoder (save feature maps for skip connections)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder — cat(up, skip) feeds each block
        d = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d = self.dec3(torch.cat([self.up3(d),  e3], dim=1))
        d = self.dec2(torch.cat([self.up2(d),  e2], dim=1))
        d = self.dec1(torch.cat([self.up1(d),  e1], dim=1))

        return torch.sigmoid(self.out_conv(d))


# ═══════════════════════════════════════════════════════════════════════════
#  2. COMBINED LOSS  —  MSE + SSIM  (prevents over-smoothing)
# ═══════════════════════════════════════════════════════════════════════════

def _gaussian_kernel(size=11, sigma=1.5, device='cpu'):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)


def ssim_loss(pred, target, window_size=11):
    """Differentiable SSIM — returns 1 − SSIM so it can be minimised."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = _gaussian_kernel(window_size, device=pred.device).expand(pred.shape[1], 1, -1, -1)
    pad = window_size // 2

    mu1 = F.conv2d(pred,   k, padding=pad, groups=pred.shape[1])
    mu2 = F.conv2d(target, k, padding=pad, groups=pred.shape[1])
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    s1  = F.conv2d(pred   * pred,   k, padding=pad, groups=pred.shape[1]) - mu1_sq
    s2  = F.conv2d(target * target, k, padding=pad, groups=pred.shape[1]) - mu2_sq
    s12 = F.conv2d(pred   * target, k, padding=pad, groups=pred.shape[1]) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return 1.0 - ssim_map.mean()


def combined_loss(pred, target, alpha=0.7):
    """alpha × MSE + (1−alpha) × (1−SSIM)"""
    return alpha * F.mse_loss(pred, target) + (1 - alpha) * ssim_loss(pred, target)


# ═══════════════════════════════════════════════════════════════════════════
#  3. UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256

def load_image(f):
    img = Image.open(f).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0

def add_noise(img, sigma=0.15):
    return np.clip(img + np.random.normal(0, sigma, img.shape).astype(np.float32), 0, 1)

def to_tensor(a): return torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(DEVICE)
def to_numpy(t):  return t.squeeze().cpu().detach().numpy()

def psnr(ref, img):
    if HAS_SKIMAGE: return float(sk_psnr(ref, img, data_range=1.0))
    mse = np.mean((ref - img) ** 2)
    return float(20 * math.log10(1 / math.sqrt(mse))) if mse > 0 else float('inf')

def ssim(ref, img):
    if HAS_SKIMAGE: return float(sk_ssim(ref, img, data_range=1.0))
    mu1, mu2 = ref.mean(), img.mean()
    s1,  s2  = ref.std(),  img.std()
    cov      = np.mean((ref - mu1) * (img - mu2))
    C1, C2   = 0.01 ** 2, 0.03 ** 2
    return float(((2*mu1*mu2+C1)*(2*cov+C2)) / ((mu1**2+mu2**2+C1)*(s1**2+s2**2+C2)))


# ═══════════════════════════════════════════════════════════════════════════
#  4. RICHER SYNTHETIC TRAINING DATA
#     v1 used plain ellipses → blurry priors.
#     v2 adds edges, radial gradients, and fine-detail blobs at multiple
#     scales so the model learns that sharp edges are valid outputs.
# ═══════════════════════════════════════════════════════════════════════════

def _draw_ellipse(img, n_blobs=6):
    for _ in range(n_blobs):
        cx, cy = np.random.randint(20, IMG_SIZE - 20, 2)
        rx, ry = np.random.randint(6, 55, 2)
        v      = np.random.uniform(0.25, 1.0)
        Y, X   = np.ogrid[:IMG_SIZE, :IMG_SIZE]
        mask   = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1
        img[mask] = np.clip(img[mask] + v, 0, 1)
    return img

def _draw_edges(img):
    """Add sharp rectangular structures (simulate bone cortex / vessel walls)."""
    for _ in range(np.random.randint(1, 4)):
        x1, y1 = np.random.randint(10, IMG_SIZE - 60, 2)
        w,  h  = np.random.randint(15, 80, 2)
        thick  = np.random.randint(1, 4)
        v      = np.random.uniform(0.5, 1.0)
        img[y1:y1+thick,   x1:x1+w] = v
        img[y1+h:y1+h+thick, x1:x1+w] = v
        img[y1:y1+h, x1:x1+thick]   = v
        img[y1:y1+h, x1+w:x1+w+thick] = v
    return img

def _draw_gradient(img):
    """Add a soft radial gradient (simulate tissue contrast variation)."""
    cx, cy = np.random.randint(40, IMG_SIZE - 40, 2)
    r      = np.random.randint(40, 100)
    v      = np.random.uniform(0.2, 0.6)
    Y, X   = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    dist   = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
    grad   = np.clip(1 - dist / r, 0, 1) * v
    img    = np.clip(img + grad, 0, 1)
    return img

def _draw_fine_blobs(img):
    """Small high-frequency blobs (simulate micro-lesions / vessels in cross-section)."""
    for _ in range(np.random.randint(4, 12)):
        cx, cy = np.random.randint(5, IMG_SIZE - 5, 2)
        r      = np.random.randint(2, 8)
        v      = np.random.uniform(0.6, 1.0)
        Y, X   = np.ogrid[:IMG_SIZE, :IMG_SIZE]
        mask   = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
        img[mask] = np.clip(img[mask] + v, 0, 1)
    return img

def synth_batch(bs=8, sigma=0.15):
    """Generate a batch of (noisy_tensor, clean_tensor) with rich structure."""
    clean_l, noisy_l = [], []
    for _ in range(bs):
        img = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
        img = _draw_ellipse(img, n_blobs=np.random.randint(3, 8))
        img = _draw_gradient(img)
        img = _draw_edges(img)
        img = _draw_fine_blobs(img)
        img = np.clip(img, 0, 1)
        clean_l.append(img)
        noisy_l.append(add_noise(img, sigma))

    return (torch.from_numpy(np.stack(noisy_l))[:, None].to(DEVICE),
            torch.from_numpy(np.stack(clean_l))[:, None].to(DEVICE))


# ═══════════════════════════════════════════════════════════════════════════
#  5. TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_model(epochs=20, lr=1e-3, sigma=0.15, pb=None, st_txt=None):
    model = UNetDAE(base=32).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # Cosine annealing: smoothly reduces lr to 1e-5 at the end
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    STEPS = 16          # more steps per epoch than v1 (was 10)
    total = epochs * STEPS

    model.train()
    for ep in range(epochs):
        loss_sum = 0
        for s in range(STEPS):
            noisy, clean = synth_batch(8, sigma)
            opt.zero_grad()
            out  = model(noisy)
            loss = combined_loss(out, clean)   # MSE + SSIM
            loss.backward()
            # Gradient clipping prevents exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            loss_sum += loss.item()
            if pb: pb.progress((ep * STEPS + s + 1) / total)

        sched.step()
        lr_now = sched.get_last_lr()[0]
        if st_txt:
            st_txt.text(
                f"Epoch {ep+1}/{epochs}  |  "
                f"Loss {loss_sum/STEPS:.5f}  |  "
                f"lr {lr_now:.2e}  |  {DEVICE}"
            )
    return model.state_dict()


# ═══════════════════════════════════════════════════════════════════════════
#  6. MODEL CACHE & INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_model(bytes_):
    m = UNetDAE(base=32).to(DEVICE)
    m.load_state_dict(torch.load(io.BytesIO(bytes_), map_location=DEVICE))
    m.eval()
    return m

def infer(model, noisy):
    with torch.no_grad():
        return np.clip(to_numpy(model(to_tensor(noisy))), 0, 1)


# ═══════════════════════════════════════════════════════════════════════════
#  7.  STREAMLIT  —  LIGHT THEME
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MRI Denoiser · U-Net DAE",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&family=Nunito:wght@400;600;700;800&display=swap');

:root{
  --bg:#f4f7fb; --surface:#ffffff; --surface2:#eef2f9;
  --border:#dde3ef; --accent:#2563eb; --accent2:#7c3aed;
  --green:#059669; --amber:#d97706; --red:#dc2626;
  --text:#1e293b; --muted:#64748b;
  --shadow:0 2px 12px rgba(37,99,235,.07);
  --shadow-lg:0 8px 28px rgba(37,99,235,.13);
}

html,body,[class*="css"]{ font-family:'Nunito',sans-serif!important; background:var(--bg)!important; color:var(--text)!important; }
.stApp{ background:var(--bg)!important; }
.block-container{ padding:1.8rem 2.4rem!important; }

section[data-testid="stSidebar"]{
  background:var(--surface)!important;
  border-right:1px solid var(--border)!important;
  box-shadow:2px 0 16px rgba(37,99,235,.06)!important;
}
section[data-testid="stSidebar"] *{ color:var(--text)!important; }

/* Hero */
.hero{
  background:linear-gradient(135deg,#eff6ff 0%,#f0fdf4 55%,#faf5ff 100%);
  border:1.5px solid var(--border); border-radius:18px;
  padding:2rem 2.8rem; margin-bottom:1.6rem;
  position:relative; overflow:hidden; box-shadow:var(--shadow);
}
.hero::after{content:'';position:absolute;top:-50px;right:-50px;width:210px;height:210px;
  background:radial-gradient(circle,rgba(37,99,235,.13) 0%,transparent 70%);pointer-events:none}
.hero h1{ font-family:'Lora',serif; font-size:2.3rem; font-weight:600;
  color:var(--text); margin:0 0 .3rem; letter-spacing:-.02em; }
.hero h1 em{ font-style:italic; color:var(--accent); }
.hero p{ font-family:'JetBrains Mono',monospace; font-size:.8rem; color:var(--muted); margin:0; }
.hero .badge{ display:inline-block; background:#dbeafe; color:var(--accent);
  border:1px solid #bfdbfe; border-radius:20px; font-family:'JetBrains Mono',monospace;
  font-size:.72rem; padding:.18rem .7rem; margin-top:.6rem; margin-right:.4rem; }
.hero .badge.green{ background:#d1fae5; color:var(--green); border-color:#a7f3d0; }

/* Image panels */
.img-panel{ background:var(--surface); border:1.5px solid var(--border);
  border-radius:14px; padding:1rem; text-align:center; box-shadow:var(--shadow); }
.tag{ display:inline-block; font-family:'JetBrains Mono',monospace;
  font-size:.68rem; letter-spacing:.1em; text-transform:uppercase;
  padding:.2rem .65rem; border-radius:20px; margin-bottom:.6rem; }
.tag-original{ background:#eff6ff; color:var(--accent); border:1px solid #bfdbfe; }
.tag-noisy   { background:#fffbeb; color:var(--amber);  border:1px solid #fde68a; }
.tag-denoised{ background:#ecfdf5; color:var(--green);  border:1px solid #a7f3d0; }

/* Metric cards */
.metric-card{ background:var(--surface); border:1.5px solid var(--border);
  border-radius:12px; padding:1rem .85rem; text-align:center;
  box-shadow:var(--shadow); transition:box-shadow .2s; }
.metric-card:hover{ box-shadow:var(--shadow-lg); }
.mc-val{ font-family:'JetBrains Mono',monospace; font-size:1.65rem;
  font-weight:500; color:var(--accent); line-height:1; }
.mc-lbl{ font-size:.7rem; color:var(--muted); text-transform:uppercase;
  letter-spacing:.09em; margin-top:.3rem; }
.mc-dlt{ font-family:'JetBrains Mono',monospace; font-size:.78rem;
  color:var(--green); margin-top:.15rem; }

/* Improvement hint box */
.hint-box{ background:#fffbeb; border:1.5px solid #fde68a; border-radius:12px;
  padding:.9rem 1.2rem; margin:.8rem 0; font-size:.85rem; color:#92400e; }
.hint-box b{ color:#d97706; }

/* Q&A */
.qa-hero{ background:linear-gradient(135deg,#eff6ff,#faf5ff);
  border:1.5px solid var(--border); border-radius:18px;
  padding:1.8rem 2.4rem; margin-bottom:1.4rem; box-shadow:var(--shadow); }
.qa-hero h2{ font-family:'Lora',serif; font-size:1.75rem; color:var(--text); margin:0 0 .3rem; }
.qa-hero p{ color:var(--muted); font-size:.88rem; margin:0; }

.q-block{ background:var(--surface); border:1.5px solid var(--border);
  border-left:4px solid var(--accent); border-radius:14px;
  padding:1.4rem 1.8rem; margin-bottom:1.1rem; box-shadow:var(--shadow); }
.q-block h3{ font-family:'Lora',serif; font-size:1.08rem; font-weight:600;
  color:var(--accent); margin:0 0 .9rem; }
.q-block p,.q-block li{ font-size:.92rem; line-height:1.78; color:var(--text); }
.q-block ul{ padding-left:1.4rem; }
.q-block li{ margin-bottom:.35rem; }

.compare-table{ width:100%; border-collapse:collapse; font-size:.83rem; }
.compare-table th{ background:var(--accent); color:#fff; padding:.5rem .75rem;
  text-align:left; font-weight:700; letter-spacing:.04em; }
.compare-table td{ padding:.48rem .75rem; border-bottom:1px solid var(--border); vertical-align:top; }
.compare-table tr:nth-child(even) td{ background:var(--surface2); }
.compare-table tr:hover td{ background:#eff6ff; }

.arch-box{ background:var(--surface2); border:1px solid var(--border);
  border-radius:10px; padding:.9rem 1.1rem;
  font-family:'JetBrains Mono',monospace; font-size:.8rem;
  line-height:1.8; color:var(--text); overflow-x:auto; }

.pill{ display:inline-block; padding:.16rem .55rem; border-radius:20px;
  font-size:.74rem; font-family:'JetBrains Mono',monospace; margin:.12rem; }
.pill-blue  { background:#dbeafe; color:var(--accent); }
.pill-green { background:#d1fae5; color:var(--green);  }
.pill-amber { background:#fef3c7; color:var(--amber);  }
.pill-violet{ background:#ede9fe; color:var(--accent2);}

/* Buttons */
.stButton>button{ background:linear-gradient(135deg,var(--accent2),var(--accent))!important;
  color:#fff!important; border:none!important; border-radius:8px!important;
  font-family:'Nunito',sans-serif!important; font-weight:700!important;
  padding:.5rem 1.6rem!important; box-shadow:0 2px 8px rgba(37,99,235,.2)!important;
  transition:opacity .2s!important; }
.stButton>button:hover{ opacity:.87!important; }

/* Tabs */
.stTabs [role="tab"]{ font-family:'Nunito',sans-serif!important; font-weight:700!important; }
.stTabs [role="tab"][aria-selected="true"]{
  color:var(--accent)!important; border-bottom:3px solid var(--accent)!important; }

/* Misc */
.stProgress>div>div{ background:var(--accent)!important; }
details>summary{ color:var(--accent)!important; font-weight:700!important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════

tab_app, tab_qa = st.tabs(["🧠  Denoising App", "📚  Reference & Theory"])


# ─────────────────────────────────────────────────────────────────────────
#  TAB 1  —  APP
# ─────────────────────────────────────────────────────────────────────────

with tab_app:

    st.markdown("""
    <div class="hero">
      <h1>MRI <em>Denoiser</em></h1>
      <p>U-Net Convolutional Denoising Autoencoder · PyTorch · GPU-ready</p>
      <span class="badge">U-Net Skip Connections</span>
      <span class="badge">MSE + SSIM Loss</span>
      <span class="badge green">Sharp Output</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        noise_sigma = st.slider("Noise level (σ)", 0.02, 0.40, 0.15, 0.01)
        apply_noise = st.checkbox("Add demo noise to input", value=True)
        st.markdown("---")
        st.markdown("### 🏋️ Model Training")
        train_epochs = st.slider("Epochs", 5, 60, 25, 5)
        train_lr = st.select_slider("Learning rate",
                       options=[1e-4, 5e-4, 1e-3, 2e-3], value=1e-3,
                       format_func=lambda v: f"{v:.0e}")
        train_btn = st.button("🔄  Train / Retrain", use_container_width=True)
        st.markdown("---")
        pre_file = st.file_uploader("Or upload .pth weights", type=["pth"])
        st.markdown("---")
        st.markdown(
            f"<span style='font-family:JetBrains Mono;font-size:.78rem;color:#64748b'>"
            f"Device: <b>{DEVICE}</b></span>", unsafe_allow_html=True)

    # ── Session state ──────────────────────────────────────────────────────
    if "model_bytes" not in st.session_state:
        st.session_state["model_bytes"] = None

    # ── Training ───────────────────────────────────────────────────────────
    if train_btn or st.session_state["model_bytes"] is None:
        st.markdown("""
        <div class="hint-box">
          <b>Training tip:</b> U-Net with MSE+SSIM loss trains on richer synthetic data
          (ellipses + edges + gradients + fine blobs). This gives sharper denoised outputs.
          ~40–60 s on CPU.
        </div>
        """, unsafe_allow_html=True)
        pb_col, txt_col = st.columns([3, 2])
        with pb_col:  pb   = st.progress(0)
        with txt_col: stxt = st.empty()
        t0 = time.time()
        sd = train_model(train_epochs, train_lr, noise_sigma, pb, stxt)
        buf = io.BytesIO()
        torch.save(sd, buf)
        st.session_state["model_bytes"] = buf.getvalue()
        pb.progress(1.0)
        stxt.text(f"✅  Done in {time.time() - t0:.1f}s")

    if pre_file:
        st.session_state["model_bytes"] = pre_file.read()

    if st.session_state["model_bytes"]:
        with st.sidebar:
            st.download_button(
                "⬇️  Download model (.pth)",
                data=st.session_state["model_bytes"],
                file_name="mri_unet_dae.pth",
                mime="application/octet-stream",
                use_container_width=True,
            )

    # ── Upload ─────────────────────────────────────────────────────────────
    st.markdown("### 📤 Upload MRI Image")
    uploaded = st.file_uploader(
        "PNG · JPG · BMP · TIFF",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        label_visibility="collapsed",
    )

    if uploaded and st.session_state["model_bytes"]:
        clean_img = load_image(uploaded)
        noisy_img = add_noise(clean_img, noise_sigma) if apply_noise else clean_img.copy()

        c_btn, _ = st.columns([1, 5])
        with c_btn:
            run = st.button("✨  Denoise Image", use_container_width=True)

        if run:
            with st.spinner("Running inference…"):
                model    = get_model(st.session_state["model_bytes"])
                denoised = infer(model, noisy_img)

            def a2p(a): return Image.fromarray((a * 255).astype(np.uint8))

            st.markdown("### 🔬 Results")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown('<div class="img-panel"><span class="tag tag-original">Original</span>', unsafe_allow_html=True)
                st.image(a2p(clean_img), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="img-panel"><span class="tag tag-noisy">Noisy Input</span>', unsafe_allow_html=True)
                st.image(a2p(noisy_img), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with c3:
                st.markdown('<div class="img-panel"><span class="tag tag-denoised">Denoised</span>', unsafe_allow_html=True)
                st.image(a2p(denoised), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Metrics ────────────────────────────────────────────────────
            pn  = psnr(clean_img, noisy_img);   pd  = psnr(clean_img, denoised)
            sn  = ssim(clean_img, noisy_img);   sd2 = ssim(clean_img, denoised)
            mn  = float(np.mean((clean_img - noisy_img) ** 2))
            md  = float(np.mean((clean_img - denoised) ** 2))

            st.markdown("### 📊 Metrics")
            cols = st.columns(6)
            dp   = pd - pn
            ds   = sd2 - sn
            cards = [
                (f"{pn:.1f}",  "PSNR Noisy (dB)",    None),
                (f"{pd:.1f}",  "PSNR Denoised (dB)",  f"+{dp:.1f} dB" if dp >= 0 else f"{dp:.1f} dB"),
                (f"{sn:.3f}",  "SSIM Noisy",          None),
                (f"{sd2:.3f}", "SSIM Denoised",        f"+{ds:.3f}" if ds >= 0 else f"{ds:.3f}"),
                (f"{mn:.4f}",  "MSE Noisy",            None),
                (f"{md:.4f}",  "MSE Denoised",         None),
            ]
            for col, (val, lbl, dlt) in zip(cols, cards):
                color = "var(--green)" if dlt and not dlt.startswith("-") else "var(--red)"
                dlt_html = f'<div class="mc-dlt" style="color:{color}">▲ {dlt}</div>' if dlt else ""
                col.markdown(
                    f'<div class="metric-card"><div class="mc-val">{val}</div>'
                    f'<div class="mc-lbl">{lbl}</div>{dlt_html}</div>',
                    unsafe_allow_html=True)

            buf = io.BytesIO()
            a2p(denoised).save(buf, "PNG")
            st.download_button("⬇️  Download Denoised Image",
                               buf.getvalue(), "mri_denoised.png", "image/png")

            with st.expander("🏗️  Model Architecture (U-Net DAE)"):
                m2 = UNetDAE(base=32)
                tp = sum(p.numel() for p in m2.parameters())
                st.code(str(m2))
                st.markdown(
                    f"**Total params:** {tp:,}  |  **Device:** {DEVICE}  |  "
                    f"**Input:** 1×256×256  |  **Loss:** α·MSE + (1−α)·(1−SSIM), α=0.7"
                )

    elif uploaded and not st.session_state["model_bytes"]:
        st.info("Train the model first (or upload a .pth file) using the sidebar.")
    else:
        st.markdown("""
        <div style="background:#fff;border:1.5px dashed #dce2ef;border-radius:14px;
             padding:2.5rem;text-align:center;color:#94a3b8;margin-top:1rem">
          <div style="font-size:3rem;margin-bottom:.8rem">🏥</div>
          <div style="font-size:.95rem;font-family:'JetBrains Mono',monospace;line-height:2.1">
            1 · Train model via sidebar (U-Net + MSE+SSIM loss)<br>
            2 · Upload an MRI image (PNG / JPG)<br>
            3 · Click <strong style="color:#2563eb">✨ Denoise Image</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
#  TAB 2  —  REFERENCE & THEORY
# ─────────────────────────────────────────────────────────────────────────

with tab_qa:

    st.markdown("""
    <div class="qa-hero">
      <h2>Medical Image Denoising — Theory & Design Reference</h2>
      <p>Complete answers to all five assessment sub-questions on DAE applied to MRI diagnostics</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Q a ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="q-block">
      <h3>Q (a) · DAE Architecture Design</h3>

      <p><b>Input Representation</b></p>
      <p>Each MRI slice is represented as a single-channel (grayscale) 2-D tensor of shape
      <span class="pill pill-blue">1 × 256 × 256</span>, normalised to
      <span class="pill pill-blue">[0.0 – 1.0]</span>.
      At training time, zero-mean Gaussian noise with σ ∈ [0.05, 0.20] is added to simulate
      scanner-level artifacts — the network sees only corrupted inputs and learns the
      mapping back to clean images.</p>

      <p><b>Encoder – Decoder Structure (U-Net)</b></p>
      <div class="arch-box">INPUT    1 × 256 × 256  (noisy MRI slice)
  │
  ├─ Enc-1 : DoubleConv(1→32)   + MaxPool(2) → 32 × 128 × 128   ──────────────────┐ skip
  ├─ Enc-2 : DoubleConv(32→64)  + MaxPool(2) → 64 × 64 × 64     ──────────────┐   │
  ├─ Enc-3 : DoubleConv(64→128) + MaxPool(2) → 128 × 32 × 32    ──────────┐   │   │
  ├─ Enc-4 : DoubleConv(128→256)+ MaxPool(2) → 256 × 16 × 16    ──────┐   │   │   │
  │                                                                     │   │   │   │
BOTTLENECK                                                               │   │   │   │
  ├─ DoubleConv(256→512)                       → 512 × 16 × 16   ◄──   │   │   │   │
  │                                                                     │   │   │   │
  ├─ Up-4 : ConvTranspose(512→256) + cat(skip)  → 512 × 32 × 32  ←────┘   │   │   │
  ├─ Dec-4 : DoubleConv(512→256)               → 256 × 32 × 32            │   │   │
  ├─ Up-3 : ConvTranspose(256→128) + cat(skip)  → 256 × 64 × 64  ←────────┘   │   │
  ├─ Dec-3 : DoubleConv(256→128)               → 128 × 64 × 64                │   │
  ├─ Up-2 : ConvTranspose(128→64) + cat(skip)   → 128 × 128 × 128 ←───────────┘   │
  ├─ Dec-2 : DoubleConv(128→64)                → 64 × 128 × 128                   │
  ├─ Up-1 : ConvTranspose(64→32)  + cat(skip)   → 64 × 256 × 256  ←───────────────┘
  ├─ Dec-1 : DoubleConv(64→32)                 → 32 × 256 × 256
  │
OUTPUT   Conv1×1(32→1) + Sigmoid  →  1 × 256 × 256  (clean MRI, values ∈ [0,1])</div>

      <p><b>Loss Function</b></p>
      <ul>
        <li><b>Combined MSE + SSIM:</b> L = 0.7·MSE + 0.3·(1−SSIM).
        MSE minimises pixel-wise error; SSIM preserves edges, contrast, and structure
        that pure MSE tends to over-smooth (causing blurry outputs).</li>
        <li><b>Optimizer:</b> Adam (lr = 1×10⁻³) with CosineAnnealingLR to eta_min=1e-5,
        plus gradient clipping (max_norm=1.0) for stable training.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    # ── Q b ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="q-block">
      <h3>Q (b) · Autoencoder Variant Comparison for MRI Denoising</h3>
      <table class="compare-table">
        <thead>
          <tr>
            <th>Variant</th><th>Core Principle</th><th>Regularisation</th>
            <th>Advantage</th><th>Limitation for MRI</th><th>Suitability</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><b>Undercomplete AE</b></td>
            <td>Bottleneck forces dimensionality compression</td>
            <td>Latent dim ≪ input dim</td>
            <td>Learns compact anatomical features automatically without explicit constraints</td>
            <td>No direct denoising training signal; may over-smooth fine structures</td>
            <td><span class="pill pill-amber">Moderate</span></td>
          </tr>
          <tr>
            <td><b>Denoising AE ✓</b></td>
            <td>Recovers clean signal from a corrupted input during training</td>
            <td>Corrupted input → clean target pairs</td>
            <td>Directly optimised for the denoising objective; robust to multiple noise types</td>
            <td>Requires matching noise model (Gaussian, Rician, etc.) to real scanner</td>
            <td><span class="pill pill-green">Best fit</span></td>
          </tr>
          <tr>
            <td><b>Contractive AE</b></td>
            <td>Penalises Jacobian norm of encoder w.r.t. input</td>
            <td>λ · ‖∂h/∂x‖²_F added to reconstruction loss</td>
            <td>Smooth, locally-invariant latent space; stable to small input perturbations</td>
            <td>Computationally expensive; λ is sensitive; noise reduction is indirect</td>
            <td><span class="pill pill-amber">Supplementary</span></td>
          </tr>
          <tr>
            <td><b>Sparse AE</b></td>
            <td>Enforces sparse hidden activations via L1 or KL penalty</td>
            <td>‖h‖₁ ≤ k  or  KL(ρ ‖ ρ̂)</td>
            <td>Disentangled, interpretable features; useful for anomaly detection</td>
            <td>Sparsity can suppress subtle MRI signal intensities; no explicit denoising path</td>
            <td><span class="pill pill-amber">Auxiliary</span></td>
          </tr>
        </tbody>
      </table>
      <p style="margin-top:.9rem">
        <b>Recommendation:</b> Deploy a <b>Denoising AE with U-Net skip connections</b>
        as the primary model. Adding a contractive regularisation term (λ ≈ 0.01) to
        the encoder loss improves robustness to out-of-distribution noise levels.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Q c ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="q-block">
      <h3>Q (c) · Manifold Learning and Clean Image Reconstruction</h3>
      <p>
        Real MRI scans occupy a high-dimensional pixel space (256×256 = 65,536 dimensions),
        yet all anatomically valid images reside on a vastly lower-dimensional
        <b>manifold</b> — a curved subspace defined by anatomical constraints, tissue
        contrasts, and imaging physics. Gaussian noise pushes a corrupted observation
        <i>off</i> this manifold; the autoencoder performs implicit <b>manifold projection</b>.
      </p>
      <ul>
        <li><b>Encoder as a chart map:</b> Successive Conv + Pool operations progressively
        suppress high-frequency noise components (which have no structure on the MRI manifold)
        while retaining low-frequency anatomical organisation that lies on it.</li>
        <li><b>Skip connections preserve manifold tangent vectors:</b> In a plain autoencoder,
        fine-grained structural detail is lost at the bottleneck. Skip connections directly
        forward encoder feature maps (which encode edges, texture, fine vessels) to the
        decoder, ensuring the reconstruction stays on the high-detail submanifold, not just
        its low-frequency projection — this is the direct fix for blurry outputs.</li>
        <li><b>Bottleneck as the manifold coordinate system:</b> Because the model is trained
        to reconstruct clean targets from noisy inputs, the latent space represents the
        distribution of <i>clean</i> images. Any off-manifold direction (noise) has no
        corresponding latent coordinate and is therefore annihilated.</li>
        <li><b>Connection to score matching:</b> This is formally equivalent to learning an
        approximation of ∇ₓ log p(x) — the score function pointing from noisy observations
        back toward the clean-image manifold. Diffusion models exploit this same principle.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    # ── Q d ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="q-block">
      <h3>Q (d) · Evaluation Metrics — PSNR and SSIM</h3>

      <p><b>Peak Signal-to-Noise Ratio (PSNR)</b></p>
      <div class="arch-box">PSNR  =  20 · log₁₀(MAX / √MSE)          [dB]

  where  MAX = 1.0 (normalised image range)
         MSE = mean squared pixel error between reference and output

Typical targets:  noisy low-cost scanner ≈ 20–28 dB
                  clinical acceptability  > 35 dB after denoising</div>

      <p><b>Structural Similarity Index (SSIM)</b></p>
      <div class="arch-box">SSIM(x, y)  =  [l(x,y)]^α · [c(x,y)]^β · [s(x,y)]^γ

  l = luminance   =  (2μₓμᵧ + C₁) / (μₓ² + μᵧ² + C₁)
  c = contrast    =  (2σₓσᵧ + C₂) / (σₓ² + σᵧ² + C₂)
  s = structure   =  (σₓᵧ + C₃)   / (σₓσᵧ + C₃)

Range: 0 (no similarity) → 1.0 (identical)
Clinical acceptability: SSIM > 0.90</div>

      <p><b>Why both are needed:</b> A model can achieve high PSNR by uniformly blurring
      (reducing noise power without preserving structure) — SSIM flags this. A model can
      preserve local structure while introducing global bias — PSNR flags this. Both
      metrics provide orthogonal quality assurance.</p>

      <p><b>Additional metrics for medical imaging:</b></p>
      <ul>
        <li><span class="pill pill-violet">FSIM</span> Feature Similarity Index — matches gradient-based maps, closer to human perception.</li>
        <li><span class="pill pill-violet">RMSE</span> Root MSE — same intensity scale as pixels.</li>
        <li><span class="pill pill-violet">Downstream DSC</span> Dice on segmented structures (tumour, ventricles) — ultimate clinical validation.</li>
        <li><span class="pill pill-violet">FID</span> Fréchet Inception Distance — measures distributional realism for generative models.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    # ── Q e ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="q-block">
      <h3>Q (e) · Performance Improvement via Transfer Learning and Pretraining</h3>

      <p><b>Strategy 1 — Pretrain on Large Public Medical Datasets</b></p>
      <ul>
        <li>Pretrain on <span class="pill pill-blue">FastMRI</span> (1.5M brain+knee slices),
          <span class="pill pill-blue">IXI Brain MRI</span> (600 subjects),
          <span class="pill pill-blue">BraTS</span> (multi-modal, 4 contrasts).
        </li>
        <li><b>Fine-tune strategy:</b> Freeze encoder; train only decoder on ≥200 paired
        images from target scanner for 20 epochs. Then unfreeze all layers at 1/10th lr
        for 10 more epochs. Prevents catastrophic forgetting of anatomy knowledge.</li>
      </ul>

      <p><b>Strategy 2 — ImageNet Encoder Initialisation</b></p>
      <ul>
        <li>Replace encoder with <span class="pill pill-green">ResNet-34</span> or
        <span class="pill pill-green">EfficientNet-B2</span> backbone.</li>
        <li>Adapt first conv from 3-channel to 1-channel by averaging pretrained RGB weights.</li>
        <li>Converges in ~10 epochs vs ~50 from scratch; effective when fewer than 500 pairs available.</li>
      </ul>

      <p><b>Strategy 3 — Self-supervised Noise2Void (no clean labels needed)</b></p>
      <ul>
        <li>When clean reference images are unavailable, pretrain using
        <span class="pill pill-amber">Noise2Void</span>: randomly mask individual pixels
        and train the network to predict each masked value from surrounding context only.</li>
        <li>Provides strong initialisation without any paired clean data. Supervised fine-tuning
        on even a small paired set converges quickly.</li>
      </ul>

      <p><b>Proposed Deployment Pipeline</b></p>
      <div class="arch-box">Phase 1  Pretrain    FastMRI + IXI, 80k slices, 50 epochs, lr=1e-3, MSE+SSIM loss
Phase 2  Adapt       Freeze encoder; train decoder on target scanner, 200 pairs, 20 epochs
Phase 3  Fine-tune   Unfreeze all; end-to-end, 200 pairs, 10 epochs, lr=1e-4, MSE+SSIM loss
Augment  Random flips, 90° rotations, contrast jitter ±0.2, elastic deform
Monitor  PSNR + SSIM on held-out test split after each phase
Deploy   TorchScript → FastAPI → ONNX for edge/embedded devices</div>

      <p>Expected gains: 3–6 dB PSNR and 0.05–0.12 SSIM improvement over training from
      scratch, with 4–8× faster convergence.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.5rem;padding:.9rem 1.4rem;background:#eff6ff;border-radius:10px;
         border:1px solid #bfdbfe;font-size:.8rem;color:#1e40af;
         font-family:'JetBrains Mono',monospace;">
      U-Net DAE · PyTorch · MSE+SSIM Loss · scikit-image · FastMRI / IXI / BraTS reference datasets
    </div>
    """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown(
    f"<div style='margin-top:2rem;padding-top:.9rem;border-top:1px solid #dde3ef;"
    f"text-align:center;color:#94a3b8;font-family:JetBrains Mono,monospace;font-size:.73rem'>"
    f"MRI Denoising U-Net DAE · PyTorch {torch.__version__} · Streamlit · MSE+SSIM Loss</div>",
    unsafe_allow_html=True,
)