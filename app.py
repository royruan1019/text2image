import streamlit as st
import anthropic
import requests
import random
import json
from io import BytesIO
from urllib.parse import quote
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

# ── Constants ──────────────────────────────────────────────────────────────────
STYLE_SUFFIXES = {
    "Photorealistic": ", photorealistic, DSLR photography, ultra sharp, 8K resolution, detailed textures",
    "Cinematic": ", cinematic shot, anamorphic lens, film grain, dramatic lighting, movie still",
    "Digital Art": ", digital art, concept art, trending on ArtStation, vivid colors, highly detailed",
    "Oil Painting": ", oil painting, thick brushstrokes, impasto technique, museum quality, masterpiece",
    "Watercolor": ", watercolor painting, soft washes, paper texture, elegant, delicate",
    "Anime": ", anime style, detailed illustration, Studio Ghibli inspired, soft colors",
    "3D Render": ", 3D render, octane render, subsurface scattering, ray tracing, photorealistic CGI",
    "Sketch": ", pencil sketch, clean line art, monochrome, hatching, detailed",
    "None": "",
}

ASPECT_OPTIONS = {
    "1:1  (1024×1024)": (1024, 1024),
    "16:9 (1280×720)":  (1280, 720),
    "9:16 (720×1280)":  (720,  1280),
    "4:3  (1024×768)":  (1024, 768),
    "3:4  (768×1024)":  (768,  1024),
}

EXAMPLE_PROMPTS = [
    "A lone astronaut standing on a red Martian cliff at golden hour, vast alien landscape",
    "Ancient Japanese temple surrounded by cherry blossoms in heavy rain, misty atmosphere",
    "Underwater city of glass and coral, bioluminescent creatures drifting through sunlit waters",
    "A wolf made entirely of northern lights, standing on a frozen tundra at midnight",
    "Futuristic Tokyo street market at night, neon reflections on wet pavement, cyberpunk",
]

# Pollinations models (all free, no key)
POLLINATIONS_MODELS = {
    "FLUX (best quality)": "flux",
    "FLUX Realism": "flux-realism",
    "FLUX Anime": "flux-anime",
    "FLUX 3D": "flux-3d",
    "Turbo (fastest)": "turbo",
}

# HuggingFace free inference models (free account token only, no credit card)
HF_FREE_MODELS = {
    "FLUX.1-schnell (fast, free)": "black-forest-labs/FLUX.1-schnell",
    "Stable Diffusion XL":         "stabilityai/stable-diffusion-xl-base-1.0",
    "Stable Diffusion 2.1":        "stabilityai/stable-diffusion-2-1",
}

GENERATION_MODES = [
    "🆓 Pollinations.ai  (no key needed)",
    "🤗 HuggingFace Free  (free token)",
    "🔮 Claude Preview  (description only)",
]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Text-to-Image · Free",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', system-ui, sans-serif; }

.stApp { background-color: #080818; color: #e8e8f8; }

section[data-testid="stSidebar"] {
    background-color: #060614;
    border-right: 1px solid rgba(99,102,241,0.1);
}

.cosmos-header {
    background: linear-gradient(160deg, #0e0e2a 0%, #14143a 60%, #0e0e2a 100%);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 16px;
}

.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 500;
    margin-right: 6px;
    margin-bottom: 4px;
}
.badge-free  { background:rgba(34,197,94,0.12); color:#86efac; border:1px solid rgba(34,197,94,0.25); }
.badge-model { background:rgba(99,102,241,0.12); color:#a5b4fc; border:1px solid rgba(99,102,241,0.25); }
.badge-link  { background:rgba(56,189,248,0.10); color:#7dd3fc; border:1px solid rgba(56,189,248,0.20); }

.how-to-box {
    background: rgba(99,102,241,0.05);
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 16px;
    font-size: 12px;
}

.prompt-preview {
    background: rgba(99,102,241,0.05);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 11px;
    font-family: monospace;
    color: rgba(255,255,255,0.55);
    word-break: break-word;
}

.result-card {
    background: #0d0d22;
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 14px;
}

.result-meta { padding: 14px 16px; }
.result-desc { font-size:13px; color:rgba(255,255,255,0.7); line-height:1.65; margin-bottom:10px; }
.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 100px;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.2);
    color: #a5b4fc;
    font-size: 11px;
    margin-right: 4px;
    margin-bottom: 4px;
}
.result-footer {
    margin-top:10px; padding-top:10px;
    border-top:1px solid rgba(255,255,255,0.06);
    color:rgba(255,255,255,0.25); font-size:11px; font-family:monospace;
}

.empty-state {
    display:flex; align-items:center; justify-content:center;
    flex-direction:column; gap:12px; min-height:200px;
    border:1px dashed rgba(99,102,241,0.15); border-radius:12px;
    color:rgba(255,255,255,0.2); padding:40px; text-align:center;
}

.stTextArea textarea, .stTextInput input {
    background-color:#0f0f28 !important;
    border:1px solid rgba(99,102,241,0.18) !important;
    color:rgba(255,255,255,0.8) !important;
    font-family:'DM Sans',sans-serif !important;
    border-radius:8px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color:rgba(99,102,241,0.5) !important;
    box-shadow:0 0 0 2px rgba(99,102,241,0.15) !important;
}
.stSelectbox select, div[data-baseweb="select"] {
    background-color:#0f0f28 !important;
    color:rgba(255,255,255,0.8) !important;
}
.stButton > button {
    background:linear-gradient(135deg,#6366f1,#4338ca) !important;
    color:white !important; border:none !important;
    border-radius:10px !important; font-size:15px !important;
    font-weight:600 !important; padding:14px 28px !important;
    width:100% !important; letter-spacing:0.02em !important;
    box-shadow:0 4px 20px rgba(99,102,241,0.3) !important;
    transition:all 0.2s !important;
}
.stButton > button:hover {
    box-shadow:0 6px 28px rgba(99,102,241,0.45) !important;
    transform:translateY(-1px) !important;
}
label[data-testid="stWidgetLabel"], .stSlider label {
    color:rgba(255,255,255,0.45) !important;
    font-size:11px !important; text-transform:uppercase !important;
    letter-spacing:0.08em !important; font-weight:500 !important;
}
hr { border-color:rgba(255,255,255,0.06) !important; }
.stAlert { background-color:rgba(239,68,68,0.08) !important; border:1px solid rgba(239,68,68,0.2) !important; color:#fca5a5 !important; border-radius:8px !important; }

.sidebar-section-title {
    color:#a5b4fc; font-size:11px; font-weight:600;
    letter-spacing:0.08em; text-transform:uppercase;
    margin:12px 0 6px 0;
}
.model-info {
    color:rgba(255,255,255,0.3); font-size:11px; line-height:1.7;
    background:rgba(99,102,241,0.04); border:1px solid rgba(99,102,241,0.08);
    border-radius:8px; padding:10px 12px; margin-top:8px;
}
.model-info-title { color:#a5b4fc; font-size:11px; font-weight:600; margin-bottom:4px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section-title">⚡ Generation Mode</div>', unsafe_allow_html=True)
    mode = st.radio("Mode", GENERATION_MODES, index=0, label_visibility="collapsed")

    st.divider()

    if mode == GENERATION_MODES[0]:   # Pollinations
        st.markdown('<div class="sidebar-section-title">🎛️ Pollinations Model</div>', unsafe_allow_html=True)
        poll_model_label = st.selectbox("Model", list(POLLINATIONS_MODELS.keys()),
                                        label_visibility="collapsed")
        poll_model = POLLINATIONS_MODELS[poll_model_label]
        st.markdown("""
        <div class="model-info">
            <div class="model-info-title">ℹ️ Pollinations.ai</div>
            Completely free · No signup · No API key<br>
            Powered by FLUX & open-source models
        </div>""", unsafe_allow_html=True)

    elif mode == GENERATION_MODES[1]: # HuggingFace free
        st.markdown('<div class="sidebar-section-title">🤗 HuggingFace Model</div>', unsafe_allow_html=True)
        hf_model_label = st.selectbox("Model", list(HF_FREE_MODELS.keys()),
                                      label_visibility="collapsed")
        hf_model = HF_FREE_MODELS[hf_model_label]
        hf_token = st.text_input("HuggingFace Token", type="password",
                                 placeholder="hf_xxxxxxxxxx",
                                 help="Free token — sign up at huggingface.co, no credit card needed.")
        st.markdown("""
        <div class="model-info">
            <div class="model-info-title">ℹ️ Free Token</div>
            Sign up at huggingface.co<br>
            Settings → Access Tokens → New token<br>
            No credit card required
        </div>""", unsafe_allow_html=True)

    else:                             # Claude preview
        claude_api_key = st.text_input("Claude API Key", type="password",
                                       placeholder="sk-ant-xxxxxxxxxx",
                                       help="console.anthropic.com — new accounts get free credits.")
        st.markdown("""
        <div class="model-info">
            <div class="model-info-title">ℹ️ Preview Mode</div>
            Claude describes what the image<br>
            would look like — no real image.<br>
            New accounts get free API credits.
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="sidebar-section-title">🎨 Style</div>', unsafe_allow_html=True)
    style = st.selectbox("Style", list(STYLE_SUFFIXES.keys()), label_visibility="collapsed")

    st.markdown('<div class="sidebar-section-title">📐 Aspect Ratio</div>', unsafe_allow_html=True)
    ratio_label = st.selectbox("Aspect Ratio", list(ASPECT_OPTIONS.keys()),
                               index=1, label_visibility="collapsed")
    w, h = ASPECT_OPTIONS[ratio_label]

    st.markdown('<div class="sidebar-section-title">🖼️ Images</div>', unsafe_allow_html=True)
    num_images = st.slider("Number of Images", 1, 4, 2)

    st.divider()
    st.markdown('<div class="sidebar-section-title">🎲 Seed</div>', unsafe_allow_html=True)
    random_seed = st.checkbox("Random Seed", value=True)
    seed_val = st.number_input("Seed Value", value=42, min_value=0,
                               max_value=2147483647, disabled=random_seed)

    st.markdown('<div class="sidebar-section-title">⚙️ Generation</div>', unsafe_allow_html=True)
    guidance = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
    steps = st.slider("Inference Steps", 4, 50, 20)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cosmos-header">
    <div style="font-family:'Space Mono',monospace;font-size:26px;font-weight:700;letter-spacing:-1px;margin-bottom:6px">
        🌌 <span style="background:linear-gradient(90deg,#818cf8,#38bdf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent">Text-to-Image</span>
        <span style="color:rgba(255,255,255,0.9);font-size:20px"> Generator</span>
    </div>
    <div style="color:rgba(255,255,255,0.45);font-size:13px;margin-bottom:14px">
        Free image generation · Pollinations.ai · HuggingFace FLUX · No cost
    </div>
    <span class="badge badge-free">✅ 100% Free</span>
    <span class="badge badge-model">⚡ FLUX · SDXL</span>
    <span class="badge badge-link">🔮 Claude Preview</span>
</div>
<div class="how-to-box">
    <span style="color:#a5b4fc;font-weight:600;font-size:11px;letter-spacing:.08em;text-transform:uppercase">How to use </span>
    <span style="color:rgba(255,255,255,0.4);font-size:12px">
        ① Pick a mode in the sidebar → ② Enter a prompt → ③ Set style &amp; params → ④ Click Generate
    </span>
</div>
""", unsafe_allow_html=True)

# ── Prompt area ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    hcol1, hcol2 = st.columns([4, 1])
    with hcol1:
        st.markdown("**✍️ Prompt**")
    with hcol2:
        if st.button("Example ↗", key="example_btn"):
            st.session_state["_prompt"] = random.choice(EXAMPLE_PROMPTS)
            st.rerun()

    prompt = st.text_area(
        "Prompt",
        value=st.session_state.get("_prompt", ""),
        placeholder="A lone astronaut on a red Martian cliff at golden hour, cinematic lighting...",
        height=120,
        label_visibility="collapsed",
    )

with col2:
    st.markdown("**🚫 Negative Prompt**")
    neg_prompt = st.text_area(
        "Negative Prompt",
        value="blurry, low quality, deformed, watermark, text, ugly",
        height=120,
        label_visibility="collapsed",
    )

if prompt.strip():
    full = prompt.strip() + STYLE_SUFFIXES[style]
    preview = full[:180] + ("…" if len(full) > 180 else "")
    st.markdown(f'<div class="prompt-preview"><span style="color:rgba(255,255,255,0.3)">Full prompt: </span>{preview}</div>',
                unsafe_allow_html=True)

# ── Generate ───────────────────────────────────────────────────────────────────
generate_clicked = st.button("🚀 Generate Images", use_container_width=True)

if generate_clicked:
    if not prompt.strip():
        st.error("⚠️ Please enter a prompt.")
    else:
        used_seed = random.randint(0, 2147483647) if random_seed else int(seed_val)
        full_prompt = prompt.strip() + STYLE_SUFFIXES[style]

        # ── MODE 1: Pollinations.ai ────────────────────────────────────────────
        if mode == GENERATION_MODES[0]:
            st.info(f"🆓 Pollinations.ai · model={poll_model} · {w}×{h} · seed={used_seed}")
            cols = st.columns(min(num_images, 2))
            with st.spinner("Generating…"):
                for i in range(num_images):
                    encoded = quote(full_prompt)
                    url = (
                        f"https://image.pollinations.ai/prompt/{encoded}"
                        f"?width={w}&height={h}&seed={used_seed+i}"
                        f"&model={poll_model}&nologo=true&enhance=true"
                    )
                    try:
                        resp = requests.get(url, timeout=120)
                        if resp.status_code == 200:
                            col_idx = i % 2
                            with cols[col_idx]:
                                st.image(BytesIO(resp.content),
                                         caption=f"Image {i+1} · seed={used_seed+i} · {poll_model}",
                                         use_container_width=True)
                        else:
                            st.error(f"Image {i+1}: HTTP {resp.status_code}")
                    except Exception as e:
                        st.error(f"Image {i+1} error: {e}")

        # ── MODE 2: HuggingFace free models ───────────────────────────────────
        elif mode == GENERATION_MODES[1]:
            if not hf_token.strip():
                st.error("⚠️ Enter your free HuggingFace token in the sidebar. Sign up at huggingface.co — no credit card.")
            else:
                st.info(f"🤗 {hf_model} · {w}×{h} · seed={used_seed}")
                cols = st.columns(min(num_images, 2))
                with st.spinner("Generating…"):
                    try:
                        client = InferenceClient(model=hf_model, token=hf_token.strip())
                        for i in range(num_images):
                            try:
                                kwargs = dict(
                                    width=w,
                                    height=h,
                                    num_inference_steps=steps,
                                    seed=used_seed + i,
                                )
                                # FLUX.1-schnell doesn't support guidance_scale or negative_prompt
                                if "flux" not in hf_model.lower():
                                    kwargs["guidance_scale"] = guidance
                                    kwargs["negative_prompt"] = neg_prompt

                                pil_img = client.text_to_image(full_prompt, **kwargs)
                                buf = BytesIO()
                                pil_img.save(buf, format="PNG")
                                buf.seek(0)
                                with cols[i % 2]:
                                    st.image(buf, caption=f"Image {i+1} · seed={used_seed+i}",
                                             use_container_width=True)
                            except HfHubHTTPError as e:
                                st.error(f"Image {i+1} — HF API error: {e}")
                            except Exception as e:
                                st.error(f"Image {i+1} error: {e}")
                    except Exception as e:
                        st.error(f"Connection failed: {e}")

        # ── MODE 3: Claude preview ─────────────────────────────────────────────
        else:
            if not claude_api_key.strip():
                st.error("⚠️ Enter your Claude API key in the sidebar.")
            else:
                st.info(f"🔮 Claude preview · seed={used_seed} · {w}×{h}")
                system_prompt = f"""You are simulating a text-to-image model.
Respond ONLY with a JSON array (no markdown) of exactly {num_images} objects, each with:
- "alt": vivid 2-sentence description of the generated image
- "tags": 4-5 short visual descriptor strings
- "seed": numeric seed used
Be specific and visual. Match the style in the prompt."""

                user_msg = (f'Generate {num_images} image description(s) for:\n"{full_prompt}"\n'
                            f'Negative: "{neg_prompt}"\nSeed: {used_seed}, Size: {w}x{h}')

                with st.spinner("Generating via Claude…"):
                    try:
                        ac = anthropic.Anthropic(api_key=claude_api_key)
                        msg = ac.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=1200,
                            system=system_prompt,
                            messages=[{"role": "user", "content": user_msg}],
                        )
                        raw = msg.content[0].text.replace("```json", "").replace("```", "").strip()
                        descs = json.loads(raw)

                        cols = st.columns(min(num_images, 2))
                        for i, d in enumerate(descs):
                            tags_html = "".join(f'<span class="tag">{t}</span>' for t in (d.get("tags") or []))
                            with cols[i % 2]:
                                st.markdown(f"""
                                <div class="result-card">
                                    <div style="height:12px;background:linear-gradient(135deg,#1a0533,#0d1f4a,#0a2a1f)"></div>
                                    <div class="result-meta">
                                        <div class="result-desc">{d.get("alt","")}</div>
                                        <div>{tags_html}</div>
                                        <div class="result-footer">Image {i+1} · seed={d.get("seed", used_seed+i)} · {w}×{h}</div>
                                    </div>
                                </div>""", unsafe_allow_html=True)

                        st.info("🔮 Preview mode: these are text descriptions, not real images. Switch to Pollinations.ai for free real images.")
                    except json.JSONDecodeError:
                        st.error("Failed to parse Claude response.")
                    except anthropic.AuthenticationError:
                        st.error("Invalid Claude API key.")
                    except Exception as e:
                        st.error(f"Generation failed: {e}")

else:
    st.markdown("""
    <div class="empty-state">
        <div style="font-size:48px">🌌</div>
        <div style="font-size:15px">Enter a prompt and click Generate</div>
        <div style="font-size:12px;color:rgba(255,255,255,0.12);margin-top:4px">
            🆓 Pollinations.ai is selected by default — no API key required
        </div>
    </div>""", unsafe_allow_html=True)
