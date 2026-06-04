import streamlit as st
import anthropic
import random
import json
from io import BytesIO
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
    "1:1 (1024×1024)": (1024, 1024),
    "16:9 (1280×720)": (1280, 720),
    "9:16 (720×1280)": (720, 1280),
    "4:3 (1024×768)": (1024, 768),
    "3:4 (768×1024)": (768, 1024),
}

EXAMPLE_PROMPTS = [
    "A lone astronaut standing on a red Martian cliff at golden hour, vast alien landscape",
    "Ancient Japanese temple surrounded by cherry blossoms in heavy rain, misty atmosphere",
    "Underwater city of glass and coral, bioluminescent creatures drifting through sunlit waters",
    "A wolf made entirely of northern lights, standing on a frozen tundra at midnight",
    "Futuristic Tokyo street market at night, neon reflections on wet pavement, cyberpunk",
]

HF_MODEL = "nvidia/Cosmos3-Super-Text2Image"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cosmos3 Text-to-Image",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', system-ui, sans-serif;
}

.stApp {
    background-color: #080818;
    color: #e8e8f8;
}

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

.cosmos-title {
    font-family: 'Space Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -1px;
    margin-bottom: 6px;
}

.cosmos-subtitle {
    color: rgba(255,255,255,0.45);
    font-size: 13px;
    margin-bottom: 14px;
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

.badge-model {
    background: rgba(99,102,241,0.12);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.25);
}

.badge-link {
    background: rgba(56,189,248,0.1);
    color: #7dd3fc;
    border: 1px solid rgba(56,189,248,0.2);
}

.how-to-box {
    background: rgba(99,102,241,0.05);
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 16px;
    font-size: 12px;
}

.how-to-label {
    color: #a5b4fc;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
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

.result-meta {
    padding: 14px 16px;
}

.result-desc {
    font-size: 13px;
    color: rgba(255,255,255,0.7);
    line-height: 1.65;
    margin-bottom: 10px;
}

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
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid rgba(255,255,255,0.06);
    color: rgba(255,255,255,0.25);
    font-size: 11px;
    font-family: monospace;
}

.preview-note {
    background: rgba(56,189,248,0.05);
    border: 1px solid rgba(56,189,248,0.12);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 12px;
    color: rgba(255,255,255,0.4);
    margin-top: 14px;
}

.empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 12px;
    min-height: 200px;
    border: 1px dashed rgba(99,102,241,0.15);
    border-radius: 12px;
    color: rgba(255,255,255,0.2);
    padding: 40px;
    text-align: center;
}

.stTextArea textarea, .stTextInput input {
    background-color: #0f0f28 !important;
    border: 1px solid rgba(99,102,241,0.18) !important;
    color: rgba(255,255,255,0.8) !important;
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 8px !important;
}

.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
}

.stSelectbox select, div[data-baseweb="select"] {
    background-color: #0f0f28 !important;
    color: rgba(255,255,255,0.8) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4338ca) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 14px 28px !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    box-shadow: 0 6px 28px rgba(99,102,241,0.45) !important;
    transform: translateY(-1px) !important;
}

.stSlider > div > div > div {
    background: rgba(99,102,241,0.25) !important;
}

div[data-testid="stMetricValue"] {
    color: #a5b4fc !important;
}

label[data-testid="stWidgetLabel"], .stSlider label {
    color: rgba(255,255,255,0.45) !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-weight: 500 !important;
}

hr {
    border-color: rgba(255,255,255,0.06) !important;
}

.stAlert {
    background-color: rgba(239,68,68,0.08) !important;
    border: 1px solid rgba(239,68,68,0.2) !important;
    color: #fca5a5 !important;
    border-radius: 8px !important;
}

.stInfo {
    background-color: rgba(99,102,241,0.05) !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    color: rgba(255,255,255,0.6) !important;
}

div[data-testid="stSidebarContent"] label {
    color: rgba(255,255,255,0.45) !important;
}

.sidebar-section-title {
    color: #a5b4fc;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 12px 0 6px 0;
}

.model-info {
    color: rgba(255,255,255,0.3);
    font-size: 11px;
    line-height: 1.7;
    background: rgba(99,102,241,0.04);
    border: 1px solid rgba(99,102,241,0.08);
    border-radius: 8px;
    padding: 10px 12px;
    margin-top: 8px;
}

.model-info-title {
    color: #a5b4fc;
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section-title">🔑 API Keys</div>', unsafe_allow_html=True)

    claude_api_key = st.text_input(
        "Claude API Key",
        type="password",
        placeholder="sk-ant-xxxxxxxxxx",
        help="Required for preview mode. Get yours at console.anthropic.com",
    )

    hf_token = st.text_input(
        "HuggingFace Token",
        type="password",
        placeholder="hf_xxxxxxxxxx",
        help="Required for real image generation. Leave blank for Claude preview mode.",
    )

    hf_endpoint = st.text_input(
        "Custom Endpoint URL (optional)",
        placeholder="https://xyz.endpoints.huggingface.cloud",
        help="Paste a dedicated HF Inference Endpoint URL. Required for large models like Cosmos3 (64B) that aren't on the free API.",
    )

    st.divider()
    st.markdown('<div class="sidebar-section-title">🎨 Image Style</div>', unsafe_allow_html=True)

    style = st.selectbox(
        "Style",
        list(STYLE_SUFFIXES.keys()),
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-section-title">📐 Aspect Ratio</div>', unsafe_allow_html=True)

    ratio_label = st.selectbox(
        "Aspect Ratio",
        list(ASPECT_OPTIONS.keys()),
        index=1,
        label_visibility="collapsed",
    )
    w, h = ASPECT_OPTIONS[ratio_label]

    st.markdown('<div class="sidebar-section-title">🖼️ Images</div>', unsafe_allow_html=True)
    num_images = st.slider("Number of Images", min_value=1, max_value=4, value=2)

    st.divider()
    st.markdown('<div class="sidebar-section-title">🎲 Seed</div>', unsafe_allow_html=True)
    random_seed = st.checkbox("Random Seed", value=True)
    seed_val = st.number_input("Seed Value", value=42, min_value=0, max_value=2147483647,
                               disabled=random_seed, label_visibility="visible")

    st.markdown('<div class="sidebar-section-title">⚙️ Generation</div>', unsafe_allow_html=True)
    guidance = st.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=7.5, step=0.5)
    steps = st.slider("Inference Steps", min_value=10, max_value=50, value=28)

    st.divider()
    st.markdown("""
    <div class="model-info">
        <div class="model-info-title">MODEL INFO</div>
        nvidia/Cosmos3-Super-Text2Image<br>
        Params: 64B · MoT Architecture<br>
        License: OpenMDW 1.1<br>
        Released: 2026-05-31
    </div>
    """, unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cosmos-header">
    <div class="cosmos-title">
        🌌 <span style="background:linear-gradient(90deg,#818cf8,#38bdf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent">Cosmos3</span>
        <span style="color:rgba(255,255,255,0.9);font-size:20px"> Text-to-Image</span>
    </div>
    <div class="cosmos-subtitle">Powered by NVIDIA Cosmos3-Super-Text2Image · 64B Mixture-of-Transformers</div>
    <span class="badge badge-model">🤖 nvidia/Cosmos3-Super-Text2Image</span>
    <span class="badge badge-link">⇥ HuggingFace</span>
    <span class="badge badge-link">⇥ Streamlit Demo</span>
</div>
<div class="how-to-box">
    <span class="how-to-label">How to use </span>
    <span style="color:rgba(255,255,255,0.4)">
        ① Enter a prompt → ② Set style &amp; parameters in the sidebar → ③ Click Generate → ④ View results
    </span>
</div>
""", unsafe_allow_html=True)

# ── Prompt area ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    pcol1, pcol2 = st.columns([4, 1])
    with pcol1:
        st.markdown("**✍️ Prompt**")
    with pcol2:
        if st.button("Example ↗", key="example_btn", help="Fill with a random example prompt"):
            st.session_state["prompt"] = random.choice(EXAMPLE_PROMPTS)

    prompt = st.text_area(
        "Prompt",
        value=st.session_state.get("prompt", ""),
        placeholder="A lone astronaut on a red Martian cliff at golden hour, cinematic lighting, ultra detailed...",
        height=120,
        label_visibility="collapsed",
        key="prompt_input",
    )

with col2:
    st.markdown("**🚫 Negative Prompt**")
    neg_prompt = st.text_area(
        "Negative Prompt",
        value="blurry, low quality, deformed, watermark, text, ugly",
        height=120,
        label_visibility="collapsed",
    )

# Full prompt preview
if prompt.strip():
    full_prompt = prompt.strip() + STYLE_SUFFIXES[style]
    preview_text = full_prompt[:180] + ("…" if len(full_prompt) > 180 else "")
    st.markdown(f"""
    <div class="prompt-preview">
        <span style="color:rgba(255,255,255,0.3)">Full prompt: </span>{preview_text}
    </div>
    """, unsafe_allow_html=True)

# ── Generate button ────────────────────────────────────────────────────────────
generate_clicked = st.button("🚀 Generate Images", use_container_width=True)

# ── Generation logic ───────────────────────────────────────────────────────────
if generate_clicked:
    if not prompt.strip():
        st.error("⚠️ Please enter a prompt.")
    else:
        used_seed = random.randint(0, 2147483647) if random_seed else int(seed_val)
        full_prompt = prompt.strip() + STYLE_SUFFIXES[style]
        mode_real = bool(hf_token.strip())

        if mode_real:
            # ── Real image generation via HuggingFace InferenceClient ──────────
            endpoint = hf_endpoint.strip() or None
            target = endpoint if endpoint else HF_MODEL
            st.info(f"🔄 Calling `{target}`… seed={used_seed} · {w}×{h} · steps={steps}")
            cols = st.columns(min(num_images, 2))

            with st.spinner("Generating images…"):
                try:
                    client = InferenceClient(
                        model=endpoint if endpoint else HF_MODEL,
                        token=hf_token.strip(),
                    )
                    for i in range(num_images):
                        try:
                            pil_img = client.text_to_image(
                                full_prompt,
                                negative_prompt=neg_prompt,
                                width=w,
                                height=h,
                                num_inference_steps=steps,
                                guidance_scale=guidance,
                                seed=used_seed + i,
                            )
                            buf = BytesIO()
                            pil_img.save(buf, format="PNG")
                            buf.seek(0)
                            col_idx = i % 2
                            with cols[col_idx]:
                                st.image(buf, caption=f"Image {i+1} · seed={used_seed+i}", use_container_width=True)
                        except HfHubHTTPError as e:
                            st.error(f"Image {i+1} — HuggingFace API error: {e}")
                        except Exception as e:
                            st.error(f"Image {i+1} error: {e}")
                except Exception as e:
                    st.error(f"Failed to connect to HuggingFace: {e}")
                    st.warning(
                        f"`{HF_MODEL}` is a 64B model and is **not available on the free inference API**. "
                        "You need a [HuggingFace Inference Endpoint](https://ui.endpoints.huggingface.co/) "
                        "or a Space URL. Paste the endpoint URL in the sidebar field above the token."
                    )

        else:
            # ── Preview mode via Claude API ────────────────────────────────────
            if not claude_api_key.strip():
                st.error("⚠️ Enter your Claude API key in the sidebar (or a HuggingFace token for real generation).")
            else:
                st.info(f"🔄 Claude preview mode · seed={used_seed} · {w}×{h} · steps={steps} · guidance={guidance}")

                system_prompt = f"""You are simulating NVIDIA's Cosmos3-Super-Text2Image model output descriptions.
When given a text prompt with parameters, respond ONLY with a JSON array (no markdown, no code blocks) of exactly {num_images} objects.
Each object must have:
- "alt": a vivid 2-sentence description of what the generated image looks like (as if describing a real photo/artwork)
- "tags": array of 4-5 short visual descriptor strings (colors, mood, composition, lighting)
- "seed": the numeric seed used

Be specific, evocative, and visual. Match the style suffix in the prompt."""

                user_msg = f"""Generate {num_images} image description(s) for this prompt:
"{full_prompt}"
Negative: "{neg_prompt}"
Seed: {used_seed}, Guidance: {guidance}, Steps: {steps}, Size: {w}x{h}"""

                with st.spinner("Generating descriptions via Claude…"):
                    try:
                        client = anthropic.Anthropic(api_key=claude_api_key)
                        message = client.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=1200,
                            system=system_prompt,
                            messages=[{"role": "user", "content": user_msg}],
                        )
                        raw = message.content[0].text
                        clean = raw.replace("```json", "").replace("```", "").strip()
                        descriptions = json.loads(clean)

                        st.markdown(f"""
                        <div style="color:rgba(255,255,255,0.35);font-size:12px;font-family:monospace;margin:8px 0">
                            ✓ Generated {len(descriptions)} description(s) · style: {style} · {w}×{h} · guidance: {guidance}
                        </div>
                        """, unsafe_allow_html=True)

                        cols = st.columns(min(num_images, 2))
                        for i, desc in enumerate(descriptions):
                            col_idx = i % 2
                            with cols[col_idx]:
                                tags_html = "".join(
                                    f'<span class="tag">{t}</span>'
                                    for t in (desc.get("tags") or [])
                                )
                                img_seed = desc.get("seed", used_seed + i)
                                st.markdown(f"""
                                <div class="result-card">
                                    <div style="height:12px;background:linear-gradient(135deg,#1a0533,#0d1f4a,#0a2a1f);"></div>
                                    <div class="result-meta">
                                        <div class="result-desc">{desc.get("alt","")}</div>
                                        <div>{tags_html}</div>
                                        <div class="result-footer">
                                            Image {i+1} · seed={img_seed} · {w}×{h}
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                        st.markdown("""
                        <div class="preview-note">
                            💡 <strong style="color:#7dd3fc">Preview Mode</strong>: The above are Claude-generated image descriptions.
                            For real images, add your HuggingFace token in the sidebar — images will be generated by
                            <code style="color:#a5b4fc;font-size:11px">nvidia/Cosmos3-Super-Text2Image</code>.
                        </div>
                        """, unsafe_allow_html=True)

                    except json.JSONDecodeError as e:
                        st.error(f"Failed to parse response: {e}")
                    except anthropic.AuthenticationError:
                        st.error("Invalid Claude API key. Please check your key in the sidebar.")
                    except Exception as e:
                        st.error(f"Generation failed: {e}")

else:
    # Empty state
    st.markdown("""
    <div class="empty-state">
        <div style="font-size:48px">🌌</div>
        <div style="font-size:15px">Enter a prompt and click Generate</div>
        <div style="font-size:12px;color:rgba(255,255,255,0.12)">
            Preview mode uses Claude API · Real images need a HuggingFace token
        </div>
    </div>
    """, unsafe_allow_html=True)
