import os, io
import streamlit as st
from PIL import Image

st.set_page_config(page_title="GAN Dashboard", page_icon="🎨", layout="wide")

from backend import ModelRegistry, generate_images, discriminate_image
CGAN_CLASSES = ["panda", "bunny"]
if "registry" not in st.session_state:
    st.session_state.registry = ModelRegistry()

registry: ModelRegistry = st.session_state.registry

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎛️ GAN Dashboard")
    st.markdown("---")

    with st.expander("Load DCGAN weights", expanded=False):
        dc_g = st.text_input("Generator .pth path",     key="dc_g", value="models/dcgan_generator.pth")
        dc_d = st.text_input("Discriminator .pth path", key="dc_d", value="models/dcgan_discriminator.pth")
        if st.button("Load DCGAN", key="btn_dc"):
            registry.load_dcgan(dc_g or None, dc_d or None)
            st.success("DCGAN loaded ✓")

    with st.expander("Load Vanilla GAN weights", expanded=False):
        vg_g = st.text_input("Generator .pth path",     key="vg_g", value="models/vanilla_generator.pth")
        vg_d = st.text_input("Discriminator .pth path", key="vg_d", value="models/vanilla_discriminator.pth")
        if st.button("Load Vanilla GAN", key="btn_vg"):
            registry.load_vanilla(vg_g or None, vg_d or None)
            st.success("Vanilla GAN loaded ✓")

    with st.expander("Load CGAN weights", expanded=False):
        cg_g  = st.text_input("Generator .pth path",     key="cg_g", value="models/cgan_generator.pth")
        cg_d  = st.text_input("Discriminator .pth path", key="cg_d", value="models/cgan_discriminator.pth")
        cg_nc = st.number_input("Num classes", min_value=2, max_value=100, value=2, key="cg_nc")
        cg_cls = st.text_area("Class names (one per line)", value="\n".join(CGAN_CLASSES), key="cg_cls")
        if st.button("Load CGAN", key="btn_cg"):
            names = [c.strip() for c in cg_cls.strip().split("\n") if c.strip()]
            registry.load_cgan(cg_g or None, cg_d or None, int(cg_nc), names)
            st.success("CGAN loaded ✓")

    st.markdown("---")
    loaded = registry.list_models()
    if loaded:
        st.markdown("**Loaded models:** " + ", ".join(f"`{m}`" for m in loaded))
    else:
        st.info("No models loaded yet.")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_gen, tab_disc, tab_custom = st.tabs(
    ["🖼️  Generate Images", "🔍  Real or Fake?", "📦  Add Custom Model"]
)

# ── Tab 1: Generate ───────────────────────────────────────────────────────────
with tab_gen:
    st.header("Image Generation")
    loaded = registry.list_models()
    if not loaded:
        st.warning("Load at least one model from the sidebar first.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            model_name = st.selectbox("Select model", loaded, key="gen_model")
            entry = registry.get(model_name)
            meta  = entry["meta"] if entry else {}
            n_images = st.slider("Number of images", 1, 16, 8, key="gen_n")
            class_idx = None
            if meta.get("type") in ("cgan",) or (meta.get("type") == "dynamic" and meta.get("is_conditional")):
                class_names = meta.get("class_names", CGAN_CLASSES)
                chosen = st.selectbox("Class to generate", class_names, key="gen_cls")
                class_idx = class_names.index(chosen)
            generate_btn = st.button("✨ Generate", key="gen_btn", use_container_width=True)
        with col2:
            if generate_btn:
                if entry is None:
                    st.error("Model not found.")
                else:
                    with st.spinner("Generating…"):
                        img = generate_images(entry, n=n_images, class_idx=class_idx)
                    st.image(img, caption=f"{model_name} – {n_images} samples", use_container_width=True)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    st.download_button("⬇️ Download grid", data=buf.getvalue(),
                                       file_name="generated.png", mime="image/png")
            else:
                st.info("Configure options on the left and hit **Generate**.")

# ── Tab 2: Discriminate ───────────────────────────────────────────────────────
with tab_disc:
    st.header("Is it Real or Fake?")
    st.markdown("Upload any image and the chosen discriminator will score it. Score closer to **1 = Real**, closer to **0 = Fake**.")
    loaded = registry.list_models()
    if not loaded:
        st.warning("Load at least one model from the sidebar first.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            model_name = st.selectbox("Discriminator model", loaded, key="disc_model")
            entry = registry.get(model_name)
            meta  = entry["meta"] if entry else {}
            class_idx = None
            if meta.get("type") == "cgan" or (meta.get("type") == "dynamic" and meta.get("is_conditional")):
                class_names = meta.get("class_names", CGAN_CLASSES)
                chosen = st.selectbox("Label for the image", class_names, key="disc_cls")
                class_idx = class_names.index(chosen)
            uploaded = st.file_uploader("Upload image (jpg / png)", type=["jpg","jpeg","png"], key="disc_upload")
            run_btn = st.button("🔍 Classify", key="disc_btn", use_container_width=True, disabled=uploaded is None)
        with col2:
            if uploaded:
                pil_img = Image.open(uploaded).convert("RGB")
                st.image(pil_img, caption="Uploaded image", width=300)
                if run_btn:
                    if entry is None:
                        st.error("Model not found.")
                    else:
                        with st.spinner("Running discriminator…"):
                            score = discriminate_image(entry, pil_img, class_idx)
                        st.markdown("### Result")
                        verdict = "🟢 **REAL**" if score >= 0.5 else "🔴 **FAKE**"
                        st.markdown(f"**Verdict:** {verdict}")
                        st.metric("Discriminator score", f"{score:.4f}",
                                  delta=f"{'above' if score>=0.5 else 'below'} 0.5 threshold")
                        st.progress(float(score))
            else:
                st.info("Upload an image and click **Classify**.")

# ── Tab 3: Custom Model ───────────────────────────────────────────────────────
with tab_custom:
    st.header("Add a New Model")
    mode = st.radio("Architecture source",
                    ["Built-in (DCGAN / Vanilla / CGAN)", "Custom (upload your own .py)"],
                    horizontal=True, key="cust_mode")
    st.markdown("---")

    # ── Mode A: Built-in ──────────────────────────────────────────────────
    if mode == "Built-in (DCGAN / Vanilla / CGAN)":
        st.markdown("Upload `.pth` weight files that match one of the three built-in architectures.")
        c1, c2 = st.columns(2)
        with c1:
            custom_name = st.text_input("Model name (unique)", key="cust_name", placeholder="my_new_dcgan")
            arch = st.selectbox("Architecture", ["dcgan", "vanilla", "cgan"], key="cust_arch")
            num_cls = 2
            cls_names = CGAN_CLASSES[:]
            if arch == "cgan":
                num_cls = st.number_input("Num classes", 2, 100, 2, key="cust_nc")
                cls_txt = st.text_area("Class names (one per line)", value="\n".join(CGAN_CLASSES), key="cust_cls")
                cls_names = [c.strip() for c in cls_txt.strip().split("\n") if c.strip()]
        with c2:
            g_file = st.file_uploader("Generator weights (.pth)",     type=["pth"], key="cust_g_file")
            d_file = st.file_uploader("Discriminator weights (.pth)", type=["pth"], key="cust_d_file")
        upload_btn = st.button("📦 Register Model", key="cust_btn",
                               use_container_width=True, disabled=not custom_name)
        if upload_btn:
            save_dir = "uploaded_models"
            os.makedirs(save_dir, exist_ok=True)
            g_path, d_path = None, None
            if g_file:
                g_path = os.path.join(save_dir, f"{custom_name}_G.pth")
                with open(g_path, "wb") as f: f.write(g_file.read())
            if d_file:
                d_path = os.path.join(save_dir, f"{custom_name}_D.pth")
                with open(d_path, "wb") as f: f.write(d_file.read())
            try:
                registry.load_custom(name=custom_name, g_path=g_path, d_path=d_path,
                                     model_type=arch, num_classes=int(num_cls), class_names=cls_names)
                st.success(f"✅ **{custom_name}** registered! Switch to other tabs to use it.")
            except Exception as e:
                st.error(f"Error loading model: {e}")

    # ── Mode B: Custom .py ────────────────────────────────────────────────
    else:
        st.markdown("Upload your own architecture `.py` file alongside the weights. "
                    "Your file **must** define two classes: `Generator` and `Discriminator`, "
                    "both subclassing `torch.nn.Module`.")

        with st.expander("📄 Show architecture template", expanded=False):
            st.code('''
import torch.nn as nn

# z shape fed by app: (batch, 128, 1, 1)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # ... your layers ...
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
    def forward(self, z):
        return self.net(z)

# img shape fed by app: (batch, 3, 64, 64)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # ... your layers ...
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

# For conditional GANs, use:
# class Generator(nn.Module):
#     def __init__(self, num_classes): ...
#     def forward(self, z, labels): ...
''', language="python")

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            dyn_name = st.text_input("Model name (unique)", key="dyn_name", placeholder="my_stylegan")
            is_cond = st.checkbox("Conditional model (forward takes labels)", key="dyn_cond")
            dyn_num_cls = 2
            dyn_cls_names = []
            if is_cond:
                dyn_num_cls = st.number_input("Num classes", 2, 100, 2, key="dyn_nc")
                cls_txt = st.text_area("Class names (one per line)", placeholder="cat\ndog", key="dyn_cls")
                dyn_cls_names = [c.strip() for c in cls_txt.strip().split("\n") if c.strip()]
        with c2:
            arch_file  = st.file_uploader("Architecture file (.py)",       type=["py"],  key="dyn_arch_file")
            dyn_g_file = st.file_uploader("Generator weights (.pth)",       type=["pth"], key="dyn_g_file")
            dyn_d_file = st.file_uploader("Discriminator weights (.pth)",   type=["pth"], key="dyn_d_file")

        if not dyn_name:   st.caption("⚠️ Enter a model name to continue.")
        if not arch_file:  st.caption("⚠️ Upload an architecture `.py` file to continue.")

        dyn_btn = st.button("📦 Register Custom Model", key="dyn_btn",
                            use_container_width=True, disabled=not (dyn_name and arch_file))
        if dyn_btn:
            save_dir = "uploaded_models"
            os.makedirs(save_dir, exist_ok=True)
            arch_path = os.path.join(save_dir, f"{dyn_name}_arch.py")
            with open(arch_path, "wb") as f: f.write(arch_file.read())
            g_path, d_path = None, None
            if dyn_g_file:
                g_path = os.path.join(save_dir, f"{dyn_name}_G.pth")
                with open(g_path, "wb") as f: f.write(dyn_g_file.read())
            if dyn_d_file:
                d_path = os.path.join(save_dir, f"{dyn_name}_D.pth")
                with open(d_path, "wb") as f: f.write(dyn_d_file.read())
            try:
                registry.load_dynamic(name=dyn_name, arch_path=arch_path, g_path=g_path,
                                      d_path=d_path, is_conditional=is_cond,
                                      num_classes=int(dyn_num_cls), class_names=dyn_cls_names or None)
                st.success(f"✅ **{dyn_name}** registered from `{arch_file.name}`! Switch to other tabs to use it.")
            except AttributeError as e:
                st.error(f"❌ Architecture file error: {e}")
            except ImportError as e:
                st.error(f"❌ Could not import architecture: {e}")
            except RuntimeError as e:
                st.error(f"❌ Weight mismatch — architecture doesn't match the .pth file.\n\n`{e}`")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

    # ── registered custom models list ─────────────────────────────────────
    all_models = registry.list_models()
    custom_models = [m for m in all_models if m not in ("dcgan", "vanilla", "cgan")]
    if custom_models:
        st.markdown("---")
        st.subheader("Currently registered custom models")
        for m in custom_models:
            e = registry.get(m)
            meta = e["meta"]
            arch_label = f"`{meta['arch_file']}`" if meta["type"] == "dynamic" else f"`{meta['type']}`"
            cond_label = " · conditional" if meta.get("is_conditional") else ""
            st.markdown(f"**{m}** – arch: {arch_label}{cond_label}, classes: {meta.get('num_classes', '–')}")