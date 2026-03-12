import sys
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import io
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import make_grid

from registry.model_definitions import (
    DCGAN_Discriminator, DCGAN_Generator,
    VanillaGAN_Discriminator, VanillaGAN_Generator,
    CGAN_Discriminator, CGAN_Generator
)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_SIZE  = 128
IMAGE_SIZE   = 64
CGAN_CLASSES = ["panda", "bunny"]
STATS_MEAN   = (0.5, 0.5, 0.5)
STATS_STD    = (0.5, 0.5, 0.5)

_preprocess = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(STATS_MEAN, STATS_STD),
])


def denorm(tensor):
    mean = torch.tensor(STATS_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std  = torch.tensor(STATS_STD,  device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


def tensor_to_pil(tensor):
    tensor = denorm(tensor.unsqueeze(0)).squeeze(0)
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def grid_to_pil(tensors, nrow=4):
    grid = make_grid(denorm(tensors), nrow=nrow, padding=2)
    arr  = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


class ModelRegistry:
    def __init__(self):
        self._registry: dict = {}

    # ── built-in loaders ──────────────────────────────────────────────────

    def load_dcgan(self, g_path, d_path):
        G = DCGAN_Generator().to(DEVICE).eval()
        D = DCGAN_Discriminator().to(DEVICE).eval()
        if g_path and os.path.exists(g_path):
            G.load_state_dict(torch.load(g_path, map_location=DEVICE))
        if d_path and os.path.exists(d_path):
            D.load_state_dict(torch.load(d_path, map_location=DEVICE))
        self._registry["dcgan"] = {
            "G": G, "D": D,
            "meta": {"type": "dcgan", "latent_shape": (LATENT_SIZE, 1, 1)},
        }

    def load_vanilla(self, g_path, d_path):
        G = VanillaGAN_Generator().to(DEVICE).eval()
        D = VanillaGAN_Discriminator().to(DEVICE).eval()
        if g_path and os.path.exists(g_path):
            G.load_state_dict(torch.load(g_path, map_location=DEVICE))
        if d_path and os.path.exists(d_path):
            D.load_state_dict(torch.load(d_path, map_location=DEVICE))
        self._registry["vanilla"] = {
            "G": G, "D": D,
            "meta": {"type": "vanilla", "latent_shape": (LATENT_SIZE, 1, 1)},
        }

    def load_cgan(self, g_path, d_path, num_classes, class_names):
        G = CGAN_Generator(num_classes).to(DEVICE).eval()
        D = CGAN_Discriminator(num_classes).to(DEVICE).eval()
        if g_path and os.path.exists(g_path):
            G.load_state_dict(torch.load(g_path, map_location=DEVICE))
        if d_path and os.path.exists(d_path):
            D.load_state_dict(torch.load(d_path, map_location=DEVICE))
        self._registry["cgan"] = {
            "G": G, "D": D,
            "meta": {
                "type": "cgan",
                "num_classes": num_classes,
                "class_names": class_names,
                "latent_shape": (LATENT_SIZE, 1, 1),
            },
        }

    def load_custom(self, name, g_path, d_path, model_type,
                    num_classes=2, class_names=None):
        if model_type == "dcgan":
            G = DCGAN_Generator().to(DEVICE).eval()
            D = DCGAN_Discriminator().to(DEVICE).eval()
        elif model_type == "vanilla":
            G = VanillaGAN_Generator().to(DEVICE).eval()
            D = VanillaGAN_Discriminator().to(DEVICE).eval()
        elif model_type == "cgan":
            G = CGAN_Generator(num_classes).to(DEVICE).eval()
            D = CGAN_Discriminator(num_classes).to(DEVICE).eval()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        if g_path and os.path.exists(g_path):
            G.load_state_dict(torch.load(g_path, map_location=DEVICE))
        if d_path and os.path.exists(d_path):
            D.load_state_dict(torch.load(d_path, map_location=DEVICE))
        self._registry[name] = {
            "G": G, "D": D,
            "meta": {
                "type": model_type,
                "num_classes": num_classes,
                "class_names": class_names or CGAN_CLASSES[:num_classes],
                "latent_shape": (LATENT_SIZE, 1, 1),
            },
        }
        return True

    # ── dynamic loader ────────────────────────────────────────────────────

    def load_dynamic(self, name, arch_path, g_path, d_path,
                     is_conditional=False, num_classes=2, class_names=None):
        import importlib.util, uuid
        mod_name = f"user_arch_{uuid.uuid4().hex[:8]}"
        spec = importlib.util.spec_from_file_location(mod_name, arch_path)
        if spec is None:
            raise ImportError(f"Cannot load architecture file: {arch_path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise ImportError(f"Error executing architecture file: {e}")
        if not hasattr(module, "Generator"):
            raise AttributeError("Architecture file must define a class named 'Generator'")
        if not hasattr(module, "Discriminator"):
            raise AttributeError("Architecture file must define a class named 'Discriminator'")
        try:
            G = module.Generator(num_classes).to(DEVICE).eval() if is_conditional \
                else module.Generator().to(DEVICE).eval()
        except TypeError:
            G = module.Generator().to(DEVICE).eval()
        try:
            D = module.Discriminator(num_classes).to(DEVICE).eval() if is_conditional \
                else module.Discriminator().to(DEVICE).eval()
        except TypeError:
            D = module.Discriminator().to(DEVICE).eval()
        if g_path and os.path.exists(g_path):
            G.load_state_dict(torch.load(g_path, map_location=DEVICE))
        if d_path and os.path.exists(d_path):
            D.load_state_dict(torch.load(d_path, map_location=DEVICE))
        self._registry[name] = {
            "G": G, "D": D,
            "meta": {
                "type": "dynamic",
                "is_conditional": is_conditional,
                "num_classes": num_classes,
                "class_names": class_names or [str(i) for i in range(num_classes)],
                "latent_shape": (LATENT_SIZE, 1, 1),
                "arch_file": os.path.basename(arch_path),
            },
        }
        return True

    # ── accessors ─────────────────────────────────────────────────────────

    def list_models(self):
        return list(self._registry.keys())

    def get(self, name):
        return self._registry.get(name)


# ── Inference helpers ─────────────────────────────────────────────────────────

def generate_images(entry, n=8, class_idx=None):
    G    = entry["G"]
    meta = entry["meta"]
    ls   = meta["latent_shape"]
    with torch.no_grad():
        z = torch.randn(n, *ls, device=DEVICE)
        if meta["type"] == "cgan" or (meta["type"] == "dynamic" and meta.get("is_conditional")):
            if class_idx is None:
                class_idx = 0
            labels = torch.full((n,), class_idx, dtype=torch.long, device=DEVICE)
            imgs = G(z, labels)
        elif meta["type"] == "vanilla":
            imgs = G(z)
        else:
            imgs = G(z)
    return grid_to_pil(imgs)


def discriminate_image(entry, pil_img, class_idx=None):
    D    = entry["D"]
    meta = entry["meta"]
    tensor = _preprocess(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        if meta["type"] == "cgan" or (meta["type"] == "dynamic" and meta.get("is_conditional")):
            if class_idx is None:
                class_idx = 0
            labels = torch.tensor([class_idx], dtype=torch.long, device=DEVICE)
            out = D(tensor, labels)
        else:
            out = D(tensor)
    return float(out.squeeze())