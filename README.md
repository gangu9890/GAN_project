# GAN Dashboard – Setup & Usage Guide

## Project Structure

```
gan_app/
├── app.py                  ← Streamlit frontend (run this)
├── backend.py              ← Inference helpers & model registry
├── model_definitions.py    ← All 3 GAN architectures (DCGAN, Vanilla, CGAN)
├── requirements.txt
├── models/                 ← Put your pre-trained .pth files here
│   ├── dcgan_generator.pth
│   ├── dcgan_discriminator.pth
│   ├── vanilla_generator.pth
│   ├── vanilla_discriminator.pth
│   ├── cgan_generator.pth
│   └── cgan_discriminator.pth
└── uploaded_models/        ← Auto-created; stores .pth files uploaded via UI
```

---

## Step 1 – Install Dependencies

```bash
pip install -r requirements.txt
```

> For GPU support install the CUDA-enabled PyTorch version:
> https://pytorch.org/get-started/locally/

---

## Step 2 – Save your trained weights

At the end of your training notebooks add:

```python
# DCGAN
torch.save(generator.state_dict(),     "models/dcgan_generator.pth")
torch.save(discriminator.state_dict(), "models/dcgan_discriminator.pth")

# Vanilla GAN
torch.save(generator_vanilla.state_dict(),     "models/vanilla_generator.pth")
torch.save(discriminator_vanilla.state_dict(), "models/vanilla_discriminator.pth")

# CGAN
torch.save(generator.state_dict(),     "models/cgan_generator.pth")
torch.save(discriminator.state_dict(), "models/cgan_discriminator.pth")
```

---

## Step 3 – Run the App

```bash
cd gan_app
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Features

### 🖼️ Generate Images tab
- Select any loaded model from the dropdown
- Choose number of images (1–16)
- For **CGAN**: pick a class name → generates images of that class
- Download the generated image grid as PNG

### 🔍 Real or Fake? tab
- Upload any image (jpg/png)
- Select which discriminator to use
- For **CGAN**: also select the class label the image belongs to
- Get a score in [0, 1] → **≥ 0.5 = Real**, **< 0.5 = Fake**

### 📦 Add Custom Model tab
- Upload new `.pth` generator and/or discriminator weights **directly from the browser**
- Give it a custom name — it immediately appears in all other tabs
- No server restart needed

---

## Loading Models in the Sidebar

In the left sidebar, expand the section for the model you want and paste the
file paths to the `.pth` files, then click Load.  
You only need to load models once per session.

---

## Customising CGAN Class Names

If your CGAN was trained on a custom dataset (not CIFAR-10), update the
`CGAN_CLASSES` list in `model_definitions.py` to match your training labels:

```python
CGAN_CLASSES = ["cat", "dog", "bird", ...]   # your classes here
```

Or supply them dynamically via the sidebar / custom model uploader in the UI.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `RuntimeError: size mismatch` | The architecture in `model_definitions.py` doesn't match the weights. Verify hyperparams (latent_size, num_classes etc.) |
| Blank/noisy images | Weights may not be trained long enough, or normalisation stats differ |
| CUDA out of memory | Reduce number of generated images |
| App freezes on CPU | Normal for large batches; reduce n_images |
