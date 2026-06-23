# --- Fichier : make_dataset_samples.py ---
"""
Génère, pour chaque dataset, 1 image Haute Fidélité (HF) et sa version
Basse Fidélité (BF) — la MÊME image, dégradée par le pipeline canonique du
projet (downscale 64 px + bruit sigma=0.15 + JPEG q60).

À lancer SUR LE SERVEUR (env pf22, datasets présents), depuis la racine du repo :
    conda activate pf22
    python make_dataset_samples.py

Produit 6 PNG dans ./dataset_samples/ :
    sample_Animals-10_HF.png  sample_Animals-10_BF.png
    sample_Imagewoof_HF.png   sample_Imagewoof_BF.png
    sample_Intel_HF.png       sample_Intel_BF.png
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from PIL import Image
from torchvision.utils import save_image

import env_config
from degradation import (
    clean_tensor_transform,
    degrade_tensor,
    DEFAULT_DOWNSCALE,
    DEFAULT_SIGMA,
    DEFAULT_JPEG_QUALITY,
)

OUT = os.path.join(ROOT, "dataset_samples")
os.makedirs(OUT, exist_ok=True)

DATASETS = ["Animals-10", "Imagewoof", "Intel"]
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
clean = clean_tensor_transform(224)  # Resize 224 + ToTensor (float [0,1], pas de Normalize)


def first_image(root):
    """Premier fichier image trouvé sous `root` (récursif, ignore les ._*)."""
    for dirpath, _dirs, files in os.walk(root):
        imgs = sorted(f for f in files if f.lower().endswith(EXTS) and not f.startswith("._"))
        if imgs:
            return os.path.join(dirpath, imgs[0])
    return None


for ds in DATASETS:
    dd = env_config.data_dir(ds)
    # image propre HF : d'abord le dossier test (100 % HF), sinon train/HF
    img_path = None
    for sub in (os.path.join(dd, "test"), os.path.join(dd, "train", "HF"), dd):
        if os.path.isdir(sub):
            img_path = first_image(sub)
            if img_path:
                break
    if not img_path:
        print(f"[!] aucune image trouvée pour {ds} (cherché dans {dd})")
        continue

    pil = Image.open(img_path).convert("RGB")
    hf = clean(pil)  # [0,1] CHW 224x224
    bf = degrade_tensor(
        hf.clone(),
        downscale=DEFAULT_DOWNSCALE,   # 64
        sigma=DEFAULT_SIGMA,           # 0.15
        jpeg_quality=DEFAULT_JPEG_QUALITY,  # 60
    )

    save_image(hf, os.path.join(OUT, f"sample_{ds}_HF.png"))
    save_image(bf, os.path.join(OUT, f"sample_{ds}_BF.png"))
    print(f"[ok] {ds:12s} <- {os.path.basename(img_path)}  (HF + BF)")

print("\nImages générées dans :", OUT)
