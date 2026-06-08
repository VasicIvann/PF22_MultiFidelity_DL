# --- Fichier : src/degradation.py ---
"""
Source UNIQUE de la dégradation Basse Fidélité (BF) du projet multi-fidélité.

Historique du problème corrigé ici
----------------------------------
Avant ce module, la "dégradation BF" existait en 3 variantes incohérentes :
  - Train BF Animals-10  : downsample NEAREST -> taille d'origine, bruit, JPEG q60
  - Train BF Imagewoof   : downsample BILINEAR -> 224, bruit, JPEG q90
  - Test BF (les deux)   : 224 -> 64 -> 224 bilinéaire antialias, bruit, AUCUN JPEG
Le modèle était donc entraîné et évalué sur des distributions BF différentes
(decalage de domaine train/test parasite).

Choix retenu (centralisé ici)
-----------------------------
  * UNE seule fonction de dégradation, appliquée À LA VOLÉE au train comme au test.
    Les images BF sont stockées PROPRES sur disque ; la dégradation est appliquée
    au chargement. Train BF et Test BF partagent ainsi exactement le même pipeline.
  * Dégradation canonique = downsample bilinéaire antialias (224 -> `downscale` -> 224)
    + bruit gaussien (sigma) + compression JPEG (quality), dans cet ordre.
  * Déterministe par index (seed = base_seed + idx) : la dégradation d'une image
    donnée est reproductible et identique à chaque époque et entre les runs.
    -> évaluation stable ; train BF figé par image (pas d'effet d'augmentation).

Les paramètres (downscale, sigma, jpeg_quality) sont exposés pour permettre les
balayages d'intensité de dégradation (étape "robustesse vs dégradation").
"""

import io

import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

# --- Normalisation ImageNet (commune à tout le projet) ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Paramètres canoniques du "domaine BF" de référence ---
# (= niveau de dégradation utilisé dans les tableaux de résultats principaux)
DEFAULT_SIZE = 224           # résolution d'entrée du réseau
DEFAULT_DOWNSCALE = 64       # résolution intermédiaire (perte de détail)
DEFAULT_SIGMA = 0.15         # écart-type du bruit gaussien
DEFAULT_JPEG_QUALITY = 60    # qualité de compression JPEG (1-95), 100 = pas de JPEG


def jpeg_compress_tensor(img, quality):
    """Round-trip JPEG d'un tenseur image float [0,1] (C,H,W) -> tenseur float [0,1].

    Reproduit fidèlement les artefacts de compression JPEG (et la quantification
    8 bits) via un encodage/décodage PIL en mémoire.
    """
    pil = TF.to_pil_image(img.clamp(0.0, 1.0))
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    out = Image.open(buffer).convert("RGB")
    return TF.to_tensor(out)


def degrade_tensor(img, downscale=DEFAULT_DOWNSCALE, sigma=DEFAULT_SIGMA,
                   jpeg_quality=DEFAULT_JPEG_QUALITY, generator=None):
    """Applique la dégradation BF canonique à un tenseur image float [0,1] (C,H,W).

    Ordre : sous-échantillonnage bilinéaire antialias -> bruit gaussien -> JPEG.

    Args:
        img: tenseur float [0,1] de forme (C,H,W).
        downscale: résolution intermédiaire du sous-échantillonnage. None/0 = pas
            de perte de résolution.
        sigma: écart-type du bruit gaussien. 0 = pas de bruit.
        jpeg_quality: qualité JPEG (1-95). >=100 ou None = pas de compression JPEG.
        generator: torch.Generator optionnel rendant le bruit reproductible.

    Returns:
        tenseur float [0,1] de forme (C,H,W).
    """
    c, h, w = img.shape

    # 1. Sous-échantillonnage puis sur-échantillonnage (flou / perte de détail)
    if downscale and downscale > 0:
        img = TF.resize(img, [int(downscale), int(downscale)], antialias=True)
        img = TF.resize(img, [h, w], antialias=True)

    # 2. Bruit gaussien (simule le bruit ISO d'un capteur bas de gamme)
    if sigma and sigma > 0:
        noise = torch.randn(img.shape, generator=generator) * float(sigma)
        img = torch.clamp(img + noise, 0.0, 1.0)

    # 3. Compression JPEG (artefacts de bloc + quantification 8 bits)
    if jpeg_quality is not None and jpeg_quality < 100:
        img = jpeg_compress_tensor(img, jpeg_quality)

    return img


def hf_transform(size=DEFAULT_SIZE):
    """Transform Haute Fidélité : Resize -> ToTensor -> Normalize (aucune dégradation)."""
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def clean_tensor_transform(size=DEFAULT_SIZE):
    """Transform de base pour DegradedDataset : Resize -> ToTensor (PAS de Normalize).

    Renvoie un tenseur float [0,1] que DegradedDataset dégradera puis normalisera.
    """
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
    ])


def normalize_transform():
    """Normalisation ImageNet seule (post_transform par défaut de DegradedDataset)."""
    return T.Normalize(IMAGENET_MEAN, IMAGENET_STD)


class DegradedDataset(Dataset):
    """Enveloppe un dataset propre et applique la dégradation BF à la volée.

    Le dataset de base doit renvoyer (tenseur float [0,1] (C,H,W), label),
    c.-à-d. un ImageFolder avec `transform=clean_tensor_transform()`.

    Pipeline par échantillon :
        1. dégradation BF canonique (downscale, sigma, jpeg_quality),
           déterministe par index si `seeded=True` (seed = base_seed + idx) ;
        2. `post_transform` appliqué au tenseur dégradé [0,1].
           Par défaut : normalisation ImageNet.
           Pour le "Student" de Noisy Student, on passe un post_transform qui
           ajoute l'augmentation agressive APRÈS la dégradation, p. ex. :
               Compose([ToPILImage(), RandomHorizontalFlip(), RandAugment(...),
                        ColorJitter(...), ToTensor(), RandomErasing(...),
                        Normalize(...)])
           L'augmentation reste aléatoire à chaque époque (le seed ne fige que
           la dégradation BF, pas le post_transform).

    Délègue les attributs inconnus au dataset de base (`classes`, `targets`,
    `samples`, ...), ce qui permet de l'utiliser de façon transparente avec les
    samplers/Subset existants.
    """

    def __init__(self, base, post_transform="normalize",
                 downscale=DEFAULT_DOWNSCALE, sigma=DEFAULT_SIGMA,
                 jpeg_quality=DEFAULT_JPEG_QUALITY,
                 seeded=True, base_seed=0):
        self.base = base
        if post_transform == "normalize":
            post_transform = normalize_transform()
        self.post_transform = post_transform
        self.downscale = downscale
        self.sigma = sigma
        self.jpeg_quality = jpeg_quality
        self.seeded = seeded
        self.base_seed = int(base_seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]  # img: tenseur float [0,1] (C,H,W)

        generator = None
        if self.seeded:
            generator = torch.Generator()
            generator.manual_seed(self.base_seed + int(idx))

        img = degrade_tensor(
            img,
            downscale=self.downscale,
            sigma=self.sigma,
            jpeg_quality=self.jpeg_quality,
            generator=generator,
        )

        if self.post_transform is not None:
            img = self.post_transform(img)

        return img, label

    def __getattr__(self, name):
        # Délégation des attributs (classes, targets, samples, imgs, ...) au dataset
        # de base. __getattr__ n'est appelé que si l'attribut n'existe pas déjà.
        base = self.__dict__.get("base")
        if base is not None and hasattr(base, name):
            return getattr(base, name)
        raise AttributeError(name)
