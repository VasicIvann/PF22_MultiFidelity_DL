# --- Fichier : src/generate_multifidelity_datasets.py ---
# Script de génération de l'environnement Multi-Fidélité pour Animals-10
# Combo : Coût d'Acquisition (A) + Dégradation Visuelle (1)

import os
import random
import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from tqdm import tqdm # Pour la barre de progression

# --- Configuration des chemins et Ratios ---
# Ces chemins sont relatifs au serveur Colab une fois le Drive monté
BASE_DIR_DRIVE = Path("/content/drive/MyDrive/UTBM_PF22/datasets/Animals-10")
SOURCE_DIR = BASE_DIR_DRIVE / "raw_data" # Dossier contenant les 10 dossiers originaux

# Nouveau dossier où nous allons ranger les données prêtes
OUTPUT_DIR = BASE_DIR_DRIVE / "processed_multifidelity"

# --- Définition des Ratios de la population totale (~28 000 images) ---
# 1. Jeu de TEST (HF) : 10% (Reste intouché pour validation équitable)
RATIO_TEST = 0.10 

# 2. Jeu d'ENTRAÎNEMENT TOTAL (90% restants) :
#    Dans ce jeu, nous appliquons le ratio Multi-Fidélité :
#    - 10% des images d'entraînement seront HF (CHÈRES)
#    - 90% des images d'entraînement seront BF (PAS CHÈRES & DÉGRADÉES)
RATIO_TRAIN_HF_CHERE = 0.10 

# --- Fonction de Dégradation de la qualité (Stratégie 1) ---
def degrade_image_visual(img_pil):
    """
    Applique une dégradation visuelle :
    1. Baisse massive de résolution -> Upscale (crée du flou/pixelisation)
    2. Ajout de Bruit Gaussien (simule un capteur de mauvaise qualité)
    """
    original_size = img_pil.size
    
    # 1. Pipeline de pixelisation/flou
    degradation_transforms = T.Compose([
        T.Resize((64, 64), interpolation=T.InterpolationMode.NEAREST), # Très basse résolution
        T.Resize(original_size, interpolation=T.InterpolationMode.BILINEAR) # Remise à taille normale
    ])
    
    img_degraded_pil = degradation_transforms(img_pil)
    
    # 2. Pipeline d'ajout de bruit
    # On passe en tenseur pour le calcul, puis on ajoute du bruit, puis on repasse en PIL
    img_tensor = T.ToTensor()(img_degraded_pil) # Conversion 0-1
    
    # Intensité du bruit (0.15 est déjà assez visible sur des photos)
    noise = torch.randn_like(img_tensor) * 0.15 
    img_noisy_tensor = img_tensor + noise
    
    # Clamp pour rester entre 0 et 1, puis conversion uint8 et PIL
    img_noisy_tensor = torch.clamp(img_noisy_tensor, 0, 1)
    img_noisy_pil = T.ToPILImage()(img_noisy_tensor)
    
    return img_noisy_pil

# --- Script principal de génération ---
def main():
    print("--- ⚙️ GÉNÉRATION DE L'ENVIRONNEMENT MULTI-FIDÉLITÉ UTBM (Combo A1) ---")
    
    if not SOURCE_DIR.exists():
        print(f"❌ ERREUR : Le dossier source n'existe pas : {SOURCE_DIR}")
        print("Vérifie que tu as bien téléchargé le dataset dans 'raw_data' sur ton Drive.")
        return

    # Création de l'arborescence cible
    # train/HF, train/BF, test/
    for split in ['train/HF', 'train/BF', 'test']:
        os.makedirs(OUTPUT_DIR / split, exist_ok=True)
        # Création des sous-dossiers de classe (chien, chat...) dans chaque split
        classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(SOURCE_DIR / d)]
        for cls in classes:
            os.makedirs(OUTPUT_DIR / split / cls, exist_ok=True)

    # Parcours des classes
    classes = sorted([d for d in os.listdir(SOURCE_DIR) if os.path.isdir(SOURCE_DIR / d)])
    total_processed = 0
    errors = 0

    for cls in classes:
        print(f"\nTraitement de la classe : {cls}")
        images = sorted(os.listdir(SOURCE_DIR / cls))
        random.seed(42) # Pour que le découpage soit reproductible
        random.shuffle(images)
        
        num_images = len(images)
        
        # Calcul des indices de découpage
        idx_test = int(num_images * RATIO_TEST)
        idx_train_hf_limit = int((num_images - idx_test) * RATIO_TRAIN_HF_CHERE) + idx_test
        
        # Séparation des images
        images_test = images[:idx_test]
        images_train_hf = images[idx_test:idx_train_hf_limit]
        images_train_bf = images[idx_train_hf_limit:]
        
        # --- Fonction interne pour traiter un lot (gain de place) ---
        def process_lot(lot_images, split_path, degrade=False):
            nonlocal total_processed, errors
            print(f"   Generating {split_path} ({len(lot_images)} images)...")
            for img_name in tqdm(lot_images):
                src_path = SOURCE_DIR / cls / img_name
                dst_path = split_path / cls / img_name
                
                # Éviter de recalculer si le fichier existe déjà
                if dst_path.exists():
                    continue
                
                try:
                    # Lecture de l'image (PIL est plus sûr que read_image pour la compatibilité)
                    with Image.open(src_path) as img:
                        # Si l'image est en CMYK ou RGBA, conversion en RGB
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                            
                        if degrade:
                            # Application du script de dégradation visuelle
                            img_processed = degrade_image_visual(img)
                        else:
                            # Garder la Haute Fidélité
                            img_processed = img
                            
                        # Sauvegarde (on utilise quality=95 pour HF pour minimiser les artefacts)
                        # et quality=60 pour BF (ajoute de la compression JPEG en plus du bruit)
                        save_quality = 60 if degrade else 95
                        img_processed.save(dst_path, "JPEG", quality=save_quality)
                        
                    total_processed += 1
                except Exception as e:
                    errors += 1
                    # print(f"❌ Erreur sur {src_path} : {e}") # Décommenter pour debug
                    pass

        # Exécution du traitement pour les 3 jeux
        process_lot(images_test, OUTPUT_DIR / 'test', degrade=False)
        process_lot(images_train_hf, OUTPUT_DIR / 'train' / 'HF', degrade=False)
        process_lot(images_train_bf, OUTPUT_DIR / 'train' / 'BF', degrade=True)

    print("\n--- ✅ GÉNÉRATION TERMINÉE AVEC SUCCÈS ---")
    print(f"📁 Données prêtes dans : {OUTPUT_DIR}")
    print(f"📊 Images traitées : {total_processed}")
    print(f"⚠️ Erreurs de lecture (corrompues ou formats invalides) : {errors}")

if __name__ == "__main__":
    main()