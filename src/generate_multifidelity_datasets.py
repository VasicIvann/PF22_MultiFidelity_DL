# --- Fichier : src/generate_multifidelity_datasets.py ---
# Script de génération de l'environnement Multi-Fidélité pour Animals-10.
#
# IMPORTANT (changement de pipeline) :
#   Les images BF sont désormais stockées PROPRES (non dégradées) sur disque.
#   La dégradation BF (downsample bilinéaire antialias + bruit gaussien + JPEG)
#   est appliquée À LA VOLÉE au chargement, au train comme au test, via le module
#   partagé src/degradation.py. Cela garantit que train BF et test BF proviennent
#   exactement de la même distribution (plus de décalage de domaine parasite) et
#   permet de faire varier l'intensité de dégradation sans régénérer le dataset.
#
#   Le découpage (test / train HF / train BF) reste STRICTEMENT identique à
#   l'ancien pipeline (même seed, mêmes ratios) : seule la nature des images BF
#   stockées change (propres au lieu de dégradées).

import os
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm  # Pour la barre de progression

# --- Configuration des chemins et Ratios (valeurs par défaut) ---
# Ces chemins sont relatifs au serveur Colab une fois le Drive monté
BASE_DIR_DRIVE = Path("/content/drive/MyDrive/UTBM_PF22/datasets/Animals-10")
SOURCE_DIR = BASE_DIR_DRIVE / "raw_data" / "raw-img"  # Dossier contenant les 10 dossiers originaux

# Dossier de sortie par défaut (peut être surchargé, ex. SSD local pour la vitesse)
OUTPUT_DIR = BASE_DIR_DRIVE / "processed_multifidelity"

# --- Définition des Ratios de la population totale (~28 000 images) ---
# 1. Jeu de TEST (HF) : 10% (Reste intouché pour validation équitable)
RATIO_TEST = 0.10

# 2. Jeu d'ENTRAÎNEMENT TOTAL (90% restants) :
#    Dans ce jeu, nous appliquons le ratio Multi-Fidélité :
#    - 10% des images d'entraînement seront HF (CHÈRES)
#    - 90% des images d'entraînement seront BF (PAS CHÈRES, dégradées à la volée)
RATIO_TRAIN_HF_CHERE = 0.10

# Qualité JPEG de sauvegarde (toutes les images sont stockées propres)
SAVE_QUALITY = 95


# --- Script principal de génération ---
def main(source_dir=None, output_dir=None):
    """Génère l'arborescence test / train(HF, BF), toutes images PROPRES.

    Args:
        source_dir: dossier source des images brutes (défaut: SOURCE_DIR sur Drive).
        output_dir: dossier de sortie (défaut: OUTPUT_DIR sur Drive). Passer un
            chemin SSD local (ex. '/content/processed_multifidelity') accélère
            fortement l'écriture des milliers de fichiers.
    """
    source_dir = Path(source_dir) if source_dir is not None else SOURCE_DIR
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR

    print("--- ⚙️ GÉNÉRATION DE L'ENVIRONNEMENT MULTI-FIDÉLITÉ UTBM ---")
    print("ℹ️ Les images BF sont stockées PROPRES : la dégradation est appliquée")
    print("   à la volée (train + test) via src/degradation.py.")
    print(f"📂 Source : {source_dir}")
    print(f"📂 Sortie : {output_dir}")

    if not source_dir.exists():
        print(f"❌ ERREUR : Le dossier source n'existe pas : {source_dir}")
        print("Vérifie que tu as bien téléchargé le dataset dans 'raw_data' sur ton Drive.")
        return

    # Création de l'arborescence cible
    # train/HF, train/BF, test/
    for split in ['train/HF', 'train/BF', 'test']:
        os.makedirs(output_dir / split, exist_ok=True)
        # Création des sous-dossiers de classe (chien, chat...) dans chaque split
        classes = [d for d in os.listdir(source_dir) if os.path.isdir(source_dir / d)]
        for cls in classes:
            os.makedirs(output_dir / split / cls, exist_ok=True)

    # Parcours des classes
    classes = sorted([d for d in os.listdir(source_dir) if os.path.isdir(source_dir / d)])
    total_processed = 0
    errors = 0

    for cls in classes:
        print(f"\nTraitement de la classe : {cls}")
        images = sorted(os.listdir(source_dir / cls))
        random.seed(42)  # Pour que le découpage soit reproductible
        random.shuffle(images)

        num_images = len(images)

        # Calcul des indices de découpage (identique à l'ancien pipeline)
        idx_test = int(num_images * RATIO_TEST)
        idx_train_hf_limit = int((num_images - idx_test) * RATIO_TRAIN_HF_CHERE) + idx_test

        # Séparation des images
        images_test = images[:idx_test]
        images_train_hf = images[idx_test:idx_train_hf_limit]
        images_train_bf = images[idx_train_hf_limit:]

        # --- Fonction interne pour traiter un lot (toujours en propre) ---
        def process_lot(lot_images, split_path):
            nonlocal total_processed, errors
            print(f"   Generating {split_path} ({len(lot_images)} images)...")
            for img_name in tqdm(lot_images):
                src_path = source_dir / cls / img_name
                dst_path = split_path / cls / img_name

                # Éviter de recalculer si le fichier existe déjà
                if dst_path.exists():
                    continue

                try:
                    # Lecture de l'image (PIL plus sûr que read_image pour la compatibilité)
                    with Image.open(src_path) as img:
                        # Si l'image est en CMYK ou RGBA, conversion en RGB
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        # Sauvegarde PROPRE (HF, BF et test sont tous stockés sans dégradation)
                        img.save(dst_path, "JPEG", quality=SAVE_QUALITY)

                    total_processed += 1
                except Exception as e:
                    errors += 1
                    # print(f"❌ Erreur sur {src_path} : {e}") # Décommenter pour debug
                    pass

        # Exécution du traitement pour les 3 jeux (tous propres)
        process_lot(images_test, output_dir / 'test')
        process_lot(images_train_hf, output_dir / 'train' / 'HF')
        process_lot(images_train_bf, output_dir / 'train' / 'BF')

    print("\n--- ✅ GÉNÉRATION TERMINÉE AVEC SUCCÈS ---")
    print(f"📁 Données prêtes dans : {output_dir}")
    print(f"📊 Images traitées : {total_processed}")
    print(f"⚠️ Erreurs de lecture (corrompues ou formats invalides) : {errors}")


if __name__ == "__main__":
    main()
