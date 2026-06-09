# --- Fichier : src/env_config.py ---
"""
Résolution centralisée des chemins, compatible **Colab** ET **serveur/local**.

- Sur Colab : données sur le SSD `/content/processed_multifidelity` (un seul
  dataset à la fois, extrait depuis un zip sur le Drive), résultats sur le Drive.
- Sur serveur/local (pas de google.colab) : layout sous la racine du projet
  (ou `$PF22_HOME`), avec les **deux datasets côte à côte** :
      <home>/data/Animals-10/processed_multifidelity/{train/HF,train/BF,test}
      <home>/data/Imagewoof/processed_multifidelity/{...}
      <home>/results/                  (Animals-10)
      <home>/results/Imagewoof/        (Imagewoof)
      <home>/results/comparison/

Les notebooks et `train_baselines.py` appellent ces fonctions au lieu de coder
les chemins en dur. `data_dir`/`results_dir` ne dépendent PAS du répertoire
courant (résolus depuis l'emplacement de ce fichier).
"""

import os

DATASETS = ("Animals-10", "Imagewoof")

# Zips Colab (extraits sur le SSD par ensure_dataset_ready)
_COLAB_ZIP = {
    "Animals-10": "/content/drive/MyDrive/UTBM_PF22/datasets/Animals-10/dataset_multifidelity.zip",
    "Imagewoof": "/content/drive/MyDrive/UTBM_PF22/datasets/Imagewoof/dataset_multifidelity.zip",
}
_COLAB_RESULTS = "/content/drive/MyDrive/UTBM_PF22/results"
_COLAB_SSD = "/content/processed_multifidelity"


def in_colab():
    """True si on s'exécute dans Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def _repo_root():
    # src/env_config.py -> racine du repo = parent du dossier src/
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def project_home():
    """Racine du projet en local/serveur (surchargée par la variable PF22_HOME)."""
    return os.environ.get("PF22_HOME", _repo_root())


def data_dir(dataset_name):
    """Dossier processed_multifidelity (train/HF, train/BF, test) du dataset."""
    if in_colab():
        return _COLAB_SSD
    return os.path.join(project_home(), "data", dataset_name, "processed_multifidelity")


def results_dir(dataset_name):
    """Dossier des résultats (JSON/modèles) du dataset."""
    base = _COLAB_RESULTS if in_colab() else os.path.join(project_home(), "results")
    return base if dataset_name == "Animals-10" else os.path.join(base, "Imagewoof")


def comparison_dir():
    """Dossier des figures/CSV comparatifs (bilan, robustesse, sensibilité)."""
    base = _COLAB_RESULTS if in_colab() else os.path.join(project_home(), "results")
    return os.path.join(base, "comparison")


def raw_dir(dataset_name):
    """Dossier des images brutes (avant découpage multi-fidélité)."""
    if in_colab():
        if dataset_name == "Animals-10":
            return "/content/drive/MyDrive/UTBM_PF22/datasets/Animals-10/raw_data"
        return "/content/imagewoof2"
    return os.path.join(project_home(), "data", dataset_name, "raw")


def ensure_dataset_ready(dataset_name):
    """Garantit que le dataset traité est disponible et renvoie son `data_dir`.

    - Colab : monte le Drive et extrait le zip sur le SSD si absent.
    - Serveur/local : vérifie simplement la présence (la préparation doit avoir
      été lancée au préalable).
    """
    dd = data_dir(dataset_name)
    if in_colab():
        import shutil
        import zipfile
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
        except Exception:
            pass
        if not os.path.isdir(dd):
            zp = _COLAB_ZIP[dataset_name]
            if os.path.exists(zp):
                shutil.copy2(zp, '/content/_ds.zip')
                with zipfile.ZipFile('/content/_ds.zip', 'r') as z:
                    z.extractall('/content/')
                os.remove('/content/_ds.zip')
    if not os.path.isdir(dd):
        raise FileNotFoundError(
            f"Dataset introuvable : {dd}\n"
            f"Lance d'abord la préparation du dataset '{dataset_name}'.")
    os.makedirs(results_dir(dataset_name), exist_ok=True)
    return dd
