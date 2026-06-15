# --- Fichier : src/analysis_utils.py ---
"""
Utilitaires partagés par les notebooks d'analyse (18, 20, 21).

Lit les résultats du snapshot GPU (`results_gpu_run/`) — Animals-10 à la racine,
Imagewoof/ et Intel/ en sous-dossiers. Stdlib pur (aucun pandas/numpy requis ici)
pour rester testable hors notebook ; les notebooks construisent le DataFrame.
"""

import os
import json

# Datasets du run GPU (Food101 abandonné).
DATASETS = ["Animals-10", "Imagewoof", "Intel"]

# Modèles comparés : (label, fichier JSON, couleur, famille, optimisé_Optuna ?)
# Familles : baseline / S1 / S2 / S3 / S5. Chaque stratégie a une version de base
# et une version "_optimized" (hyperparamètres trouvés par Optuna).
MODELS = [
    ("BL1 (HF)",       "results_baseline_HF.json",                              "#9aa7b8", "baseline", False),
    ("BL2 (BF)",       "results_baseline_BF.json",                              "#6e7a8a", "baseline", False),
    ("BL3 (Mixte)",    "results_baseline_MIXTE.json",                           "#3b4a5a", "baseline", False),
    ("S1 Transfer",    "results_strategy1_transfer_learning.json",             "#4C72B0", "S1", False),
    ("S1 +Optuna",     "results_strategy1_transfer_learning_optimized.json",   "#2f4f7a", "S1", True),
    ("S2 CoTrain",     "results_strategy2_cotraining_reweighting.json",         "#55A868", "S2", False),
    ("S2 +Optuna",     "results_strategy2_cotraining_reweighting_optimized.json","#357a4a", "S2", True),
    ("S3 Curriculum",  "results_strategy3_curriculum_learning.json",            "#C44E52", "S3", False),
    ("S3 +Optuna",     "results_strategy3_curriculum_learning_optimized.json",  "#8a2f33", "S3", True),
    ("S5 EWC",         "results_strategy5_ewc.json",                            "#8172B2", "S5", False),
    ("S5 +Optuna",     "results_strategy5_ewc_optimized.json",                  "#574a85", "S5", True),
]


def find_results_root():
    """Localise le dossier results_gpu_run (override par PF22_RESULTS_ROOT)."""
    candidates = [
        os.environ.get("PF22_RESULTS_ROOT"),
        "results_gpu_run", "../results_gpu_run", "../../results_gpu_run",
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "Dossier 'results_gpu_run' introuvable. Définis PF22_RESULTS_ROOT ou lance "
        "depuis la racine du repo / notebooks/.")


def ds_dir(root, dataset):
    """Dossier des résultats d'un dataset (Animals-10 = racine)."""
    return root if dataset == "Animals-10" else os.path.join(root, dataset)


def load_rows(root):
    """Charge toutes les paires (dataset, modèle) disponibles en liste de dicts."""
    rows = []
    for ds in DATASETS:
        d_dir = ds_dir(root, ds)
        if not os.path.isdir(d_dir):
            continue
        for label, fn, color, fam, opt in MODELS:
            p = os.path.join(d_dir, fn)
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            rows.append({
                "dataset": ds,
                "model": label,
                "family": fam,
                "optimized": opt,
                "color": color,
                "HF": d.get("accuracy_HF"),
                "HF_std": d.get("accuracy_HF_std", 0.0),
                "BF": d.get("accuracy_BF"),
                "BF_std": d.get("accuracy_BF_std", 0.0),
                "Mixte": d.get("accuracy_Mixte"),
                "Mixte_std": d.get("accuracy_Mixte_std", 0.0),
                "data_cost": d.get("data_cost_CA"),
                "compute": d.get("compute_images_seen"),
                "total_cost": d.get("total_cost_CA"),
                "delta_HF": d.get("delta_accuracy_HF"),   # présent sur les versions _optimized
                "n_hf_pool": d.get("n_hf_pool"),
                "n_bf_pool": d.get("n_bf_pool"),
                "hf_images_seen": d.get("hf_images_seen"),
                "bf_images_seen": d.get("bf_images_seen"),
            })
    return rows


def optuna_gains(root):
    """Pour chaque (dataset, stratégie), gain HF de la version optimisée vs base."""
    pairs = [
        ("S1", "results_strategy1_transfer_learning"),
        ("S2", "results_strategy2_cotraining_reweighting"),
        ("S3", "results_strategy3_curriculum_learning"),
        ("S5", "results_strategy5_ewc"),
    ]
    out = []
    for ds in DATASETS:
        d_dir = ds_dir(root, ds)
        for fam, stem in pairs:
            pb = os.path.join(d_dir, stem + ".json")
            po = os.path.join(d_dir, stem + "_optimized.json")
            if not (os.path.exists(pb) and os.path.exists(po)):
                continue
            b = json.load(open(pb, encoding="utf-8"))
            o = json.load(open(po, encoding="utf-8"))
            out.append({
                "dataset": ds, "family": fam,
                "HF_base": b.get("accuracy_HF"), "HF_opt": o.get("accuracy_HF"),
                "gain_HF": o.get("accuracy_HF") - b.get("accuracy_HF"),
                "delta_HF_reported": o.get("delta_accuracy_HF"),
            })
    return out
