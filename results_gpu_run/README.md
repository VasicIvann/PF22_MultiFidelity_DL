# results_gpu_run — Résultats du run GPU (serveur V100S)

Snapshot complet du run sur le **frontal GPU UTBM** (Tesla V100S), récupéré le 2026-06-15.

## Contenu (3 datasets, multi-seed = 3 seeds, moyenne ± écart-type)
- **Animals-10** : fichiers `results_*.json` / `*.png` à la racine de ce dossier
- **Imagewoof/** , **Intel/** : idem par dataset
- **comparison/** : `robustness_degradation.json`, `cost_ratio_sensitivity.json`, figures
- `_archive_results_gpu.tar.gz` : archive d'origine

## Méthodes par dataset
- Baselines : `results_baseline_{HF,BF,MIXTE}.json`
- Stratégies (base **et** `_optimized` = hyperparamètres Optuna) :
  - S1 `strategy1_transfer_learning`, S2 `strategy2_cotraining_reweighting`,
    S3 `strategy3_curriculum_learning`, S5 `strategy5_ewc`
  - `*_optuna_best_params.json` = métadonnées HPO (best_params, n_trials=20, all_trials)
- (Animals uniquement) `results_strategy3_noisy_student.json` = ancienne strat. 3 (héritage)

## Note
**Food101 abandonné** (problèmes de données : fichiers AppleDouble `._*` du zip Kaggle).
Layout = convention env_config (Animals-10 à la racine). Analyse à faire plus tard.
