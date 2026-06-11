#!/usr/bin/env bash
# ============================================================
#  PF22 — Exécution de TOUS les notebooks (01 -> 45) en série
#  Conçu pour tourner dans un tmux (survit à la déconnexion).
#  Chaque notebook est exécuté "inplace" : ses sorties (texte +
#  figures) sont enregistrées dans le .ipynb lui-même.
# ============================================================
set -u

# --- Réglages ---
# W&B en mode hors-ligne pour ne JAMAIS bloquer un run sans attention.
# (Mettre "online" si tu as fait `wandb login` et que tu veux le tracking live.)
export WANDB_MODE=${WANDB_MODE:-offline}

# Racine du projet = dossier de ce script
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/notebooks" || { echo "Dossier notebooks/ introuvable"; exit 1; }

LOG="$HOME/pf22_run_$(date +%Y%m%d_%H%M%S).log"
DONE_FLAG="$HOME/PF22_DONE"
rm -f "$DONE_FLAG"

NBX=(jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1)

echo "============================================================" | tee -a "$LOG"
echo " PF22 — démarrage : $(date)" | tee -a "$LOG"
echo " WANDB_MODE=$WANDB_MODE | GPU : $(python -c 'import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")' 2>/dev/null)" | tee -a "$LOG"
echo " Log : $LOG" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

fail=0
failed_list=""
# Les préfixes 01..45 sont à 2 chiffres -> le tri lexical = tri numérique.
for nb in $(ls [0-9][0-9]_*.ipynb | sort); do
  echo "" | tee -a "$LOG"
  echo ">>> $(date '+%H:%M:%S')  $nb" | tee -a "$LOG"
  start=$(date +%s)
  if "${NBX[@]}" "$nb" >> "$LOG" 2>&1; then
    dur=$(( $(date +%s) - start ))
    echo "    OK    $nb  (${dur}s)" | tee -a "$LOG"
  else
    dur=$(( $(date +%s) - start ))
    echo "    ECHEC $nb  (${dur}s) — on continue" | tee -a "$LOG"
    fail=$((fail+1))
    failed_list="$failed_list $nb"
  fi
done

echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo " TERMINÉ : $(date)" | tee -a "$LOG"
if [ "$fail" -eq 0 ]; then
  echo " ✅ Tous les notebooks ont réussi." | tee -a "$LOG"
else
  echo " ⚠️  $fail notebook(s) en échec :$failed_list" | tee -a "$LOG"
fi
echo "============================================================" | tee -a "$LOG"

# Drapeau de fin (à vérifier d'un coup d'œil)
{ echo "fin: $(date)"; echo "echecs: $fail$failed_list"; echo "log: $LOG"; } > "$DONE_FLAG"
echo "Drapeau de fin écrit : $DONE_FLAG"
