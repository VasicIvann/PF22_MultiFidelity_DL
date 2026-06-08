# --- Fichier : src/cost.py ---
"""
Source UNIQUE de la métrique de coût du projet multi-fidélité.

Deux coûts DISTINCTS sont désormais séparés explicitement :

  1. Coût DONNÉE (Crédit d'Annotation, CA) — coût d'ACQUISITION du jeu de données,
     payé UNE SEULE FOIS : nombre d'images distinctes × coût unitaire. Il ne dépend
     PAS du nombre d'époques d'entraînement. C'est la métrique centrale du projet
     (le verrou économique = le coût de la donnée).

  2. Coût CALCUL — proxy du coût d'entraînement : nombre total d'images traitées
     (époques × taille du train), agnostique au matériel. Le temps wall-clock est
     suivi séparément (dépendant du matériel/charge serveur).

Coût unitaire en fonction de la dégradation (modèle « résolution au carré »)
---------------------------------------------------------------------------
Le coût d'acquisition d'une image est piloté par sa RÉSOLUTION effective (qualité
du capteur / soin de la prise de vue / stockage), donc ~ proportionnel au nombre
de pixels (résolution²). Le bruit et la compression JPEG sont considérés comme du
post-traitement « gratuit » (ils n'augmentent pas le coût d'acquisition).

    coût(d) = C_MIN + (C_HF - C_MIN) * (d / size)^2

avec deux ancres calibrées :
  - image pleine résolution (HF, d = size = 224)        -> C_HF      = 10 CA
  - BF canonique (d = 64, le niveau des résultats)       -> ~ 1 CA
C_MIN est dérivé de ces deux ancres (≈ 0,2 : coût plancher minimal de toute image).

Conséquence importante : toutes les méthodes qui exploitent le MÊME pool de données
(HF + BF) ont le MÊME coût donnée. Elles ne se distinguent alors que par leur
PRÉCISION et leur coût de CALCUL — ce qui isole proprement la contribution de la
stratégie d'apprentissage du coût d'acquisition.
"""

DEFAULT_SIZE = 224

# Ancres de calibrage
COST_HF = 10.0                 # coût d'une image pleine résolution (référence)
CANONICAL_BF_DOWNSCALE = 64    # résolution intermédiaire du BF « canonique »
CANONICAL_BF_COST = 1.0        # coût cible du BF canonique (calibrage)


def _c_min(size=DEFAULT_SIZE):
    """Coût plancher dérivé des deux ancres (HF=COST_HF à `size`px ; BF=1 à 64px)."""
    q = (CANONICAL_BF_DOWNSCALE / float(size)) ** 2
    # CANONICAL_BF_COST = C_MIN + (COST_HF - C_MIN) * q  ->  on résout C_MIN
    return (CANONICAL_BF_COST - COST_HF * q) / (1.0 - q)


C_MIN = _c_min()  # ≈ 0.20


def unit_cost(downscale=None, size=DEFAULT_SIZE):
    """Coût d'acquisition d'une image (en CA), fonction de sa résolution.

    Args:
        downscale: résolution intermédiaire de sous-échantillonnage. None ou
            >= size -> image pleine résolution (HF) -> COST_HF. Une valeur plus
            petite (image plus dégradée) coûte moins cher.
        size: résolution de référence (224).

    Le bruit gaussien et la compression JPEG n'entrent PAS dans le coût
    (post-traitement gratuit).
    """
    if downscale is None or downscale >= size:
        return float(COST_HF)
    q = (float(downscale) / float(size)) ** 2
    return float(C_MIN + (COST_HF - C_MIN) * q)


def hf_unit_cost(size=DEFAULT_SIZE):
    """Coût unitaire d'une image HF (pleine résolution)."""
    return unit_cost(None, size)


def bf_unit_cost(downscale=CANONICAL_BF_DOWNSCALE, size=DEFAULT_SIZE):
    """Coût unitaire d'une image BF au niveau de dégradation donné (défaut: canonique)."""
    return unit_cost(downscale, size)


def data_cost(n_hf=0, n_bf=0, bf_downscale=CANONICAL_BF_DOWNSCALE, size=DEFAULT_SIZE):
    """Coût DONNÉE total (acquisition, UNIQUE) en CA.

    = n_hf images HF (pleine résolution) + n_bf images BF (au niveau `bf_downscale`).
    Indépendant du nombre d'époques.
    """
    return n_hf * unit_cost(None, size) + n_bf * unit_cost(bf_downscale, size)


# ============================================================
#  Variante paramétrée par le ratio HF:BF (analyse de sensibilité)
# ============================================================
# Le modèle résolution² est RE-CALIBRÉ pour chaque ratio R : C_HF reste fixe,
# et le BF canonique (64px) coûte C_HF / R. C_MIN est re-dérivé en conséquence.

def c_min_for_ratio(ratio, size=DEFAULT_SIZE):
    """C_MIN re-calibré pour un ratio HF:BF donné (BF canonique = C_HF / ratio)."""
    q = (CANONICAL_BF_DOWNSCALE / float(size)) ** 2
    c_bf_canonical = COST_HF / float(ratio)
    return (c_bf_canonical - COST_HF * q) / (1.0 - q)


def unit_cost_ratio(downscale, ratio, size=DEFAULT_SIZE):
    """Coût unitaire sous un ratio HF:BF donné (résolution² re-calibrée)."""
    if downscale is None or downscale >= size:
        return float(COST_HF)
    cmin = c_min_for_ratio(ratio, size)
    q = (float(downscale) / float(size)) ** 2
    return float(cmin + (COST_HF - cmin) * q)


def data_cost_ratio(n_hf=0, n_bf=0, ratio=10.0,
                    bf_downscale=CANONICAL_BF_DOWNSCALE, size=DEFAULT_SIZE):
    """Coût donnée sous un ratio HF:BF donné."""
    return (n_hf * unit_cost_ratio(None, ratio, size)
            + n_bf * unit_cost_ratio(bf_downscale, ratio, size))
