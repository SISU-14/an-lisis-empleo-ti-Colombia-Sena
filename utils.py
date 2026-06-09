"""utils.py – Helper functions and shared utilities for the app."""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from constants import (
    SALARIO_MEDIANA,
    TECH_MULTIPLIERS,
    MODALIDAD_MULTIPLIERS,
    CONTRATO_MULTIPLIERS,
    CLUSTER_INFO,
    PKL_FILES,
)

# ---------------------------------------------------------------------------
# Model loading  (cached so Streamlit doesn't reload on every interaction)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def cargar_modelos(pkl_files=None):
    """Load all required .pkl model files.

    Parameters
    ----------
    pkl_files : dict | None
        Mapping of logical key → filename. Defaults to PKL_FILES from constants.
    """
    if pkl_files is None:
        pkl_files = PKL_FILES

    modelos = {}
    errores = []
    requeridos = ['kmeans', 'pca', 'scaler', 'features']
    for key, fname in pkl_files.items():
        if os.path.exists(fname):
            try:
                modelos[key] = joblib.load(fname)
            except Exception as e:
                if key in requeridos:
                    errores.append(f"❌ Error al cargar **{fname}**: {e}")
                else:
                    # Opcional pero no requerido — reportar de todos modos
                    errores.append(f"⚠️ **{fname}** existe pero está corrupto: {e}")
        else:
            if key in requeridos:
                errores.append(f"❌ Archivo no encontrado: **{fname}**")
    return modelos, errores


# ---------------------------------------------------------------------------
# Build feature vector
# ---------------------------------------------------------------------------
def construir_vector(techs_sel, experiencia, modalidad, tipo_contrato, feature_cols):
    """Return a single-row DataFrame representing the user profile."""
    fila = {col: 0 for col in feature_cols}
    fila['num_lenguajes']   = len(techs_sel)
    fila['salario']         = SALARIO_MEDIANA[experiencia]
    fila['vacantes']        = 1
    fila['mes_publicacion'] = 6

    for tech in techs_sel:
        col = 'tiene_' + tech.lower().replace(' ', '_').replace('.', '')
        if col in fila:
            fila[col] = 1

    for col in [f'exp_{experiencia}', f'mod_{modalidad}', f'contrato_{tipo_contrato}']:
        if col in fila:
            fila[col] = 1

    return pd.DataFrame([fila])[feature_cols]


# ---------------------------------------------------------------------------
# Salary calculation  (fixed: now returns per-tech individual breakdown)
# ---------------------------------------------------------------------------
def calcular_salario(salario_base, techs_sel, modalidad, contrato):
    """Compute a detailed salary breakdown.

    Returns
    -------
    salario_final : int
    detalle : dict with keys:
        base, tech_bonuses (list of {tech, pct, val}),
        tech_bonus_pct_real, tech_bonus_val, excede_max_tech,
        mod_mul, mod_val, cont_mul, cont_val, final
    """
    # ── Per-technology bonuses ────────────────────────────────────────────────
    tech_bonuses     = []
    tech_bonus_pct   = 0.0
    for tech in techs_sel:
        mul = TECH_MULTIPLIERS.get(tech, 0.0)
        if mul > 0.0:
            bonus_val = int(round(salario_base * mul, -4))
            tech_bonuses.append({'tech': tech, 'pct': mul, 'val': bonus_val})
            tech_bonus_pct += mul

    # Cap at +40 %
    excede_max_tech     = tech_bonus_pct > 0.40
    tech_bonus_pct_real = min(tech_bonus_pct, 0.40)
    tech_bonus_val      = int(round(salario_base * tech_bonus_pct_real, -4))

    # ── Modalidad & contrato ──────────────────────────────────────────────────
    mod_mul  = MODALIDAD_MULTIPLIERS.get(modalidad, 0.0)
    mod_val  = int(round(salario_base * mod_mul, -4))
    cont_mul = CONTRATO_MULTIPLIERS.get(contrato, 0.0)
    cont_val = int(round(salario_base * cont_mul, -4))

    # ── Final salary ──────────────────────────────────────────────────────────
    total_mul    = 1.0 + tech_bonus_pct_real + mod_mul + cont_mul
    salario_final = int(round(salario_base * total_mul, -4))

    detalle = {
        'base':              salario_base,
        'tech_bonuses':      tech_bonuses,       # list — individual per-tech items
        'tech_bonus_pct_real': tech_bonus_pct_real,
        'tech_bonus_val':    tech_bonus_val,
        'excede_max_tech':   excede_max_tech,
        'mod_mul':           mod_mul,
        'mod_val':           mod_val,
        'cont_mul':          cont_mul,
        'cont_val':          cont_val,
        'final':             salario_final,
    }
    return salario_final, detalle


# ---------------------------------------------------------------------------
# Rule-based cluster classifier  (fallback when .pkl models are unavailable)
# ---------------------------------------------------------------------------
def clasificar_por_reglas(techs_sel, experiencia):
    """Classify a profile into a cluster (0-3) using simple heuristic rules.

    This is intentionally a rough approximation — only used when the K-Means
    model files are not available.

    Rules (experience is the primary signal, tech count refines):
        Senior (5+ yr)       → cluster 0
        Semi-Senior (2-5 yr) → cluster 1
        Junior, many techs   → cluster 2  (≥3 technologies)
        Junior, few techs    → cluster 3  (<3 technologies)
    """
    if experiencia == 'Senior':
        return 0
    if experiencia == 'Semi-Senior':
        return 1
    # Junior branch
    return 2 if len(techs_sel) >= 3 else 3


# ---------------------------------------------------------------------------
# Cached salary-distribution sampler  (problem #6 — performance)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generar_distribucion_salarial(cluster_id: int):
    """Generate synthetic salary samples for a cluster using a beta distribution.

    Data is sourced exclusively from CLUSTER_INFO (constants.py) so there is a
    single source of truth and no risk of SAL_PARAMS drifting out of sync.

    Parameters
    ----------
    cluster_id : int  (0–3)

    Returns
    -------
    muestras : np.ndarray  — array of salary samples (length = n_ofertas)
    """
    info    = CLUSTER_INFO[cluster_id]
    sal_min = info['sal_min']
    sal_max = info['sal_max']
    sal_med = info['sal_median']
    n       = info['n_ofertas']

    rango  = sal_max - sal_min
    alpha  = max(((sal_med - sal_min) / rango) * 4, 0.5)
    beta_p = max((1 - (sal_med - sal_min) / rango) * 4, 0.5)

    rng = np.random.default_rng(42)
    return rng.beta(alpha, beta_p, n) * rango + sal_min
