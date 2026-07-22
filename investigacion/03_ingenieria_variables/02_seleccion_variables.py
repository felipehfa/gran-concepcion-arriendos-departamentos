import json
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
#
# Selección de variables para un problema de regresión de corte transversal
# (precio de arriendo/venta por aviso), sin dimensión temporal:
#   - El split train/val es un hold-out aleatorio (train_test_split), no
#     una ventana de fechas.
#   - La CV de estabilidad usa KFold(shuffle=True): no hay fuga hacia
#     adelante que evitar al no existir una dimensión temporal.
#   - No se calcula PSI train→val: PSI mide drift de distribución entre dos
#     ventanas temporales distintas, y con un split aleatorio de la misma
#     población no aporta información (ambas muestras vienen de la misma
#     distribución por construcción).
#   - No hay deduplicación intra-familia por sufijos (_lag/_ma/...) ni
#     diversificación por dominio: no aplican a este dataset. Solo queda la
#     red de seguridad de correlación final (Paso 4).
#   - Métricas de la curva k: MAE, RMSE y R² (más estándar para precios
#     que sMAPE, que sirve para variables de conteo).

# Rutas relativas a la ubicación del script (no al directorio de trabajo
# actual), para que funcione igual sin importar desde dónde se ejecute
# (ej. VS Code con cwd en la raíz del repo en vez de en esta carpeta).
SCRIPT_DIR  = Path(__file__).resolve().parent
INPUT_PATH  = SCRIPT_DIR / "save" / "ingeniaria_variables" / "datos_ingenieria_variables.csv"
OUTPUT_DIR  = SCRIPT_DIR / "save" / "seleccion_variables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL     = "id_aviso"
TARGET_COL = "costo_total_clp"

# Columnas que existen en el dataset pero no deben entrar como candidatas a
# feature: 'gastos_comunes' porque es uno de los dos sumandos del target
# (costo_total_clp = precio_clp + gastos_comunes) — usarla como feature sería
# entrenar con parte de la respuesta ya en el input. 'precio_clp' por el
# mismo motivo (es el otro sumando, y se conserva en el dataset solo como
# columna informativa/de auditoría, no como target ni como feature).
COLUMNAS_EXCLUIDAS_FEATURES = ["gastos_comunes", "precio_clp"]

VAL_SIZE = 0.2   # hold-out aleatorio para el paso 3 (selección de k)

# Selección por estabilidad
N_MODELS         = 30
CV_FOLDS         = 5
RANDOM_SEED_BASE = 42

# Importancia SHAP (recomendado para variables en distintas escalas)
IMPORTANCE_TYPE = "shap"

# Umbral de correlación post-selección (red de seguridad, sobre el set final)
CORR_THRESHOLD_FINAL = 0.95

# Varianza mínima sobre X_train
CONSTANT_VAR_THRESHOLD = 1e-6

# Presencia mínima: % de modelos donde importancia > 0
PRESENCE_THRESHOLD = 0.6

# Rango de k para la curva MAE vs k
K_MIN = 3
K_MAX = None   # se determina dinámicamente

# Número de semillas sobre las que se promedia cada punto de la curva
N_SEEDS_K_CURVE = 5

# Parámetros del XGBRegressor para selección de variables
#
# n_jobs=1 (no -1): en Windows, usar todos los núcleos dentro de un loop
# que entrena 150 modelos (N_MODELS x CV_FOLDS) hace que el overhead de
# crear/destruir el pool de hilos en cada fit() se acumule y el proceso
# se vuelva extremadamente lento — a veces da la impresión de estar
# colgado, aunque en realidad avanza muy despacio. Con datasets de este
# tamaño (miles de filas, decenas de features), un solo hilo por modelo
# es más rápido y estable que pelear por los núcleos en cada fit.
XGB_PARAMS = dict(
    n_estimators     = 200,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    n_jobs           = 1,
    verbosity        = 0,
    eval_metric      = "mae",
)


# ──────────────────────────────────────────────────────────────────────────────
# Preparación de datos
# ──────────────────────────────────────────────────────────────────────────────

def load_and_split(input_path: Path, id_col: str, target_col: str,
                    val_size: float, seed: int,
                    columnas_excluidas: list = COLUMNAS_EXCLUIDAS_FEATURES) -> tuple:
    """
    Carga el dataset, separa id/target de las features, y arma un split
    aleatorio train/val (no hay orden temporal que preservar).
    """
    df = pd.read_csv(input_path)

    if id_col in df.columns:
        df = df.drop(columns=[id_col])

    df = df.dropna(subset=[target_col])

    excluidas_presentes = [c for c in columnas_excluidas if c in df.columns]
    if excluidas_presentes:
        print(f"  Excluidas de las candidatas a feature (fuga de datos hacia el target): {excluidas_presentes}")
        df = df.drop(columns=excluidas_presentes)

    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, shuffle=True
    )

    print("\nSplit aleatorio train/val (hold-out, sin dimensión temporal)")
    print(f"  Train: {len(X_train)} filas")
    print(f"  Val:   {len(X_val)} filas  (val_size={val_size})")

    return X_train, X_val, y_train, y_val, feature_cols


def remove_constant_features(X: pd.DataFrame,
                              threshold: float = CONSTANT_VAR_THRESHOLD) -> tuple:
    """
    Elimina features con varianza cero o cuasi-cero sobre X_train.
    Returns: (selected, dropped)
    """
    variances = X.var(axis=0, skipna=True)
    low_var   = variances[variances <= threshold].index.tolist()
    nan_feat  = variances[variances.isna()].index.tolist()
    to_drop   = sorted(set(low_var + nan_feat))
    selected  = [c for c in X.columns if c not in to_drop]
    return selected, to_drop


# ──────────────────────────────────────────────────────────────────────────────
# Importancia SHAP (con fallback)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_importance(model: XGBRegressor,
                         X_fold: pd.DataFrame,
                         imp_type: str) -> dict:
    """
    Calcula importancia SHAP mean abs sobre X_fold usando el TreeSHAP nativo
    de xgboost (booster.predict(..., pred_contribs=True)), no la librería
    `shap`: en este entorno Windows, `import shap` dispara una llamada a
    scipy.linalg.inv que crashea el proceso a nivel de SO (incompatibilidad
    del binario LAPACK vendorizado por scipy con esta CPU), sin traceback
    de Python. El resultado de TreeSHAP nativo es equivalente.
    Fallback a 'weight' si imp_type no es 'shap'.
    """
    booster = model.get_booster()

    if imp_type == "shap":
        contribs = booster.predict(xgb.DMatrix(X_fold), pred_contribs=True)
        contribs = contribs[:, :-1]  # última columna = bias/expected value
        mean_abs = np.abs(contribs).mean(axis=0)
        return dict(zip(X_fold.columns, mean_abs.tolist()))

    raw = booster.get_score(importance_type=imp_type)
    return {feat: float(raw.get(feat, 0.0)) for feat in X_fold.columns}


# ──────────────────────────────────────────────────────────────────────────────
# Paso 2 — Selección por estabilidad (K-Fold aleatorio, sobre X_train)
# ──────────────────────────────────────────────────────────────────────────────

def select_features_by_stability(
    X_train:            pd.DataFrame,
    y_train:            pd.Series,
    n_models:           int   = N_MODELS,
    cv_folds:           int   = CV_FOLDS,
    random_seed_base:   int   = RANDOM_SEED_BASE,
    presence_threshold: float = PRESENCE_THRESHOLD,
    importance_type:    str   = IMPORTANCE_TYPE,
) -> pd.DataFrame:
    """
    Calcula estabilidad de features entrenando N modelos con KFold
    aleatorio (shuffle=True) sobre X_train exclusivamente. Cada modelo usa
    una semilla distinta tanto para el split de folds como para el propio
    XGBRegressor, de forma que la estabilidad capture tanto sensibilidad
    al modelo como al particionado.

    stability_score = (1 / (1 + CV)) * presence_pct
    Importancia: SHAP mean abs

    Returns: DataFrame ordenado por stability_score descendente.
    """
    all_importances: dict = {feat: [] for feat in X_train.columns}
    all_mae:  list = []
    all_rmse: list = []

    print(f"\n  Entrenando {n_models} modelos (importancia='{importance_type}', "
          f"KFold aleatorio, {cv_folds} folds)...", flush=True)
    t0 = time.time()

    for i in range(n_models):
        seed = random_seed_base + i
        kf   = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

        fold_imps       = {feat: [] for feat in X_train.columns}
        fold_mae_list   = []
        fold_rmse_list  = []

        for tr_idx, va_idx in kf.split(X_train):
            X_tr = X_train.iloc[tr_idx].fillna(0)
            X_va = X_train.iloc[va_idx].fillna(0)
            y_tr = y_train.iloc[tr_idx]
            y_va = y_train.iloc[va_idx]

            params = {**XGB_PARAMS, "random_state": seed}
            model  = XGBRegressor(**params)
            model.fit(X_tr, y_tr, verbose=False)

            y_pred = np.clip(model.predict(X_va), 0, None)

            fold_mae_list.append(mean_absolute_error(y_va, y_pred))
            fold_rmse_list.append(np.sqrt(mean_squared_error(y_va, y_pred)))

            imp_dict = _compute_importance(model, X_va, importance_type)
            for feat in X_train.columns:
                fold_imps[feat].append(imp_dict.get(feat, 0.0))

        all_mae.append(float(np.mean(fold_mae_list)))
        all_rmse.append(float(np.mean(fold_rmse_list)))

        for feat in X_train.columns:
            vals = fold_imps[feat]
            all_importances[feat].append(float(np.mean(vals)) if vals else 0.0)

        elapsed = time.time() - t0
        eta     = (elapsed / (i + 1)) * (n_models - i - 1)
        print(f"    Modelo {i+1:>2}/{n_models}  "
              f"MAE_fold={all_mae[-1]:,.0f}  RMSE_fold={all_rmse[-1]:,.0f}  "
              f"[{elapsed:>5.1f}s transcurridos, ETA ~{eta:>5.1f}s]", flush=True)

    results = []
    for feat in X_train.columns:
        gains    = np.array(all_importances[feat])
        mean_g   = float(np.mean(gains))
        std_g    = float(np.std(gains))
        median_g = float(np.median(gains))
        presence = float(np.mean(gains > 0))

        if mean_g > 0:
            cv    = std_g / mean_g
            score = (1.0 / (1.0 + cv)) * presence
        else:
            cv    = np.nan
            score = 0.0

        results.append({
            "feature":         feat,
            "mean_imp":        round(mean_g,   6),
            "std_imp":         round(std_g,    6),
            "median_imp":      round(median_g, 6),
            "cv_imp":          round(cv, 4) if not np.isnan(cv) else np.nan,
            "presence_pct":    round(presence, 4),
            "stability_score": round(score,    6),
        })

    stability_df = (
        pd.DataFrame(results)
        .sort_values("stability_score", ascending=False)
        .reset_index(drop=True)
    )
    stability_df["passes_presence"] = stability_df["presence_pct"] >= presence_threshold

    print(f"\n  MAE medio global (folds internos):  {np.mean(all_mae):,.0f} ± {np.std(all_mae):,.0f}")
    print(f"  RMSE medio global (folds internos): {np.mean(all_rmse):,.0f} ± {np.std(all_rmse):,.0f}")
    print(f"  Features con presence >= {presence_threshold}: "
          f"{stability_df['passes_presence'].sum()} / {len(stability_df)}")

    return stability_df


# ──────────────────────────────────────────────────────────────────────────────
# Paso 3 — Selección del número óptimo de features usando Validation MAE
# ──────────────────────────────────────────────────────────────────────────────

def select_k_features_via_val(
    stability_df: pd.DataFrame,
    X_train:      pd.DataFrame,
    y_train:      pd.Series,
    X_val:        pd.DataFrame,
    y_val:        pd.Series,
    k_min:        int = K_MIN,
    k_max:        int = None,
    seed:         int = RANDOM_SEED_BASE,
    n_seeds:      int = N_SEEDS_K_CURVE,
) -> tuple:
    """
    Determina el número óptimo de features evaluando MAE/RMSE/R² en X_val,
    sobre las candidatas que pasan el filtro de presencia, ordenadas por
    stability_score (sin diversificación por dominio).

    Cada punto de la curva (cada k) se promedia sobre n_seeds semillas.

    SE = std(MAE entre semillas EN EL k DEL MÍNIMO) / sqrt(n_seeds)
    threshold = best_mae + SE
    → k más pequeño cuyo MAE_promedio <= threshold

    Returns:
        best_k         : número óptimo de features
        selected_feats : lista de features seleccionadas
        metric_curve   : DataFrame con la curva MAE/RMSE/R² vs k
    """
    eligible_df = stability_df[stability_df["passes_presence"]]
    candidates  = eligible_df.sort_values("stability_score", ascending=False)["feature"].tolist()
    candidates  = [f for f in candidates if f in X_train.columns and f in X_val.columns]

    if not candidates:
        print("  ⚠ Sin candidatas después del filtro de presencia.")
        return 0, [], pd.DataFrame()

    k_max_eff = min(k_max or len(candidates), len(candidates))
    k_range   = range(k_min, k_max_eff + 1)
    seeds     = [seed + i for i in range(n_seeds)]

    print(f"\n  Evaluando curva MAE/RMSE/R² vs k features (k={k_min}..{k_max_eff}, "
          f"promediado sobre {n_seeds} semillas)...", flush=True)
    print(f"  Orden de candidatas por stability_score (primeras 10): {candidates[:10]}")

    records = []
    t0_curve = time.time()
    for k in k_range:
        feats = candidates[:k]

        mae_per_seed, rmse_per_seed, r2_per_seed = [], [], []
        for s in seeds:
            params = {**XGB_PARAMS, "random_state": s, "n_estimators": 300}
            model  = XGBRegressor(**params)
            model.fit(
                X_train[feats].fillna(0),
                y_train,
                eval_set=[(X_val[feats].fillna(0), y_val)],
                verbose=False,
            )

            y_pred = np.clip(model.predict(X_val[feats].fillna(0)), 0, None)
            mae_per_seed.append(mean_absolute_error(y_val, y_pred))
            rmse_per_seed.append(np.sqrt(mean_squared_error(y_val, y_pred)))
            r2_per_seed.append(r2_score(y_val, y_pred))

        mae_k     = float(np.mean(mae_per_seed))
        mae_k_std = float(np.std(mae_per_seed))
        rmse_k    = float(np.mean(rmse_per_seed))
        r2_k      = float(np.mean(r2_per_seed))

        records.append({
            "k": k,
            "mae_val": round(mae_k, 2),
            "mae_val_std_seeds": round(mae_k_std, 2),
            "rmse_val": round(rmse_k, 2),
            "r2_val": round(r2_k, 4),
        })

        elapsed_curve = time.time() - t0_curve
        print(f"    k={k:>3}  MAE={mae_k:,.0f} (±{mae_k_std:,.0f})  "
              f"RMSE={rmse_k:,.0f}  R²={r2_k:.3f}  "
              f"[{elapsed_curve:>5.1f}s transcurridos]", flush=True)

    metric_curve = pd.DataFrame(records)

    # ── Regla de 1 error estándar ─────────────────────
    argmin_idx  = metric_curve["mae_val"].idxmin()
    argmin_k    = int(metric_curve.loc[argmin_idx, "k"])
    best_mae    = float(metric_curve.loc[argmin_idx, "mae_val"])
    std_at_best = float(metric_curve.loc[argmin_idx, "mae_val_std_seeds"])
    se_at_best  = std_at_best / np.sqrt(n_seeds)
    threshold   = best_mae + se_at_best

    parsimonious = metric_curve[metric_curve["mae_val"] <= threshold]
    best_k       = int(parsimonious.iloc[0]["k"])
    best_k_row   = metric_curve[metric_curve["k"] == best_k].iloc[0]

    print(f"\n  MAE mínimo: {best_mae:,.0f} (k={argmin_k}, std_entre_semillas={std_at_best:,.2f})")
    print(f"  1-SE (std_en_mínimo/√{n_seeds}): {se_at_best:,.2f}  →  threshold: {threshold:,.2f}")
    print(f"  k óptimo (parsimonia, 1-SE rule): {best_k}  "
          f"MAE={best_k_row['mae_val']:,.0f}  RMSE={best_k_row['rmse_val']:,.0f}  "
          f"R²={best_k_row['r2_val']:.3f}")

    selected_feats = candidates[:best_k]
    return best_k, selected_feats, metric_curve


# ──────────────────────────────────────────────────────────────────────────────
# Paso 4 — Eliminación de correlación post-selección (red de seguridad)
# ──────────────────────────────────────────────────────────────────────────────

def remove_correlated_after_selection(
    X_train:      pd.DataFrame,
    selected:     list,
    stability_df: pd.DataFrame,
    threshold:    float = CORR_THRESHOLD_FINAL,
    score_col:    str = "stability_score",
) -> tuple:
    """
    Elimina correlación entre las features YA seleccionadas, quedándose
    con la de mayor stability_score de cada par correlacionado.

    Returns: (keep, removed)
    """
    if not selected:
        return [], []

    avail     = [f for f in selected if f in X_train.columns]
    X_sub     = X_train[avail].fillna(0)
    score_map = stability_df.set_index("feature")[score_col].to_dict()
    ordered   = sorted(avail, key=lambda f: score_map.get(f, 0.0), reverse=True)

    corr    = X_sub[ordered].corr().abs()
    keep    = []
    removed = []

    for feat in ordered:
        if feat in removed:
            continue
        keep.append(feat)
        highly_corr = [
            other for other in ordered
            if other != feat
            and other not in removed
            and corr.loc[feat, other] > threshold
        ]
        removed.extend(highly_corr)

    return keep, removed


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────────────────────────

def run_feature_selection(input_path: Path = INPUT_PATH,
                           output_dir: Path = OUTPUT_DIR) -> dict:
    """
    Corre el pipeline completo de selección de variables para el modelo
    de precio (regresión de corte transversal).

    Exporta únicamente:
      - seleccion_variables_reporte.json  (todas las etapas)
      - selected_features.csv             (solo nombres de features finales)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando dataset: {input_path}")
    raw_df = pd.read_csv(input_path)
    n_filas_total = len(raw_df)

    X_train, X_val, y_train, y_val, feature_cols = load_and_split(
        input_path, ID_COL, TARGET_COL, VAL_SIZE, RANDOM_SEED_BASE
    )
    print(f"Features iniciales (excluyendo '{ID_COL}' y '{TARGET_COL}'): {len(feature_cols)}")

    # ── Paso 1: Eliminar constantes ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("PASO 1 — Eliminación de features constantes (sobre X_train)")
    print("="*60)

    selected_not_const, dropped_const = remove_constant_features(
        X_train, threshold=CONSTANT_VAR_THRESHOLD
    )
    print(f"  Eliminadas (varianza ≈ 0): {len(dropped_const)}")
    if dropped_const:
        print(f"    {dropped_const}")
    print(f"  Features restantes: {len(selected_not_const)}")

    X_train = X_train[selected_not_const]

    # ── Paso 2: Selección por estabilidad ────────────────────────────────
    print(f"\n{'='*60}")
    print("PASO 2 — Selección por estabilidad (K-Fold aleatorio, sobre X_train)")
    print(f"  N_MODELS={N_MODELS}, CV_FOLDS={CV_FOLDS}")
    print(f"  Importancia: '{IMPORTANCE_TYPE}'")
    print("="*60)

    stability_df = select_features_by_stability(
        X_train,
        y_train,
        n_models           = N_MODELS,
        cv_folds            = CV_FOLDS,
        random_seed_base    = RANDOM_SEED_BASE,
        presence_threshold  = PRESENCE_THRESHOLD,
        importance_type     = IMPORTANCE_TYPE,
    )

    print(f"\nTop 20 features por stability_score:")
    print(
        stability_df[["feature", "mean_imp", "cv_imp",
                       "presence_pct", "stability_score"]]
        .head(20)
        .to_string(index=False)
    )

    # ── Paso 3: k óptimo via Validation MAE ──────────────────────────────
    print(f"\n{'='*60}")
    print("PASO 3 — Selección de k óptimo via Validation MAE")
    print("="*60)

    best_k, selected_feats, metric_curve = select_k_features_via_val(
        stability_df = stability_df,
        X_train      = X_train,
        y_train      = y_train,
        X_val        = X_val[selected_not_const],
        y_val        = y_val,
        k_min        = K_MIN,
        k_max        = K_MAX,
        seed         = RANDOM_SEED_BASE,
        n_seeds      = N_SEEDS_K_CURVE,
    )

    print(f"\n  Features seleccionadas (k={best_k}):")
    for f in selected_feats:
        row = stability_df[stability_df["feature"] == f].iloc[0]
        print(f"    {f:<40} stability_score={row['stability_score']:.4f}  "
              f"presence={row['presence_pct']:.2f}")

    # ── Paso 4: Eliminación de correlación post-selección ────────────────
    print(f"\n{'='*60}")
    print(f"PASO 4 — Eliminación de correlación post-selección (red de seguridad)")
    print(f"  Umbral correlación: {CORR_THRESHOLD_FINAL}")
    print("="*60)

    final_features, removed_corr = remove_correlated_after_selection(
        X_train      = X_train,
        selected     = selected_feats,
        stability_df = stability_df,
        threshold    = CORR_THRESHOLD_FINAL,
        score_col    = "stability_score",
    )

    print(f"  Eliminadas por correlación: {len(removed_corr)}")
    if removed_corr:
        print(f"    {removed_corr}")
    print(f"  Features finales: {len(final_features)}")

    # ── Paso 5: Exportar resultados ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("PASO 5 — Exportar resultados")
    print("="*60)

    json_path    = output_dir / "seleccion_variables_reporte.json"
    features_path = output_dir / "selected_features.csv"

    reporte = {
        "config": {
            "input_path": str(input_path),
            "id_col": ID_COL,
            "target_col": TARGET_COL,
            "val_size": VAL_SIZE,
            "n_models": N_MODELS,
            "cv_folds": CV_FOLDS,
            "random_seed_base": RANDOM_SEED_BASE,
            "importance_type": IMPORTANCE_TYPE,
            "presence_threshold": PRESENCE_THRESHOLD,
            "constant_var_threshold": CONSTANT_VAR_THRESHOLD,
            "corr_threshold_final": CORR_THRESHOLD_FINAL,
            "k_min": K_MIN,
            "n_seeds_k_curve": N_SEEDS_K_CURVE,
        },
        "resumen": {
            "n_filas_total": n_filas_total,
            "n_features_iniciales": len(feature_cols),
            "n_features_tras_constantes": len(selected_not_const),
            "n_candidatas_presence": int(stability_df["passes_presence"].sum()),
            "k_optimo": best_k,
            "n_eliminadas_correlacion_final": len(removed_corr),
            "n_features_finales": len(final_features),
        },
        "features_constantes_eliminadas": dropped_const,
        "stability_selection": stability_df.to_dict(orient="records"),
        "metric_curve_k": metric_curve.to_dict(orient="records"),
        "seleccion_k": {
            "best_k": best_k,
            "candidatas_usadas": selected_feats,
        },
        "correlacion_final": {
            "features_eliminadas": removed_corr,
            "features_mantenidas": final_features,
        },
        "features_finales": final_features,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Reporte completo (todas las etapas): {json_path}")

    if final_features:
        pd.DataFrame({"feature": final_features}).to_csv(features_path, index=False)
        print(f"  ✓ Nombres de features seleccionadas:  {features_path}")
    else:
        print("  ⚠ No se seleccionaron features — revisa PRESENCE_THRESHOLD y K_MIN.")

    # ── Resumen ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESUMEN")
    print("="*60)
    print(f"  Dataset total:                          {n_filas_total}")
    print(f"  Features iniciales:                     {len(feature_cols)}")
    print(f"  Después de eliminar constantes:          {len(selected_not_const)}")
    print(f"  Candidatas (presence >= {PRESENCE_THRESHOLD}):        "
          f"{int(stability_df['passes_presence'].sum())}")
    print(f"  k óptimo (1-SE rule, Val MAE):            {best_k}")
    print(f"  Eliminadas por correlación final:        {len(removed_corr)}")
    print(f"  Features finales:                        {len(final_features)}")

    return reporte


if __name__ == "__main__":
    run_feature_selection(INPUT_PATH, OUTPUT_DIR)