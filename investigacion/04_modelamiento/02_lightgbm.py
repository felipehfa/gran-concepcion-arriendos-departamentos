import faulthandler
faulthandler.enable()   # traceback de C si hay un crash nativo, en vez de morir en silencio

import json
import re
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import pickle
from datetime import datetime
from pathlib import Path
from scipy import stats as scipy_stats
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class NumpyEncoder(json.JSONEncoder):
    """Permite serializar tipos numpy (int64, float64, ndarray) en JSON."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
# Regresión de corte transversal (sin dimensión temporal): predice precio_clp
# de arriendo/venta de un aviso en el Gran Concepción.
#
# Pipeline: split estratificado por precio → Optuna (CV sobre train) →
# bagging de N_SEEDS_BAGGING modelos → evaluación en val/test. Además incluye
# una validación multi-semilla (`run_stability_across_seeds`) para separar
# mejora real de ruido de partición, dado que el dataset es chico
# (~250-300 filas).

SCRIPT_DIR = Path(__file__).resolve().parent

# El dataset completo lo exporta 01_ingenieria_variables.py y la lista de
# features seleccionadas la exporta 02_seleccion_variables.py. Rutas
# relativas a la ubicación de ESTE script (no al directorio de trabajo
# actual), para que funcione igual sin importar desde dónde se ejecute.
INGENIERIA_VARIABLES_DIR = SCRIPT_DIR.parent / "03_ingenieria_variables" / "save" / "ingeniaria_variables"
SELECCION_VARIABLES_DIR  = SCRIPT_DIR.parent / "03_ingenieria_variables" / "save" / "seleccion_variables"
FEATURES_PATH = SELECCION_VARIABLES_DIR / "selected_features.csv"
DATASET_PATH  = INGENIERIA_VARIABLES_DIR / "datos_ingenieria_variables.csv"
SAVE_MODEL_DIR = SCRIPT_DIR / "save" / "model"

ID_COL     = "id_aviso"
TARGET_COL = "costo_total_clp"

SEED       = 42
MODEL_NAME = "lightgbm_regression_precio"

TEST_SIZE = 0.15
VAL_SIZE  = 0.15   # sobre el total; se ajusta internamente sobre el resto tras sacar test

# Número de modelos del ensamble de bagging en el entrenamiento final.
# Cada modelo usa los mismos hiperparámetros (best_params) pero una
# semilla distinta. Las predicciones finales son el promedio del ensamble.
N_SEEDS_BAGGING = 10

N_TRIALS_OPTUNA = 50
CV_SPLITS_OPTUNA = 5

# n_jobs=1: en Windows, usar todos los núcleos (-1) dentro de un loop que
# entrena muchos modelos seguidos (Optuna: 50 trials x 5 folds = 250 fits;
# bagging: 5 modelos) puede generar overhead de hilos severo y dar la
# impresión de que el proceso está colgado. Un solo hilo por modelo es
# más rápido y estable para este tamaño de datos.
LGB_NJOBS = 1

# Objective de LightGBM FIJO (no forma parte del espacio de búsqueda de
# Optuna): dejar que Optuna lo tratara como un hiperparámetro más hacía que
# el "ganador" cambiara de una corrida a otra por ruido de los folds, no por
# una diferencia real de desempeño.
LGB_OBJECTIVE = "regression"   # L2 / squared error

# Cantidad de estratos (quintiles) de precio_clp usados para el split
# train/val/test estratificado (ver `split_data`).
N_ESTRATOS_PRECIO = 5

# Validación multi-semilla: repite split→Optuna→bagging→evaluación con
# estas semillas de PARTICIÓN (distintas de las semillas de bagging,
# SEED..SEED+N_SEEDS_BAGGING-1) para medir cuánto varían las métricas según
# cómo cae el split, no según la inicialización del modelo.
N_SEEDS_STABILITY = 8
STABILITY_SEED_BASE = 1000
SEEDS_STABILITY = [STABILITY_SEED_BASE + i for i in range(N_SEEDS_STABILITY)]

# MISMOS n_trials/cv_splits que la corrida principal: cada una de las
# N_SEEDS_STABILITY particiones corre la optimización de Optuna completa, no
# una versión recortada — de lo contrario no se puede distinguir "el split
# cambió el resultado" de "el split cambió el resultado porque además le
# dimos menos presupuesto de búsqueda".
N_TRIALS_OPTUNA_STABILITY = N_TRIALS_OPTUNA
CV_SPLITS_OPTUNA_STABILITY = CV_SPLITS_OPTUNA

# Etiquetado oportunidad/caro/precio_de_mercado (test): deciles de
# costo_total_clp (TARGET_COL) usados para normalizar el error del modelo
# por segmento de costo total. Distintos de N_ESTRATOS_PRECIO (quintiles),
# que se usan solo para el split y el reporting de evaluación existente y
# no cambian.
N_DECILES_OPORTUNIDAD = 10

# Constante de escalamiento del MAD (median absolute deviation) para que sea
# comparable a una desviación estándar bajo normalidad (1/Φ⁻¹(0.75) ≈ 1.4826).
MAD_SCALE_CONST = 1.4826

# Umbrales (en desviaciones robustas, z_robusto) para las etiquetas de
# costo total. error = costo_total_real - costo_total_predicho:
# costo_total_real por DEBAJO de lo esperado por el modelo (ajustado por
# decil) → "oportunidad" (z muy negativo); por ENCIMA → "caro" (z muy
# positivo).
UMBRAL_OPORTUNIDAD = 1.0
UMBRAL_CARO = 1.0

# Niveles de confianza según percentiles del coeficiente de variación
# (std/mean) de las predicciones del ensamble por fila: menor CV (los
# modelos del ensamble concuerdan más) → mayor confianza.
N_GRUPOS_CONFIANZA = 3
ETIQUETAS_CONFIANZA = ["alta confianza", "confianza media", "baja confianza"]


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────

def load_selected_features(path: Path = FEATURES_PATH) -> list:
    features = pd.read_csv(path)["feature"].tolist()
    print(f"\nFeatures seleccionadas ({len(features)}): {features}")
    return features


def load_dataset(path: Path = DATASET_PATH, features: list = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    unnamed_cols = [c for c in df.columns if re.match(r"^Unnamed:\s*\d+$", c)]
    if unnamed_cols:
        print(f"  ⚠ Descartando columnas índice residuales: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)
    if features is not None:
        cols = [c for c in [ID_COL] + features + [TARGET_COL] if c in df.columns]
        df = df[cols]
    return df


def construir_estratos_precio(precio: pd.Series, n_estratos: int = N_ESTRATOS_PRECIO) -> pd.Series:
    """Bins de `precio` (vía pd.qcut) usados para estratificar el split
    train/val/test, de forma que cada partición mantenga una mezcla similar
    de propiedades baratas/caras."""
    return pd.qcut(precio, q=n_estratos, labels=False, duplicates="drop")


def split_data(df: pd.DataFrame, test_size: float, val_size: float, seed: int,
                target_col: str = TARGET_COL, n_estratos: int = N_ESTRATOS_PRECIO) -> tuple:
    """
    Split train/val/test estratificado por quintil de precio_clp.

    Primero se separa test_size del total; luego, del resto, se separa
    val_size/(1-test_size) para que la proporción final de val sobre el
    total sea la solicitada. Los estratos del segundo split se recalculan
    sobre el subconjunto train_val (mismos bins, subíndice del array
    calculado sobre el total).
    """
    val_ratio_on_rest = val_size / (1 - test_size)

    estratos = construir_estratos_precio(df[target_col], n_estratos)

    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, shuffle=True, stratify=estratos
    )
    train, val = train_test_split(
        train_val, test_size=val_ratio_on_rest, random_state=seed, shuffle=True,
        stratify=estratos.loc[train_val.index],
    )

    print("\nSplit estratificado por quintil de precio_clp (train/val/test)")
    print(f"  Train: {len(train)} filas ({len(train)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val)} filas ({len(val)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test)} filas ({len(test)/len(df)*100:.1f}%)")

    return train, val, test


# ──────────────────────────────────────────────────────────────────────────────
# Métricas
# ──────────────────────────────────────────────────────────────────────────────

def compute_ape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Error porcentual absoluto por observación: |y_real - y_pred| / y_real."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.abs(y_true - y_pred) / y_true


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    mae_v   = mean_absolute_error(y_true, y_pred)
    rmse_v  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2_v    = r2_score(y_true, y_pred)
    ape     = compute_ape(y_true, y_pred) * 100
    mape_v  = float(np.mean(ape))
    mdape_v = float(np.median(ape))
    prefix = f"[{label}] " if label else ""
    print(f"  {prefix}MAE={mae_v:,.0f}  RMSE={rmse_v:,.0f}  R²={r2_v:.4f}  "
          f"MAPE={mape_v:.2f}%  MdAPE={mdape_v:.2f}%")
    return {
        "mae": round(mae_v, 4), "rmse": round(rmse_v, 4), "r2": round(r2_v, 4),
        "mape": round(mape_v, 4), "mdape": round(mdape_v, 4),
    }


def mae_quintil_precio_alto(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE del quintil de precio más alto (Q5) — el segmento "premium" que
    en este dataset concentra la mayor parte del error agregado."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    quintiles = pd.qcut(y_true, q=5, labels=False, duplicates="drop")
    q_top = quintiles.max()
    mask = quintiles == q_top
    return float(mean_absolute_error(y_true[mask], y_pred[mask]))


# ──────────────────────────────────────────────────────────────────────────────
# Optimización de hiperparámetros
# ──────────────────────────────────────────────────────────────────────────────

def optimize_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials:  int = N_TRIALS_OPTUNA,
    cv_splits: int = CV_SPLITS_OPTUNA,
    objective: str = LGB_OBJECTIVE,
    seed:      int = SEED,
    cv_metric: str = "mae",
) -> tuple:
    """
    Optuna minimiza `cv_metric` ('mae' o 'rmse') promedio de un KFold
    aleatorio (shuffle=True), calculado exclusivamente sobre X_train/y_train.

    val queda reservado para early stopping del modelo final y para la
    evaluación reportada — nunca influye en qué hiperparámetros se eligen.
    """
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    def objective_fn(trial):
        params = {
            "objective":         objective,
            "metric":            "rmse",
            "verbosity":         -1,
            "seed":              seed,
            "n_jobs":            LGB_NJOBS,
            "bagging_freq":      1,
            "learning_rate":     trial.suggest_float("learning_rate",     0.01, 0.10, log=True),
            "max_depth":         trial.suggest_int(  "max_depth",         2, 6),
            "num_leaves":        trial.suggest_int(  "num_leaves",        8, 64),
            "min_child_samples": trial.suggest_int(  "min_child_samples", 1, 20),
            "subsample":         trial.suggest_float("subsample",         0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree",  0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha",         1e-2, 20.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda",        1e-2, 20.0, log=True),
            "min_split_gain":    trial.suggest_float("min_split_gain",    0.0,  5.0),
        }
        fold_scores = []
        for tr_idx, va_idx in kf.split(X_train):
            X_tr = X_train.iloc[tr_idx].fillna(0)
            X_va = X_train.iloc[va_idx].fillna(0)
            y_tr = y_train.iloc[tr_idx]
            y_va = y_train.iloc[va_idx]
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval   = lgb.Dataset(X_va, label=y_va, reference=dtrain)
            model  = lgb.train(
                params, dtrain, num_boost_round=500,
                valid_sets=[dval], valid_names=["val"],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            y_pred = np.clip(model.predict(X_va, num_iteration=model.best_iteration), 0, None)
            if cv_metric == "rmse":
                fold_scores.append(float(np.sqrt(mean_squared_error(y_va, y_pred))))
            else:
                fold_scores.append(mean_absolute_error(y_va, y_pred))
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
    best_trial = study.best_trial
    best_params = {**best_trial.params, "objective": objective, "metric": "rmse", "bagging_freq": 1}
    print(f"\n  Mejor {cv_metric.upper()} CV (solo train, sin val): {best_trial.value:,.4f}  "
          f"(trial {best_trial.number})")
    print(f"  Objective (fijo): {objective}")
    print(f"  Params:       {best_trial.params}")
    optim_info = {
        "best_cv_score":  round(best_trial.value, 6),
        "cv_metric":      cv_metric,
        "best_trial":     best_trial.number,
        "best_params":    best_params,
        "objective_fijo": objective,
        "n_trials":       n_trials,
        "cv_splits":      cv_splits,
        "cv_solo_train":  True,
    }
    return best_params, optim_info


# ──────────────────────────────────────────────────────────────────────────────
# Entrenamiento — Bagging de semillas
# ──────────────────────────────────────────────────────────────────────────────

def entrenar_ensamble_bagging(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    best_params: dict,
    n_seeds_bagging: int = N_SEEDS_BAGGING,
    seed_base: int = SEED,
    verbose: bool = True,
) -> list:
    """
    Entrena un ensamble de `n_seeds_bagging` modelos con los mismos
    best_params pero semillas distintas (derivadas de `seed_base`). Cada
    modelo usa X_val/y_val para su propio early stopping. No persiste nada
    a disco — eso lo hace `train_model`, que envuelve esta función para la
    corrida principal.

    Returns: lista de lgb.Booster entrenados (el ensamble completo).
    """
    X_train_filled = X_train.fillna(0)
    X_val_filled   = X_val.fillna(0)
    dtrain = lgb.Dataset(X_train_filled, label=y_train)
    dval   = lgb.Dataset(X_val_filled, label=y_val, reference=dtrain)

    models = []
    if verbose:
        print(f"\nEntrenando ensamble de {n_seeds_bagging} modelo(s) "
              f"({best_params.get('objective','?')})...")

    for i in range(n_seeds_bagging):
        seed_i = seed_base + i
        params = {**best_params, "verbosity": -1, "seed": seed_i, "n_jobs": LGB_NJOBS}

        if verbose:
            print(f"\n  ── Modelo {i+1}/{n_seeds_bagging} (seed={seed_i}) ──")
        model = lgb.train(
            params, dtrain,
            num_boost_round=1000,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(100 if (verbose and i == 0) else 0),
            ],
        )
        if verbose:
            val_score = model.best_score["val"]["rmse"]
            print(f"    Mejor iteración: {model.best_iteration}  |  "
                  f"Mejor val score: {val_score:.4f}")
        models.append(model)

    return models


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    best_params: dict,
    n_seeds_bagging: int = N_SEEDS_BAGGING,
) -> list:
    """
    Entrena el ensamble de bagging de la corrida principal (vía
    `entrenar_ensamble_bagging`, semillas SEED..SEED+n_seeds_bagging-1) y lo
    persiste en disco.

    Returns: lista de lgb.Booster entrenados (el ensamble completo).
    """
    models = entrenar_ensamble_bagging(
        X_train, y_train, X_val, y_val, best_params,
        n_seeds_bagging=n_seeds_bagging, seed_base=SEED, verbose=True,
    )

    SAVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = SAVE_MODEL_DIR / f"{MODEL_NAME}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(models, f)
    print(f"\n  Ensamble guardado ({len(models)} modelos): {model_path}")
    return models


def predict_ensemble_matrix(models: list, X: pd.DataFrame) -> np.ndarray:
    """Predicciones individuales de cada modelo del ensamble, sin promediar.
    Returns: array (n_modelos, n_filas)."""
    X_filled = X.fillna(0)
    return np.stack([
        np.clip(m.predict(X_filled, num_iteration=m.best_iteration), 0, None)
        for m in models
    ])


def predict_ensemble(models: list, X: pd.DataFrame) -> np.ndarray:
    """Predicción del ensamble: promedio de las predicciones individuales."""
    return predict_ensemble_matrix(models, X).mean(axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Entrenamiento + evaluación end-to-end sobre un split ya construido
# ──────────────────────────────────────────────────────────────────────────────

def entrenar_y_evaluar_modelo(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
    features: list,
    seed:            int = SEED,
    n_trials:        int = N_TRIALS_OPTUNA,
    cv_splits:       int = CV_SPLITS_OPTUNA,
    n_seeds_bagging: int = N_SEEDS_BAGGING,
    persistir_modelo: bool = True,
) -> dict:
    """
    Optimiza hiperparámetros (Optuna, CV solo sobre train), entrena el
    ensamble de bagging final y evalúa en val/test, sobre un split
    train/val/test ya construido (ver `split_data`).

    Si `persistir_modelo=True`, además guarda el ensamble en disco.
    """
    X_train = train[features].fillna(0)
    X_val   = val[features].fillna(0)
    X_test  = test[features].fillna(0)

    y_train = train[TARGET_COL].astype(float)
    y_val   = val[TARGET_COL].astype(float)
    y_test  = test[TARGET_COL].astype(float)

    print(f"\n{'='*60}")
    print(f"OPTIMIZACIÓN — Optuna ({n_trials} trials, KFold={cv_splits}, CV solo train)")
    print("="*60)
    best_params, optim_info = optimize_hyperparams(
        X_train, y_train, n_trials=n_trials, cv_splits=cv_splits, seed=seed,
    )

    print(f"\n{'='*60}")
    print(f"ENTRENAMIENTO FINAL  [Bagging x{n_seeds_bagging}]")
    print("="*60)
    t_bagging0 = time.time()
    if persistir_modelo:
        models = train_model(X_train, y_train, X_val, y_val, best_params,
                              n_seeds_bagging=n_seeds_bagging)
    else:
        models = entrenar_ensamble_bagging(
            X_train, y_train, X_val, y_val, best_params,
            n_seeds_bagging=n_seeds_bagging, seed_base=seed, verbose=True,
        )
    tiempo_bagging = time.time() - t_bagging0
    print(f"\n  Tiempo ENTRENAMIENTO FINAL (bagging x{n_seeds_bagging}): "
          f"{tiempo_bagging:.1f}s ({tiempo_bagging/n_seeds_bagging:.1f}s/modelo)")

    y_val_pred  = predict_ensemble(models, X_val)
    y_test_pred = predict_ensemble(models, X_test)

    eval_val  = evaluate_model(y_val.values,  y_val_pred,  X_val, y_train, "Val")
    eval_test = evaluate_model(y_test.values, y_test_pred, X_test, y_train, "Test")

    return {
        "best_params":  best_params,
        "optim_info":   optim_info,
        "models":       models,
        "X_test":       X_test,
        "y_test":       y_test,
        "y_val_pred":   y_val_pred,
        "y_test_pred":  y_test_pred,
        "eval_val":     eval_val,
        "eval_test":    eval_test,
        "tiempo_bagging_seg": round(tiempo_bagging, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Validación multi-semilla — separar mejora real de ruido de partición
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_split_seed(
    df:       pd.DataFrame,
    features: list,
    seed:     int,
    n_trials:        int = N_TRIALS_OPTUNA_STABILITY,
    cv_splits:       int = CV_SPLITS_OPTUNA_STABILITY,
    n_seeds_bagging: int = N_SEEDS_BAGGING,
) -> dict:
    """
    Corre UNA repetición completa del pipeline (split estratificado por
    precio → Optuna con objective fijo → bagging → evaluación) para una
    semilla de partición `seed`. No exporta modelo a disco ni corre la
    importancia de features — se usa solo para medir cuánto varían las
    métricas de val/test según cómo cae el split, dado que el dataset es
    chico (~250-300 filas).
    """
    train, val, test = split_data(df, TEST_SIZE, VAL_SIZE, seed)

    X_train = train[features].fillna(0)
    y_train = train[TARGET_COL].astype(float)
    X_val   = val[features].fillna(0)
    y_val   = val[TARGET_COL].astype(float)
    X_test  = test[features].fillna(0)
    y_test  = test[TARGET_COL].astype(float)

    best_params, _ = optimize_hyperparams(
        X_train, y_train, n_trials=n_trials, cv_splits=cv_splits, seed=seed,
    )

    # seed_base derivado de `seed` (y no de SEED) para no repetir por
    # accidente las semillas de bagging de la corrida principal.
    models = entrenar_ensamble_bagging(
        X_train, y_train, X_val, y_val, best_params,
        n_seeds_bagging=n_seeds_bagging, seed_base=seed * 1000, verbose=False,
    )

    y_val_pred  = predict_ensemble(models, X_val)
    y_test_pred = predict_ensemble(models, X_test)

    val_metrics  = compute_metrics(y_val.values,  y_val_pred)
    test_metrics = compute_metrics(y_test.values, y_test_pred)
    q5_mae_test  = mae_quintil_precio_alto(y_test.values, y_test_pred)

    return {"seed": seed, "val": val_metrics, "test": test_metrics, "q5_mae_test": q5_mae_test}


def run_stability_across_seeds(
    df:       pd.DataFrame,
    features: list,
    seeds:           list = SEEDS_STABILITY,
    n_trials:        int  = N_TRIALS_OPTUNA_STABILITY,
    cv_splits:       int  = CV_SPLITS_OPTUNA_STABILITY,
    n_seeds_bagging: int  = N_SEEDS_BAGGING,
) -> pd.DataFrame:
    """
    Repite split estratificado → Optuna (objective fijo) → bagging →
    evaluación para cada semilla en `seeds`, y devuelve un DataFrame con las
    métricas de cada corrida (una fila por semilla de partición).

    Con un dataset de ~250-300 filas, un solo split no alcanza para saber si
    un cambio en el pipeline mejoró el modelo o solo tocó una partición más
    favorable — esto reporta la distribución (media ± std) de MAE/RMSE/R² en
    val y test, y del MAE del quintil de precio más alto (Q5), a través de
    las particiones. No se lanza automáticamente en `run_all` por su costo.
    """
    mismo_n_trials = (n_trials == N_TRIALS_OPTUNA) and (cv_splits == CV_SPLITS_OPTUNA)
    print(f"\n{'='*60}")
    print(f"VALIDACIÓN MULTI-SEMILLA — {len(seeds)} particiones "
          f"(split estratificado por precio, objective fijo='{LGB_OBJECTIVE}')")
    print(f"  n_trials Optuna por partición: {n_trials}  "
          f"({'igual' if mismo_n_trials else 'DISTINTO'} a la corrida principal: "
          f"{N_TRIALS_OPTUNA} trials, {CV_SPLITS_OPTUNA} folds)")
    print("="*60)

    filas = []
    t0 = time.time()
    for i, seed in enumerate(seeds):
        t_seed0 = time.time()
        print(f"\n  ── Partición {i+1}/{len(seeds)} (seed={seed}) ──", flush=True)
        r = evaluate_split_seed(
            df, features, seed, n_trials=n_trials, cv_splits=cv_splits,
            n_seeds_bagging=n_seeds_bagging,
        )
        tiempo_seed = time.time() - t_seed0
        print(f"    VAL  MAE={r['val']['mae']:,.0f}  RMSE={r['val']['rmse']:,.0f}  R²={r['val']['r2']:.3f}  |  "
              f"TEST MAE={r['test']['mae']:,.0f}  RMSE={r['test']['rmse']:,.0f}  R²={r['test']['r2']:.3f}  |  "
              f"Q5_MAE_test={r['q5_mae_test']:,.0f}  |  tiempo={tiempo_seed:.1f}s")
        filas.append({
            "seed":     seed,
            "val_mae":  r["val"]["mae"],  "val_rmse":  r["val"]["rmse"],  "val_r2":  r["val"]["r2"],
            "test_mae": r["test"]["mae"], "test_rmse": r["test"]["rmse"], "test_r2": r["test"]["r2"],
            "q5_mae_test": r["q5_mae_test"],
            "tiempo_seg": round(tiempo_seed, 1),
        })
    tiempo_total = time.time() - t0

    stability_df = pd.DataFrame(filas)
    stability_df.attrs["n_trials_optuna"] = n_trials
    stability_df.attrs["cv_splits_optuna"] = cv_splits
    stability_df.attrs["tiempo_total_seg"] = tiempo_total

    print(f"\n  Distribución de métricas a través de {len(seeds)} semillas de partición "
          f"(media ± std  [min – max]):")
    for col, nombre in [
        ("val_mae",  "VAL  MAE"), ("val_r2",  "VAL  R²"),
        ("test_mae", "TEST MAE"), ("test_r2", "TEST R²"),
        ("q5_mae_test", "Q5 MAE (test)"),
    ]:
        serie = stability_df[col]
        print(f"    {nombre:<15} {serie.mean():>14,.4f} ± {serie.std():<10,.4f}  "
              f"[{serie.min():,.4f} – {serie.max():,.4f}]")

    print(f"\n  Tiempo total: {tiempo_total:.1f}s ({tiempo_total/60:.1f} min) para {len(seeds)} "
          f"particiones × {n_trials} trials Optuna cada una "
          f"({'igual' if mismo_n_trials else 'DISTINTO'} a la corrida principal). "
          f"Tiempo medio por partición: {stability_df['tiempo_seg'].mean():.1f}s.")

    return stability_df


def imprimir_resumen_comparativo(eval_test_principal: dict, stability_df: pd.DataFrame) -> None:
    """
    Compara la corrida principal (seed=SEED) contra la distribución de
    métricas obtenida sobre múltiples semillas de partición: un std chico
    relativo a la media sugiere que el desempeño observado es real y no
    ruido de partición, dado el tamaño chico del dataset.
    """
    r2_mean, r2_std = stability_df["test_r2"].mean(), stability_df["test_r2"].std()
    mae_mean, mae_std = stability_df["test_mae"].mean(), stability_df["test_mae"].std()
    q5_mean, q5_std = stability_df["q5_mae_test"].mean(), stability_df["q5_mae_test"].std()

    print(f"\n{'='*60}")
    print("RESUMEN COMPARATIVO — corrida principal vs. multi-semilla")
    print("="*60)
    print(f"  Multi-semilla (split estratificado por precio, objective fijo='{LGB_OBJECTIVE}', "
          f"{len(stability_df)} semillas de partición):")
    print(f"    - TEST R²:  media={r2_mean:.3f}  std={r2_std:.3f}  "
          f"[{stability_df['test_r2'].min():.3f} – {stability_df['test_r2'].max():.3f}]")
    print(f"    - TEST MAE: media={mae_mean:,.0f}  std={mae_std:,.0f}  "
          f"[{stability_df['test_mae'].min():,.0f} – {stability_df['test_mae'].max():,.0f}]")
    print(f"    - Corrida principal (seed={SEED}) → "
          f"TEST R²={eval_test_principal['metricas_globales']['r2']:.3f}  "
          f"MAE={eval_test_principal['metricas_globales']['mae']:,.0f}")

    cv_r2      = abs(r2_std / r2_mean) if r2_mean else float("nan")
    cv_mae     = abs(mae_std / mae_mean) if mae_mean else float("nan")
    cv_q5_mae  = abs(q5_std / q5_mean) if q5_mean else float("nan")

    print(f"\n  Segmento premium (Q5) — MAE y su varianza entre particiones:")
    print(f"    - Q5 MAE (test): media={q5_mean:,.0f}  std={q5_std:,.0f}  CV={cv_q5_mae:.3f}  "
          f"[{stability_df['q5_mae_test'].min():,.0f} – {stability_df['q5_mae_test'].max():,.0f}]")
    print(f"    - CV MAE global de test: {cv_mae:.3f}")
    if cv_q5_mae > cv_mae:
        print(f"    → El segmento premium (Q5) no solo concentra más error que el resto (ver MAE "
              f"por quintil en la evaluación de Test), sino que además su MAE varía más entre "
              f"particiones (CV={cv_q5_mae:.3f}) que el MAE global de test (CV={cv_mae:.3f}): "
              f"es la parte del dataset donde el modelo es menos confiable y menos consistente.")
    else:
        print(f"    → El segmento premium (Q5) sigue concentrando más error que el resto, pero en "
              f"esta corrida su varianza entre particiones (CV={cv_q5_mae:.3f}) no superó la del "
              f"MAE global de test (CV={cv_mae:.3f}).")

    if cv_r2 < 0.15:
        veredicto = "R² converge razonablemente entre semillas: la señal parece real, no ruido de partición."
    else:
        veredicto = ("R² sigue oscilando bastante entre semillas: con ~250-300 filas el ruido de "
                      "partición probablemente sigue dominando sobre la señal del modelo.")
    print(f"\n  Veredicto: {veredicto}")


# ──────────────────────────────────────────────────────────────────────────────
# Baselines de comparación
# ──────────────────────────────────────────────────────────────────────────────

def compute_baselines(y_train: pd.Series, y_true: np.ndarray,
                       X_true: pd.DataFrame) -> dict:
    """
    Baselines de comparación para un problema sin orden temporal:
      - Media de train: predice siempre la media de precio_clp en train.
      - Mercado (si están disponibles las columnas): precio_m2_sector * m2.
    """
    baselines = {}

    mean_pred = np.full_like(y_true, fill_value=y_train.mean(), dtype=float)
    baselines["media_train"] = {
        "mae": round(mean_absolute_error(y_true, mean_pred), 4),
    }

    m2_col = "superficie_util_m2"
    sector_col = "precio_m2_sector_departamento"
    if m2_col in X_true.columns and sector_col in X_true.columns:
        market_pred = (X_true[sector_col] * X_true[m2_col]).values
        baselines["mercado_m2_sector"] = {
            "mae": round(mean_absolute_error(y_true, market_pred), 4),
        }

    return baselines


# ──────────────────────────────────────────────────────────────────────────────
# Evaluación exhaustiva (solo impresión en consola, sin exportar a disco)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X_true: pd.DataFrame,
    y_train: pd.Series,
    label: str,
) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)

    print(f"\n{'='*60}")
    print(f"EVALUACIÓN — {label}")
    print("="*60)

    print("\n  Métricas globales:")
    global_metrics = compute_metrics(y_true, y_pred)

    print("\n  Baselines de comparación:")
    baselines = compute_baselines(y_train, y_true, X_true)
    for nombre, info in baselines.items():
        ganancia = info["mae"] - global_metrics["mae"]
        print(f"    {nombre:<20} MAE={info['mae']:,.0f}  "
              f"→ Ganancia del modelo: {ganancia:+,.0f}")

    residuals = y_true - y_pred
    skew_val = float(scipy_stats.skew(residuals))
    kurt_val = float(scipy_stats.kurtosis(residuals))

    print(f"\n  Residuos:")
    print(f"    Media={residuals.mean():+,.0f}  Std={residuals.std():,.0f}  "
          f"Skew={skew_val:.3f}  Kurt={kurt_val:.3f}")

    print("\n  MAE/MAPE/MdAPE por quintil de precio real:")
    quintiles = pd.qcut(y_true, q=5, labels=False, duplicates="drop")
    mae_por_quintil = {}
    for q in sorted(np.unique(quintiles)):
        mask = quintiles == q
        q_mae = mean_absolute_error(y_true[mask], y_pred[mask])
        q_ape = compute_ape(y_true[mask], y_pred[mask]) * 100
        q_mape  = float(np.mean(q_ape))
        q_mdape = float(np.median(q_ape))
        q_range = f"[{y_true[mask].min():,.0f} - {y_true[mask].max():,.0f}]"
        bar = "█" * int(q_mae / global_metrics["mae"] * 10)
        print(f"    Q{q+1} {q_range:<28} MAE={q_mae:,.0f}  MAPE={q_mape:.2f}%  MdAPE={q_mdape:.2f}%  {bar}")
        mae_por_quintil[int(q)] = {
            "rango": q_range, "mae": round(q_mae, 4),
            "mape": round(q_mape, 4), "mdape": round(q_mdape, 4),
        }

    return {
        "metricas_globales":   global_metrics,
        "baselines":           baselines,
        "residuos": {
            "media":    round(float(residuals.mean()), 4),
            "std":      round(float(residuals.std()),  4),
            "skewness": round(skew_val, 4),
            "kurtosis": round(kurt_val, 4),
        },
        "mae_por_quintil_precio": mae_por_quintil,
    }


# ──────────────────────────────────────────────────────────────────────────────
# SHAP nativo de LightGBM (Booster.predict con pred_contrib=True)
# ──────────────────────────────────────────────────────────────────────────────

def compute_shap_native(models: list, X: pd.DataFrame) -> dict:
    """
    Importancia SHAP usando el TreeSHAP nativo de LightGBM
    (Booster.predict(..., pred_contrib=True)), sin la librería `shap`
    externa. Es exacto (no aproximado) para modelos de árboles.

    pred_contrib devuelve un array (n_samples, n_features + 1): la
    última columna es la contribución del valor base (bias), que se
    descarta acá porque no corresponde a ninguna feature.

    Se promedia |contribución| entre todos los modelos del ensamble.
    """
    X_filled = X.fillna(0)
    contribs_sum = None

    for model in models:
        contribs = model.predict(X_filled, num_iteration=model.best_iteration, pred_contrib=True)
        contribs = contribs[:, :-1]   # descartar columna de bias
        mean_abs = np.abs(contribs).mean(axis=0)
        contribs_sum = mean_abs if contribs_sum is None else contribs_sum + mean_abs

    contribs_mean = contribs_sum / len(models)
    shap_importance = dict(
        pd.Series(contribs_mean, index=X.columns)
        .sort_values(ascending=False)
        .round(6)
    )
    return shap_importance


# ──────────────────────────────────────────────────────────────────────────────
# Etiquetado oportunidad/caro/precio_de_mercado (test)
# ──────────────────────────────────────────────────────────────────────────────

def mediana_y_mad(x: np.ndarray) -> tuple:
    """Mediana y MAD (median absolute deviation, sin escalar) de un array."""
    mediana = float(np.median(x))
    mad = float(np.median(np.abs(x - mediana)))
    return mediana, mad


def calcular_z_robusto_por_decil(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_deciles: int = N_DECILES_OPORTUNIDAD,
    mad_scale: float = MAD_SCALE_CONST,
) -> dict:
    """
    Divide `y_true` (costo_total_clp) en `n_deciles` grupos y calcula, para
    cada fila, el error (costo_total_real - costo_total_predicho)
    normalizado de forma robusta contra la mediana y el MAD del error
    dentro de su propio decil de costo total:

        z_robusto = (error - mediana_decil) / (MAD_decil * mad_scale)

    Se usa mediana/MAD en vez de media/std porque son menos sensibles a
    outliers — relevante dado que cada decil tiene pocas filas en el set
    de test (~4-5 filas por grupo con ~40-45 filas de test).
    """
    error = y_true - y_pred
    deciles = construir_estratos_precio(pd.Series(y_true), n_estratos=n_deciles).values

    mediana_por_decil = np.zeros_like(error)
    mad_por_decil = np.zeros_like(error)
    stats_por_decil = {}
    for d in sorted(np.unique(deciles)):
        mask = deciles == d
        mediana_d, mad_d = mediana_y_mad(error[mask])
        mediana_por_decil[mask] = mediana_d
        mad_por_decil[mask] = mad_d
        stats_por_decil[int(d)] = {
            "n": int(mask.sum()),
            "mediana_error": round(mediana_d, 4),
            "mad_error": round(mad_d, 4),
        }

    mad_ajustado = mad_por_decil * mad_scale
    # Evitar división por cero en deciles donde todos los errores son
    # idénticos (MAD=0) — puede pasar con pocas filas por grupo.
    mad_ajustado_seguro = np.where(mad_ajustado == 0, 1e-6, mad_ajustado)
    z_robusto = (error - mediana_por_decil) / mad_ajustado_seguro

    return {
        "deciles": deciles, "error": error,
        "mediana_por_decil": mediana_por_decil, "mad_por_decil": mad_por_decil,
        "z_robusto": z_robusto, "stats_por_decil": stats_por_decil,
    }


def clasificar_confianza_cv(
    cv: np.ndarray,
    n_grupos: int = N_GRUPOS_CONFIANZA,
    etiquetas: list = ETIQUETAS_CONFIANZA,
) -> np.ndarray:
    """
    Clasifica el coeficiente de variación (std/mean) de las predicciones del
    ensamble por fila en grupos de confianza, según percentiles de CV sobre
    el propio set de test: menor CV (los modelos del ensamble concuerdan
    más entre sí para esa fila) → mayor confianza.
    """
    bin_idx = pd.qcut(cv, q=n_grupos, labels=False, duplicates="drop")
    n_bins_reales = int(np.max(bin_idx)) + 1
    if n_bins_reales < n_grupos:
        # Pocos valores únicos de CV para separar en n_grupos distintos:
        # usar solo los extremos disponibles (más confianza / menos
        # confianza) en vez de inventar categorías intermedias vacías.
        etiquetas_usadas = ([etiquetas[0]] if n_bins_reales == 1
                             else [etiquetas[0], etiquetas[-1]])
    else:
        etiquetas_usadas = etiquetas
    return np.array([etiquetas_usadas[i] for i in bin_idx])


def etiquetar_oportunidades(
    test: pd.DataFrame,
    models: list,
    X_test: pd.DataFrame,
    umbral_oportunidad: float = UMBRAL_OPORTUNIDAD,
    umbral_caro: float = UMBRAL_CARO,
) -> tuple:
    """
    Para cada propiedad del set de test, compara su costo total real
    (arriendo + gastos comunes) contra el promedio predicho por el ensamble
    de bagging, normalizado dentro de su propio decil de costo total (ver
    `calcular_z_robusto_por_decil`), y la etiqueta como:
      - "oportunidad": costo total real por DEBAJO de lo esperado (z_robusto
        muy negativo) — más barata que propiedades comparables.
      - "caro": costo total real por ENCIMA de lo esperado (z_robusto muy
        positivo).
      - "precio_de_mercado": dentro del rango normal para su decil.

    Además calcula un nivel de confianza por fila a partir del coeficiente
    de variación (std/mean) de las N_SEEDS_BAGGING predicciones individuales
    del ensamble: si los modelos discrepan mucho entre sí para una fila,
    la etiqueta es menos confiable aunque el z_robusto sea grande.
    """
    y_true = test[TARGET_COL].astype(float).values
    pred_matrix = predict_ensemble_matrix(models, X_test)
    y_pred = pred_matrix.mean(axis=0)

    z_info = calcular_z_robusto_por_decil(y_true, y_pred)
    z_robusto = z_info["z_robusto"]

    etiqueta = np.where(
        z_robusto < -umbral_oportunidad, "oportunidad",
        np.where(z_robusto > umbral_caro, "caro", "precio_de_mercado"),
    )

    pred_std = pred_matrix.std(axis=0)
    cv_ensamble = pred_std / np.where(y_pred == 0, 1e-6, y_pred)
    nivel_confianza = clasificar_confianza_cv(cv_ensamble)

    etiqueta_final = np.array([f"{e} ({c})" for e, c in zip(etiqueta, nivel_confianza)])

    resultado = pd.DataFrame({
        "costo_total_real":      y_true,
        "costo_total_predicho":  y_pred,
        "error":               z_info["error"],
        "decil_precio":        z_info["deciles"],
        "mediana_error_decil": z_info["mediana_por_decil"],
        "mad_error_decil":     z_info["mad_por_decil"],
        "z_robusto":           z_robusto,
        "cv_ensamble":         cv_ensamble,
        "nivel_confianza":     nivel_confianza,
        "etiqueta":            etiqueta,
        "etiqueta_final":      etiqueta_final,
    })
    if ID_COL in test.columns:
        resultado.insert(0, ID_COL, test[ID_COL].values)

    print(f"\n{'='*60}")
    print(f"ETIQUETADO OPORTUNIDAD/CARO — Test ({len(resultado)} filas, "
          f"{N_DECILES_OPORTUNIDAD} deciles de costo total, "
          f"umbral_oportunidad={umbral_oportunidad}, umbral_caro={umbral_caro})")
    print("="*60)

    resumen_etiqueta_confianza = pd.crosstab(resultado["etiqueta"], resultado["nivel_confianza"])
    resumen_decil = resultado["decil_precio"].value_counts().sort_index()
    resumen_pct = (resultado["etiqueta"].value_counts(normalize=True) * 100).round(1)

    print("\n  Conteo por etiqueta × nivel de confianza:")
    print(resumen_etiqueta_confianza.to_string())
    print("\n  Conteo por decil de precio:")
    print(resumen_decil.to_string())
    print("\n  Distribución de etiquetas (%):")
    print(resumen_pct.to_string())

    return resultado, {
        "stats_por_decil":              z_info["stats_por_decil"],
        "resumen_etiqueta_confianza":   resumen_etiqueta_confianza,
        "resumen_decil":                resumen_decil,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────────────────────────

def run_all(df: pd.DataFrame, features: list) -> dict:
    """
    Corre el pipeline completo sobre el split estratificado principal
    (seed=SEED): optimiza hiperparámetros, entrena y evalúa el ensamble de
    bagging, corre SHAP y exporta modelo + métricas a disco.

    La validación multi-semilla (`run_stability_across_seeds`) no se corre
    acá por ser costosa (N_SEEDS_STABILITY particiones × Optuna completo);
    se llama aparte cuando se quiera confirmar que el resultado no es ruido
    de partición.
    """
    SAVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Features faltantes en el dataset:\n{missing}")

    print("\n" + "="*60)
    print("MODELO — LightGBM | Precio de arriendo/venta (Gran Concepción)")
    print("="*60)

    train, val, test = split_data(df, TEST_SIZE, VAL_SIZE, SEED)

    print("\n── Estadísticas del target (precio_clp) ──")
    for nombre, sub in [("Train", train), ("Val", val), ("Test", test)]:
        y = sub[TARGET_COL].astype(float)
        print(f"  {nombre}: media={y.mean():,.0f}  std={y.std():,.0f}  "
              f"min={y.min():,.0f}  max={y.max():,.0f}")

    resultado = entrenar_y_evaluar_modelo(
        train, val, test, features,
        seed=SEED, n_trials=N_TRIALS_OPTUNA, cv_splits=CV_SPLITS_OPTUNA,
        n_seeds_bagging=N_SEEDS_BAGGING, persistir_modelo=True,
    )

    print(f"\n{'='*60}")
    print(f"SHAP nativo LightGBM — Feature Importance "
          f"(Test, promedio de {len(resultado['models'])} modelos)")
    print("="*60)
    shap_importance = compute_shap_native(resultado["models"], resultado["X_test"])
    print(pd.Series(shap_importance).head(15).to_string())
    resultado["shap_importance"] = shap_importance

    oportunidades_df, oportunidades_resumen = etiquetar_oportunidades(
        test, resultado["models"], resultado["X_test"],
    )
    oportunidades_path = SAVE_MODEL_DIR / f"{MODEL_NAME}_oportunidades_test.csv"
    oportunidades_df.to_csv(oportunidades_path, index=False)
    resumen_etiqueta_confianza_path = SAVE_MODEL_DIR / f"{MODEL_NAME}_oportunidades_resumen_etiqueta_confianza.csv"
    oportunidades_resumen["resumen_etiqueta_confianza"].to_csv(resumen_etiqueta_confianza_path)
    resumen_decil_path = SAVE_MODEL_DIR / f"{MODEL_NAME}_oportunidades_resumen_decil.csv"
    oportunidades_resumen["resumen_decil"].to_csv(resumen_decil_path)
    print(f"\n  Etiquetas exportadas:            {oportunidades_path}")
    print(f"  Resumen etiqueta×confianza:      {resumen_etiqueta_confianza_path}")
    print(f"  Resumen por decil:               {resumen_decil_path}")

    best_params  = resultado["best_params"]
    optim_info   = resultado["optim_info"]
    models       = resultado["models"]
    eval_val     = resultado["eval_val"]
    eval_test    = resultado["eval_test"]
    y_val_pred   = resultado["y_val_pred"]
    y_test_pred  = resultado["y_test_pred"]
    X_test       = resultado["X_test"]
    y_test       = resultado["y_test"]

    # ── Exportar métricas (JSON) ──
    eval_data = {
        "model":               MODEL_NAME,
        "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features":            features,
        "objetivo_lgb":        best_params.get("objective", "?"),
        "hiperparametros":     best_params,
        "optimizacion":        optim_info,
        "n_seeds_bagging":     N_SEEDS_BAGGING,
        "tiempo_bagging_seg":  resultado["tiempo_bagging_seg"],
        "split": {
            "test_size": TEST_SIZE, "val_size": VAL_SIZE, "seed": SEED,
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
        },
        "evaluacion_val":       eval_val,
        "evaluacion_test":      eval_test,
        "shap_importance":      {k: float(v) for k, v in shap_importance.items()},
        "oportunidades_test": {
            "n_deciles":          N_DECILES_OPORTUNIDAD,
            "umbral_oportunidad": UMBRAL_OPORTUNIDAD,
            "umbral_caro":        UMBRAL_CARO,
            "stats_por_decil":    oportunidades_resumen["stats_por_decil"],
            "distribucion_etiquetas": oportunidades_df["etiqueta"].value_counts().to_dict(),
        },
    }

    metrics_path = SAVE_MODEL_DIR / f"{MODEL_NAME}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\n  Métricas exportadas: {metrics_path}")

    # ── Resumen final ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESUMEN FINAL")
    print("="*60)
    print(f"  Objective:        {best_params.get('objective','?')}")
    print(f"  Mejor {optim_info['cv_metric'].upper()} CV: {optim_info['best_cv_score']:,.4f}  (solo train)")
    print(f"  Bagging:          {N_SEEDS_BAGGING} modelo(s)  "
          f"(tiempo: {resultado['tiempo_bagging_seg']:.1f}s)")
    print(f"  VAL  → MAE={eval_val['metricas_globales']['mae']:,.0f}  "
          f"RMSE={eval_val['metricas_globales']['rmse']:,.0f}  R²={eval_val['metricas_globales']['r2']:.4f}  "
          f"MAPE={eval_val['metricas_globales']['mape']:.2f}%")
    print(f"  TEST → MAE={eval_test['metricas_globales']['mae']:,.0f}  "
          f"RMSE={eval_test['metricas_globales']['rmse']:,.0f}  R²={eval_test['metricas_globales']['r2']:.4f}  "
          f"MAPE={eval_test['metricas_globales']['mape']:.2f}%")
    mejor_baseline_mae = min(b["mae"] for b in eval_test["baselines"].values())
    print(f"  Ganancia vs mejor baseline: {mejor_baseline_mae - eval_test['metricas_globales']['mae']:+,.0f} MAE")
    print(f"\n  Modelo exportado:    {SAVE_MODEL_DIR / f'{MODEL_NAME}.pkl'}")
    print(f"  Métricas exportadas: {metrics_path}")

    return {
        "models":          models,
        "best_params":     best_params,
        "X_test":          X_test,
        "y_test":          y_test,
        "y_val_pred":      y_val_pred,
        "y_test_pred":     y_test_pred,
        "features":        features,
        "eval_val":        eval_val,
        "eval_test":       eval_test,
        "shap_importance": shap_importance,
        "oportunidades_test": oportunidades_df,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(SEED)

    print(f"Versiones — lightgbm={lgb.__version__}  optuna={optuna.__version__}  "
          f"numpy={np.__version__}  pandas={pd.__version__}", flush=True)

    features = load_selected_features(FEATURES_PATH)
    df = load_dataset(DATASET_PATH, features)
    print(f"Dataset: {len(df)} filas")

    results = run_all(df, features)
