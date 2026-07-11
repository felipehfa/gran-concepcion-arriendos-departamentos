"""
Entrenamiento del modelo de PRODUCCIÓN.

No duplica la lógica de modelamiento: carga `04_modelamiento/01_xgboost.py`
como módulo (vía importlib, ya que su nombre empieza con dígito) y reutiliza
sus funciones (optimización de hiperparámetros, bagging, evaluación, SHAP).

Diferencias respecto al script de investigación (04_modelamiento/01_xgboost.py):
  - Split 85/15 (train/test) en vez de 70/15/15. El ensamble de bagging
    necesita igualmente un set para early stopping, así que se separa
    internamente un 10% del 85% de train solo para eso (~76.5% train_fit /
    ~8.5% early-stopping / 15% test) - es un detalle interno de
    entrenamiento, no una partición pública reportada.
  - Genera y persiste un identificador de VERSIÓN del modelo
    (ver `generar_version_modelo`), para poder distinguir después en la
    base de datos de producción qué predicciones se hicieron con qué
    versión cuando el modelo se reentrene.
  - Guarda el ensamble y sus parámetros en `versiones/{version}/` (no en
    `04_modelamiento/save/model/`, que es el modelo de investigación).
    Cada versión queda archivada en su propia carpeta (no se sobrescribe),
    para poder recuperar el modelo EXACTO usado en cualquier predicción
    pasada (`predicciones.version_modelo` -> `versiones/{esa versión}/`).

Dataset de entrada: por ahora, el mismo CSV curado que usa el modelo de
investigación (03_ingenieria_variables/save/.../datos_ingenieria_variables.csv
+ selected_features.csv). La incorporación de datos nuevos acumulados por el
pipeline de producción (Parte 2) queda para una futura re-corrida de este
mismo script, no para esta primera versión.
"""

import hashlib
import importlib.util
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# Carga de 04_modelamiento/01_xgboost.py como módulo (nombre empieza con
# dígito, no se puede hacer un `import` normal).
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
XGBOOST_MODULE_PATH = REPO_ROOT / "04_modelamiento" / "01_xgboost.py"


def _cargar_modulo_xgboost():
    spec = importlib.util.spec_from_file_location("modelo_investigacion_xgboost", XGBOOST_MODULE_PATH)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


mx = _cargar_modulo_xgboost()

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
SEED = 42

TEST_SIZE_PRODUCCION = 0.15
EARLY_STOP_FRACTION_OF_TRAIN = 0.10  # sobre el 85% de train, solo para early stopping del bagging

N_TRIALS_OPTUNA_PRODUCCION = mx.N_TRIALS_OPTUNA
CV_SPLITS_OPTUNA_PRODUCCION = mx.CV_SPLITS_OPTUNA
N_SEEDS_BAGGING_PRODUCCION = mx.N_SEEDS_BAGGING

SAVE_DIR = SCRIPT_DIR
CONTROL_VERSION_PATH = SAVE_DIR / "version_modelo.json"
# Cada versión se archiva en su propia carpeta (en vez de sobrescribir un
# único modelo_produccion.pkl/parametros_produccion.json) para poder
# recuperar el modelo EXACTO que se usó en cualquier predicción pasada
# (predicciones.version_modelo -> versiones/{esa versión}/).
VERSIONES_DIR = SAVE_DIR / "versiones"


# ------------------------------------------------------------------
# Versionado del modelo
# ------------------------------------------------------------------
def _hash_hiperparametros(best_params: dict) -> str:
    """sha256 (primeros 8 hex) de los hiperparámetros ganadores, en JSON
    canónico (claves ordenadas) para que el mismo dict siempre dé el mismo hash."""
    canon = json.dumps(best_params, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:8]


def generar_version_modelo(best_params: dict, control_path: Path = CONTROL_VERSION_PATH) -> tuple:
    """
    Genera el identificador de versión: v{contador:04d}_{YYYYMMDDHHMMSS}_{hash8}

    El contador es incremental y persiste en `control_path` junto con un
    historial de versiones anteriores. Se genera una versión nueva en cada
    corrida de este script, sin importar si los hiperparámetros cambiaron.
    """
    if control_path.exists():
        control = json.loads(control_path.read_text(encoding="utf-8"))
    else:
        control = {"contador": 0, "version_actual": None, "historial": []}

    contador = control["contador"] + 1
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    hash8 = _hash_hiperparametros(best_params)
    version = f"v{contador:04d}_{timestamp}_{hash8}"

    control["contador"] = contador
    control["version_actual"] = version
    control["historial"].append({
        "version": version,
        "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hash_hiperparametros": hash8,
    })
    return version, control


def guardar_control_version(control: dict, control_path: Path = CONTROL_VERSION_PATH) -> None:
    control_path.write_text(json.dumps(control, indent=2, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------
# Calibración de oportunidad/confianza — para poder etiquetar avisos NUEVOS
# en producción (Etapa 7) de forma consistente con el test set, en vez de
# recalcular deciles/terciles de una sola fila (imposible: qcut necesita una
# distribución). A diferencia de `etiquetar_oportunidades` en
# 04_modelamiento/01_xgboost.py (que aplica los bordes calculados a ESE
# mismo test set y los descarta), acá se GUARDAN los bordes para reaplicarlos
# después con `pd.cut` sobre avisos que ni siquiera existían al entrenar.
# ------------------------------------------------------------------
def calcular_calibracion_oportunidad(
    models: list, X_test: pd.DataFrame, y_test: pd.Series,
    n_deciles: int = None, n_grupos_confianza: int = None,
) -> dict:
    n_deciles = n_deciles or mx.N_DECILES_OPORTUNIDAD
    n_grupos_confianza = n_grupos_confianza or mx.N_GRUPOS_CONFIANZA

    y_true = np.asarray(y_test, dtype=float)
    pred_matrix = mx.predict_ensemble_matrix(models, X_test)
    y_pred = pred_matrix.mean(axis=0)
    error = y_true - y_pred

    # Bordes de deciles de precio_clp (test), extendidos a ±inf en los
    # extremos para que un aviso nuevo con precio fuera del rango visto en
    # test igual caiga en el decil extremo correspondiente, en vez de NaN.
    _, bordes_deciles = pd.qcut(y_true, q=n_deciles, retbins=True, duplicates="drop")
    bordes_deciles[0], bordes_deciles[-1] = -np.inf, np.inf
    deciles = pd.cut(y_true, bins=bordes_deciles, labels=False, include_lowest=True)

    stats_por_decil = {}
    for d in sorted(pd.unique(deciles)):
        mask = deciles == d
        mediana_d, mad_d = mx.mediana_y_mad(error[mask])
        stats_por_decil[int(d)] = {
            "n": int(mask.sum()), "mediana_error": mediana_d, "mad_error": mad_d,
        }

    # Terciles del coeficiente de variación del ensamble (test), mismos
    # extremos ±inf por la misma razón.
    pred_std = pred_matrix.std(axis=0)
    cv_ensamble = pred_std / np.where(y_pred == 0, 1e-6, y_pred)
    _, bordes_cv = pd.qcut(cv_ensamble, q=n_grupos_confianza, retbins=True, duplicates="drop")
    bordes_cv[0], bordes_cv[-1] = -np.inf, np.inf

    return {
        "bordes_deciles_precio": bordes_deciles.tolist(),
        "stats_por_decil": stats_por_decil,
        "bordes_cv_confianza": bordes_cv.tolist(),
        "etiquetas_confianza": mx.ETIQUETAS_CONFIANZA,
        "mad_scale_const": mx.MAD_SCALE_CONST,
        "umbral_oportunidad": mx.UMBRAL_OPORTUNIDAD,
        "umbral_caro": mx.UMBRAL_CARO,
    }


# ------------------------------------------------------------------
# Split 85/15 + carve interno para early stopping
# ------------------------------------------------------------------
def split_produccion(df, features, target_col=None, seed: int = SEED):
    target_col = target_col or mx.TARGET_COL

    estratos = mx.construir_estratos_precio(df[target_col], mx.N_ESTRATOS_PRECIO)
    train_85, test_15 = train_test_split(
        df, test_size=TEST_SIZE_PRODUCCION, random_state=seed, shuffle=True, stratify=estratos,
    )

    estratos_train = mx.construir_estratos_precio(train_85[target_col], mx.N_ESTRATOS_PRECIO)
    train_fit, train_earlystop = train_test_split(
        train_85, test_size=EARLY_STOP_FRACTION_OF_TRAIN, random_state=seed, shuffle=True,
        stratify=estratos_train,
    )

    print("\nSplit de producción (85/15, con carve interno de early-stopping)")
    print(f"  Train (fit):          {len(train_fit)} filas ({len(train_fit)/len(df)*100:.1f}%)")
    print(f"  Train (early-stop):   {len(train_earlystop)} filas ({len(train_earlystop)/len(df)*100:.1f}%)")
    print(f"  Test (holdout final): {len(test_15)} filas ({len(test_15)/len(df)*100:.1f}%)")

    return train_fit, train_earlystop, test_15


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    np.random.seed(SEED)

    print(f"Versiones — xgboost={mx.xgb.__version__}  optuna={mx.optuna.__version__}  "
          f"numpy={np.__version__}")

    features = mx.load_selected_features(mx.FEATURES_PATH)
    df = mx.load_dataset(mx.DATASET_PATH, features)
    print(f"Dataset: {len(df)} filas")

    train_fit, train_earlystop, test_15 = split_produccion(df, features)

    resultado = mx.entrenar_y_evaluar_modelo(
        train_fit, train_earlystop, test_15, features,
        seed=SEED,
        n_trials=N_TRIALS_OPTUNA_PRODUCCION,
        cv_splits=CV_SPLITS_OPTUNA_PRODUCCION,
        n_seeds_bagging=N_SEEDS_BAGGING_PRODUCCION,
        persistir_modelo=False,
    )

    models = resultado["models"]
    best_params = resultado["best_params"]

    print(f"\n{'='*60}")
    print("SHAP nativo XGBoost — Feature Importance (Test)")
    print("="*60)
    shap_importance = mx.compute_shap_native(models, resultado["X_test"])
    print(mx.pd.Series(shap_importance).head(15).to_string())

    print(f"\n{'='*60}")
    print("CALIBRACIÓN DE OPORTUNIDAD/CONFIANZA (Test)")
    print("="*60)
    calibracion = calcular_calibracion_oportunidad(models, resultado["X_test"], resultado["y_test"])
    print(f"  Bordes de deciles de precio: {[round(b, 0) for b in calibracion['bordes_deciles_precio']]}")
    print(f"  Bordes de terciles CV:       {[round(b, 4) for b in calibracion['bordes_cv_confianza']]}")

    version, control = generar_version_modelo(best_params)
    print(f"\nVersión de modelo generada: {version}")

    version_dir = VERSIONES_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)

    modelo_path = version_dir / "modelo_produccion.pkl"
    with open(modelo_path, "wb") as f:
        pickle.dump(models, f)

    params_data = {
        "version_modelo": version,
        "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": features,
        "target_col": mx.TARGET_COL,
        "hiperparametros": best_params,
        "optimizacion": resultado["optim_info"],
        "n_seeds_bagging": N_SEEDS_BAGGING_PRODUCCION,
        "tiempo_bagging_seg": resultado["tiempo_bagging_seg"],
        "split": {
            "test_size": TEST_SIZE_PRODUCCION,
            "early_stop_fraction_of_train": EARLY_STOP_FRACTION_OF_TRAIN,
            "seed": SEED,
            "n_train_fit": len(train_fit),
            "n_train_earlystop": len(train_earlystop),
            "n_test": len(test_15),
        },
        "evaluacion_subset_early_stopping": resultado["eval_val"],
        "evaluacion_test": resultado["eval_test"],
        "shap_importance": {k: float(v) for k, v in shap_importance.items()},
        "calibracion_oportunidad": calibracion,
        "dataset_origen": str(mx.DATASET_PATH),
        "features_origen": str(mx.FEATURES_PATH),
    }
    params_path = version_dir / "parametros_produccion.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params_data, f, indent=4, ensure_ascii=False, cls=mx.NumpyEncoder)

    guardar_control_version(control)

    print(f"\n{'='*60}")
    print("RESUMEN FINAL")
    print("="*60)
    eval_test = resultado["eval_test"]
    print(f"  Versión:  {version}")
    print(f"  TEST → MAE={eval_test['metricas_globales']['mae']:,.0f}  "
          f"RMSE={eval_test['metricas_globales']['rmse']:,.0f}  "
          f"R²={eval_test['metricas_globales']['r2']:.4f}  "
          f"MAPE={eval_test['metricas_globales']['mape']:.2f}%")
    print(f"\n  Modelo guardado:      {modelo_path}")
    print(f"  Parámetros guardados: {params_path}")
    print(f"  Control de versión:   {CONTROL_VERSION_PATH}")


if __name__ == "__main__":
    main()
