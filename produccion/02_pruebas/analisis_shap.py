"""
ANÁLISIS SHAP DEL MODELO DE PRODUCCIÓN VIGENTE — NO ES PARTE DEL PIPELINE.

Responde "qué variables determinan el costo total predicho, y en qué
dirección" para el ensamble de bagging que está sirviendo predicciones hoy
(`entrenamiento/versiones/{version_actual}/modelo_produccion.pkl`), sobre su
propio set de test (85/15, seed=42 — igual que en 01_entrenar_modelo_produccion.py).

`parametros_produccion.json` ya trae un `shap_importance` (magnitud promedio,
|contribución| media en CLP por feature — ver `compute_shap_native` en
`04_modelamiento/02_lightgbm.py`), calculado con el mismo TreeSHAP nativo de
LightGBM (`Booster.predict(..., pred_contrib=True)`, exacto para árboles, sin
depender de la librería externa `shap`). Este script reutiliza esa misma
técnica pero SIN colapsar a valor absoluto, para poder estimar además la
DIRECCIÓN de cada feature (sube o baja el costo total predicho) vía la
correlación fila a fila entre el valor crudo de la feature y su propia
contribución SHAP — un diagnóstico simple que la magnitud sola no da.

Reutiliza (vía importlib, mismo patrón que prototipo_prediccion_manual.py):
  - `05_prediccion.py::cargar_modelo_y_calibracion()` para el ensamble vigente
    y sus features, sin duplicar la lógica de selección de versión/algoritmo.
  - `entrenamiento/01_entrenar_modelo_produccion.py::split_produccion()` para
    reconstruir el mismo test set (85/15) que se usó al entrenar esa versión,
    sin persistir el propio test set en disco en ningún lado del pipeline.

Se probó además usar la librería externa `shap` (TreeExplainer) en vez de este
cálculo nativo: sobre este mismo ensamble y test set, sus valores dan
EXACTAMENTE los mismos números que `contribuciones_shap` de abajo (diferencia
absoluta máxima = 0.0), así que no cambia ningún resultado — pero en esta
máquina `import shap` crashea con `numpy==2.4.6` (la versión pinneada del
proyecto; incompatibilidad de ABI nativa entre el wheel de `shap==0.51.0` y
esa build de numpy, no arreglable reinstalando ni con la última versión
disponible), y su parte gráfica (matplotlib) además choca con una política de
Control de Aplicaciones de Windows en este equipo. Por eso `graficar_beeswarm`
de abajo reimplementa un beeswarm plot con matplotlib puro (que sí funciona
en este entorno) directamente sobre `shap_matrix`, sin depender de `shap`.

Salidas (no versionadas, se regeneran corriendo el script):
  - save/analisis_shap/shap_importancia.csv   (tabla completa, 29 features)
  - save/analisis_shap/shap_importancia.png   (gráfico de barras, copia local)
  - save/analisis_shap/shap_beeswarm.png      (beeswarm, copia local)
  - docs/images/shap_importancia.png          (gráfico de barras, para README/Streamlit)
  - docs/images/shap_beeswarm.png             (beeswarm, para README/Streamlit)

CÓMO CORRERLO:
    python 02_pruebas/analisis_shap.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PRODUCCION_ROOT = SCRIPT_DIR.parent
MODELO_PRODUCCION_DIR = PRODUCCION_ROOT / "01_modelo_produccion"
REPO_ROOT = PRODUCCION_ROOT.parent

SAVE_DIR = SCRIPT_DIR / "save" / "analisis_shap"
DOCS_IMAGES_DIR = REPO_ROOT / "docs" / "images"

# `05_prediccion.py` usa `import db` normal (no importlib) - solo resuelve si
# el directorio está en sys.path (mismo requisito que prototipo_prediccion_manual.py).
sys.path.insert(0, str(MODELO_PRODUCCION_DIR))


def _cargar_modulo(nombre: str, ruta: Path):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


pred = _cargar_modulo("prediccion_produccion", MODELO_PRODUCCION_DIR / "05_prediccion.py")
entrenar = _cargar_modulo(
    "entrenamiento_produccion", MODELO_PRODUCCION_DIR / "entrenamiento" / "01_entrenar_modelo_produccion.py",
)
mx = entrenar.mx  # script de investigación (02_lightgbm.py) del algoritmo ganador

# Umbral de |correlación| bajo el cual se considera que una feature no tiene
# una dirección lineal clara sobre el costo total predicho (relación mixta o
# dominada por interacciones con otras features, típico en árboles).
UMBRAL_DIRECCION = 0.15

ETIQUETA_SUBE = "sube el costo total"
ETIQUETA_BAJA = "baja el costo total"
ETIQUETA_MIXTA = "mixta / no lineal"


def reconstruir_test_set(features: list) -> tuple:
    """
    Mismo split 85/15 (seed=42, estratificado por quintil de costo total) que
    generó el modelo vigente — ver `split_produccion` en
    `01_entrenar_modelo_produccion.py`. `X_test`/`y_test` se construyen igual
    que dentro de `entrenar_y_evaluar_modelo` (`02_lightgbm.py`): `fillna(0)`
    sobre las features, target como float.
    """
    df = mx.load_dataset(mx.DATASET_PATH, features)
    _, _, test_15 = entrenar.split_produccion(df, features)
    X_test = test_15[features].fillna(0)
    y_test = test_15[mx.TARGET_COL].astype(float)
    return X_test, y_test


def contribuciones_shap(models: list, X: pd.DataFrame) -> np.ndarray:
    """
    Matriz de contribuciones SHAP (TreeSHAP nativo LightGBM, exacto) por fila
    y feature, promediada entre los modelos del ensamble de bagging — misma
    técnica que `compute_shap_native` en `02_lightgbm.py`, pero sin colapsar a
    |valor| todavía, para poder analizar dirección además de magnitud.
    """
    contribs_sum = None
    for model in models:
        contribs = model.predict(X, num_iteration=model.best_iteration, pred_contrib=True)
        contribs = contribs[:, :-1]  # descartar columna de bias
        contribs_sum = contribs if contribs_sum is None else contribs_sum + contribs
    return contribs_sum / len(models)


def resumen_shap(features: list, X_test: pd.DataFrame, shap_matrix: np.ndarray) -> pd.DataFrame:
    mean_abs = np.abs(shap_matrix).mean(axis=0)

    # Dirección: correlación de Pearson, fila a fila, entre el valor crudo de
    # la feature y su propia contribución SHAP. Es una lectura descriptiva
    # simple (no captura interacciones ni relaciones no lineales fuertes),
    # pero para features mayormente monótonas (superficie, baños, distancias,
    # dummies 0/1) resume bien el sentido del efecto.
    direccion_corr = np.array([
        np.corrcoef(X_test.iloc[:, i], shap_matrix[:, i])[0, 1] if X_test.iloc[:, i].std() > 0 else np.nan
        for i in range(len(features))
    ])

    resumen = pd.DataFrame({
        "feature": features,
        "shap_importancia_clp": mean_abs,
        "direccion_corr": direccion_corr,
    }).sort_values("shap_importancia_clp", ascending=False).reset_index(drop=True)

    resumen["share_pct"] = resumen["shap_importancia_clp"] / resumen["shap_importancia_clp"].sum() * 100
    resumen["direccion"] = np.select(
        [resumen["direccion_corr"] > UMBRAL_DIRECCION, resumen["direccion_corr"] < -UMBRAL_DIRECCION],
        [ETIQUETA_SUBE, ETIQUETA_BAJA],
        default=ETIQUETA_MIXTA,
    )
    return resumen


def graficar(resumen: pd.DataFrame, version: str, n_test: int, top_n: int = 15) -> None:
    import matplotlib.pyplot as plt

    top = resumen.head(top_n).iloc[::-1]
    color_por_direccion = {ETIQUETA_SUBE: "#2E7D32", ETIQUETA_BAJA: "#C62828", ETIQUETA_MIXTA: "#9E9E9E"}
    colores = [color_por_direccion[d] for d in top["direccion"]]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(top["feature"], top["shap_importancia_clp"], color=colores)
    ax.set_xlabel("Importancia SHAP promedio |contribución| (CLP/mes)")
    ax.set_title(f"Variables más determinantes del costo total mensual\nSHAP · modelo {version} · test n={n_test}")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    from matplotlib.patches import Patch
    leyenda = [
        Patch(color=color_por_direccion[ETIQUETA_SUBE], label=f"↑ {ETIQUETA_SUBE}"),
        Patch(color=color_por_direccion[ETIQUETA_BAJA], label=f"↓ {ETIQUETA_BAJA}"),
        Patch(color=color_por_direccion[ETIQUETA_MIXTA], label=ETIQUETA_MIXTA),
    ]
    ax.legend(handles=leyenda, loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_DIR / "shap_importancia.png", dpi=150)
    fig.savefig(DOCS_IMAGES_DIR / "shap_importancia.png", dpi=150)
    plt.close(fig)


def _densidad_gaussiana(valores: np.ndarray) -> np.ndarray:
    """
    Densidad estimada (kernel gaussiano, regla de Silverman para el ancho de
    banda) en cada uno de los propios `valores` — equivalente a
    `scipy.stats.gaussian_kde(valores)(valores)`, implementado a mano en numpy
    puro para no depender de `scipy.linalg.cholesky` (ver nota en
    `graficar_beeswarm`). O(n²), trivial para los ~245 avisos de test.
    """
    n = len(valores)
    std = valores.std()
    ancho_banda = 1.06 * std * n ** (-1 / 5)
    diffs = (valores[:, None] - valores[None, :]) / ancho_banda
    return np.exp(-0.5 * diffs ** 2).sum(axis=1) / (n * ancho_banda * np.sqrt(2 * np.pi))


def graficar_beeswarm(
    features: list, X_test: pd.DataFrame, shap_matrix: np.ndarray, version: str, n_test: int, top_n: int = 15,
) -> None:
    """
    Beeswarm plot (un punto por aviso de test, por feature) reimplementado con
    matplotlib puro — ver nota del módulo sobre por qué no se usa la librería
    `shap` para esto. Cada punto es la contribución SHAP de esa feature para
    ese aviso (eje X) y su color es el valor crudo de la feature en ese aviso
    (azul = bajo, rojo = alto, normalizado por percentiles 5-95 para que un
    outlier no aplaste la escala de color) — permite leer de un vistazo si
    "valor alto de la feature" empuja el costo hacia arriba o hacia abajo, y
    qué tan dispersa es esa relación (a diferencia del gráfico de barras, que
    solo muestra el promedio).

    El jitter vertical dentro de cada fila es proporcional a la densidad local
    de puntos en el eje X (kernel gaussiano, implementado a mano — ver
    `_densidad_gaussiana` abajo, en vez de `scipy.stats.gaussian_kde`: en esta
    máquina `scipy.linalg.cholesky`, que `gaussian_kde` usa internamente,
    crashea con la misma incompatibilidad nativa de numpy==2.4.6 que afecta a
    `shap`, sección de arriba), para simular el efecto de "enjambre" sin
    implementar el algoritmo de packing exacto que usa `shap`.
    """
    import matplotlib.pyplot as plt

    mean_abs = np.abs(shap_matrix).mean(axis=0)
    orden = np.argsort(-mean_abs)[:top_n][::-1]  # más importante arriba

    cmap = plt.get_cmap("coolwarm")
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(orden) + 2))
    for fila, feat_i in enumerate(orden):
        valores_shap = shap_matrix[:, feat_i]
        valores_crudos = X_test.iloc[:, feat_i].to_numpy(dtype=float)

        p5, p95 = np.percentile(valores_crudos, [5, 95])
        color_val = np.full_like(valores_crudos, 0.5) if p95 <= p5 else np.clip(
            (valores_crudos - p5) / (p95 - p5), 0, 1,
        )

        if valores_shap.std() > 0:
            densidad = _densidad_gaussiana(valores_shap)
            densidad = densidad / densidad.max()
        else:
            densidad = np.zeros_like(valores_shap)
        jitter = (rng.random(len(valores_shap)) - 0.5) * densidad * 0.85

        ax.scatter(
            valores_shap, fila + jitter, c=color_val, cmap=cmap, vmin=0, vmax=1,
            s=13, alpha=0.75, edgecolors="none", zorder=2,
        )

    ax.axvline(0, color="#B0B0B0", linewidth=0.8, zorder=1)
    ax.set_yticks(range(len(orden)))
    ax.set_yticklabels([features[i] for i in orden])
    ax.set_ylim(-0.6, len(orden) - 0.4)
    ax.set_xlabel("Contribución SHAP por aviso (CLP/mes)")
    ax.set_title(f"Impacto de cada variable, aviso por aviso\nSHAP · modelo {version} · test n={n_test}")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.012, fraction=0.035)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["valor bajo", "valor alto"])
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Valor de la variable en ese aviso", fontsize=9)

    fig.tight_layout()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_DIR / "shap_beeswarm.png", dpi=150)
    fig.savefig(DOCS_IMAGES_DIR / "shap_beeswarm.png", dpi=150)
    plt.close(fig)


def main() -> pd.DataFrame:
    modelo_info = pred.cargar_modelo_y_calibracion()
    features = modelo_info["features"]
    models = modelo_info["models"]
    version = modelo_info["version_modelo"]

    X_test, y_test = reconstruir_test_set(features)
    shap_matrix = contribuciones_shap(models, X_test)
    resumen = resumen_shap(features, X_test, shap_matrix)

    print(f"Modelo vigente: {version}  (algoritmo={modelo_info['algoritmo']}, n_test={len(X_test)})")
    print(f"\nTop 15 features por importancia SHAP (magnitud + dirección):\n")
    print(resumen.head(15).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SAVE_DIR / "shap_importancia.csv"
    resumen.to_csv(csv_path, index=False)
    print(f"\nTabla completa (29 features) guardada en {csv_path}")

    graficar(resumen, version, len(X_test))
    print(f"Gráfico de barras guardado en {DOCS_IMAGES_DIR / 'shap_importancia.png'}")

    graficar_beeswarm(features, X_test, shap_matrix, version, len(X_test))
    print(f"Beeswarm guardado en {DOCS_IMAGES_DIR / 'shap_beeswarm.png'}")

    return resumen


if __name__ == "__main__":
    main()
