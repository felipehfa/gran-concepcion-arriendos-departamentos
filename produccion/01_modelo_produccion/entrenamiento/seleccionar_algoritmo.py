"""
Selecciona qué algoritmo de investigación usar para entrenar el modelo de
PRODUCCIÓN, comparando los JSON de métricas más recientes de cada corrida de
investigación (04_modelamiento/save/model/*_metrics.json).

No reentrena nada: solo lee los resultados de test ya reportados por
04_modelamiento/01_xgboost.py y 04_modelamiento/02_lightgbm.py, y deja
registrada la decisión en `algoritmo_seleccionado.json` (junto a
`version_modelo.json`), para que 01_entrenar_modelo_produccion.py sepa qué
script de investigación cargar sin tener que comparar nada él mismo.

Criterio por defecto: "ponderado" (50% MAE test + 50% RMSE test). Como MAE y
RMSE viven en escalas distintas, cada métrica se normaliza dividiendo por el
promedio de ambos algoritmos para esa métrica (mae/mean_mae, rmse/mean_rmse)
antes de ponderar — así un 50/50 realmente pesa igual, en vez de que RMSE
(que siempre es mayor en magnitud) domine el puntaje. Gana el algoritmo con
puntaje ponderado MÁS BAJO.

También soporta "mae_test" o "rmse_test" a secas (una sola métrica, sin
ponderar), vía --criterio.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
MODEL_METRICS_DIR = REPO_ROOT / "04_modelamiento" / "save" / "model"

METRICS_PATHS = {
    "xgboost": MODEL_METRICS_DIR / "xgboost_regression_precio_metrics.json",
    "lightgbm": MODEL_METRICS_DIR / "lightgbm_regression_precio_metrics.json",
}

SALIDA_PATH = SCRIPT_DIR / "algoritmo_seleccionado.json"

CRITERIOS_VALIDOS = ("ponderado", "mae_test", "rmse_test")


def _cargar_metrics(algoritmo: str) -> dict:
    path = METRICS_PATHS[algoritmo]
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el JSON de métricas de investigación para "
            f"'{algoritmo}': {path}. Corre primero "
            f"04_modelamiento/{'01_xgboost.py' if algoritmo == 'xgboost' else '02_lightgbm.py'}."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _validar_comparabilidad(metrics_a: dict, metrics_b: dict, nombre_a: str, nombre_b: str) -> None:
    """Evita comparar corridas de investigación desalineadas (ej. una con 20
    features y otra con 30, o splits con distinta seed)."""
    feats_a, feats_b = set(metrics_a["features"]), set(metrics_b["features"])
    if feats_a != feats_b:
        solo_a = feats_a - feats_b
        solo_b = feats_b - feats_a
        raise ValueError(
            f"Las features de '{nombre_a}' y '{nombre_b}' no coinciden — no son "
            f"comparables. Solo en {nombre_a}: {sorted(solo_a) or '-'}. "
            f"Solo en {nombre_b}: {sorted(solo_b) or '-'}. Reentrena ambos con "
            f"el mismo selected_features.csv antes de comparar."
        )

    seed_a, seed_b = metrics_a["split"]["seed"], metrics_b["split"]["seed"]
    if seed_a != seed_b:
        raise ValueError(
            f"Los splits de '{nombre_a}' (seed={seed_a}) y '{nombre_b}' "
            f"(seed={seed_b}) usan semillas distintas — no son comparables."
        )

    n_test_a, n_test_b = metrics_a["split"]["n_test"], metrics_b["split"]["n_test"]
    if n_test_a != n_test_b:
        raise ValueError(
            f"Los test sets de '{nombre_a}' (n={n_test_a}) y '{nombre_b}' "
            f"(n={n_test_b}) tienen tamaños distintos — no son comparables."
        )


def seleccionar_algoritmo(
    criterio: str = "ponderado",
    peso_mae: float = 0.5,
    peso_rmse: float = 0.5,
    guardar: bool = True,
) -> dict:
    """
    Compara xgboost vs lightgbm según `criterio` y devuelve el resultado
    (mismo dict que se persiste en algoritmo_seleccionado.json si guardar=True).
    """
    if criterio not in CRITERIOS_VALIDOS:
        raise ValueError(f"criterio debe ser uno de {CRITERIOS_VALIDOS}, recibido: {criterio!r}")
    if criterio == "ponderado" and abs((peso_mae + peso_rmse) - 1.0) > 1e-9:
        raise ValueError(f"peso_mae + peso_rmse debe sumar 1.0 (recibido {peso_mae} + {peso_rmse})")

    metrics = {algo: _cargar_metrics(algo) for algo in METRICS_PATHS}
    _validar_comparabilidad(metrics["xgboost"], metrics["lightgbm"], "xgboost", "lightgbm")

    globales = {algo: m["evaluacion_test"]["metricas_globales"] for algo, m in metrics.items()}
    residuos = {algo: m["evaluacion_test"]["residuos"] for algo, m in metrics.items()}
    quintiles = {algo: m["evaluacion_test"]["mae_por_quintil_precio"] for algo, m in metrics.items()}

    mean_mae = sum(g["mae"] for g in globales.values()) / 2
    mean_rmse = sum(g["rmse"] for g in globales.values()) / 2

    scores = {}
    for algo, g in globales.items():
        if criterio == "mae_test":
            scores[algo] = g["mae"]
        elif criterio == "rmse_test":
            scores[algo] = g["rmse"]
        else:  # ponderado
            scores[algo] = peso_mae * (g["mae"] / mean_mae) + peso_rmse * (g["rmse"] / mean_rmse)

    ganador, perdedor = sorted(scores, key=lambda a: scores[a])
    margen_absoluto = scores[perdedor] - scores[ganador]
    margen_relativo_pct = (margen_absoluto / scores[perdedor]) * 100 if scores[perdedor] else 0.0

    # Advertencia si el ganador global no es el mejor en el quintil más caro
    # (Q5) — patrón conocido: XGBoost suele ganar ahí aunque pierda en global.
    ultimo_quintil = str(max(int(k) for k in quintiles["xgboost"]))
    mae_q5 = {algo: quintiles[algo][ultimo_quintil]["mae"] for algo in quintiles}
    ganador_q5 = min(mae_q5, key=lambda a: mae_q5[a])
    advertencia_q5 = None
    if ganador_q5 != ganador:
        advertencia_q5 = (
            f"'{ganador_q5}' tiene mejor MAE en el quintil más caro (Q5: "
            f"{mae_q5[ganador_q5]:,.2f} vs {mae_q5[ganador]:,.2f}), pero el "
            f"ganador global por '{criterio}' sigue siendo '{ganador}'."
        )

    resultado = {
        "algoritmo": ganador,
        "criterio": criterio,
        "pesos": {"mae": peso_mae, "rmse": peso_rmse} if criterio == "ponderado" else None,
        "fecha_seleccion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candidatos": {
            algo: {
                "mae_test": globales[algo]["mae"],
                "rmse_test": globales[algo]["rmse"],
                "r2_test": globales[algo]["r2"],
                "mape_test": globales[algo]["mape"],
                "mdape_test": globales[algo]["mdape"],
                "skewness_residuos_test": residuos[algo]["skewness"],
                "kurtosis_residuos_test": residuos[algo]["kurtosis"],
                "mae_q5_test": mae_q5[algo],
                "score": scores[algo],
                "fecha_entrenamiento_investigacion": metrics[algo]["fecha_entrenamiento"],
                "metrics_path": str(METRICS_PATHS[algo]),
            }
            for algo in metrics
        },
        "margen_absoluto": margen_absoluto,
        "margen_relativo_pct": margen_relativo_pct,
        "advertencia_q5": advertencia_q5,
        "n_features": len(metrics["xgboost"]["features"]),
        "seed_split": metrics["xgboost"]["split"]["seed"],
    }

    if guardar:
        SALIDA_PATH.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    return resultado


def _imprimir_reporte(resultado: dict, guardado: bool) -> None:
    cands = resultado["candidatos"]
    print("=" * 70)
    print("COMPARACIÓN xgboost vs lightgbm (evaluación TEST, investigación)")
    print("=" * 70)
    header = f"{'métrica':<26}{'xgboost':>18}{'lightgbm':>18}"
    print(header)
    print("-" * len(header))
    filas = [
        ("MAE", "mae_test", ",.2f"),
        ("RMSE", "rmse_test", ",.2f"),
        ("R²", "r2_test", ".4f"),
        ("MAPE (%)", "mape_test", ".4f"),
        ("MdAPE (%)", "mdape_test", ".4f"),
        ("Skewness residuos", "skewness_residuos_test", ".4f"),
        ("Kurtosis residuos", "kurtosis_residuos_test", ".4f"),
        ("MAE Q5 (más caro)", "mae_q5_test", ",.2f"),
        ("Score ('%s')" % resultado["criterio"], "score", ".6f"),
    ]
    for etiqueta, campo, fmt in filas:
        val_x = format(cands["xgboost"][campo], fmt)
        val_l = format(cands["lightgbm"][campo], fmt)
        print(f"{etiqueta:<26}{val_x:>18}{val_l:>18}")

    print("-" * len(header))
    print(f"\nGanador: {resultado['algoritmo'].upper()}  "
          f"(margen {resultado['margen_absoluto']:.6f}, "
          f"{resultado['margen_relativo_pct']:.2f}% relativo, criterio={resultado['criterio']})")
    if resultado["advertencia_q5"]:
        print(f"\nAdvertencia: {resultado['advertencia_q5']}")
    if guardado:
        print(f"\nDecisión guardada en: {SALIDA_PATH}")
    else:
        print("\n(--no-guardar: decisión no persistida en disco)")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--criterio", choices=CRITERIOS_VALIDOS, default="ponderado")
    parser.add_argument("--peso-mae", type=float, default=0.5)
    parser.add_argument("--peso-rmse", type=float, default=0.5)
    parser.add_argument("--no-guardar", action="store_true", help="No escribir algoritmo_seleccionado.json")
    return parser.parse_args()


def main():
    args = _parse_args()
    resultado = seleccionar_algoritmo(
        criterio=args.criterio,
        peso_mae=args.peso_mae,
        peso_rmse=args.peso_rmse,
        guardar=not args.no_guardar,
    )
    _imprimir_reporte(resultado, guardado=not args.no_guardar)


if __name__ == "__main__":
    main()
