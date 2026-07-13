# Changelog

Historial de cambios operacionales de
[gran-concepcion-rentals](README.md) que no encajan como arquitectura permanente en el README —
eventos puntuales de una corrida o migración específica, con su contexto y resultado. El estado
ya corregido del código está descrito en el [README](README.md); acá queda el detalle de cómo se
llegó a ese estado.

---

## 2026-07-12 — Sesión de mantenimiento: histórico→producción, fix de `gastos_comunes`, reentrenamiento

Sesión enfocada en tres frentes: poblar la base de producción con el histórico de investigación,
corregir un bug de escala en `gastos_comunes` (extracción → base → modelo → producción), y
ajustar el ritmo del re-chequeo.

### Producción: dejar de scrapear casas

`01_scraper_grilla_incremental.py` ahora usa `TIPOS_PROPIEDAD_PRODUCCION = ["departamento"]` como
default (configurable, no hardcodeado) en vez de `sg.TIPOS_PROPIEDAD` (`["casa",
"departamento"]`) — evita gastar presupuesto de scraping en avisos que el resto del pipeline de
producción descartaría de todas formas (el modelo solo predice departamentos, ver
[README, sección 4](README.md#4-features-del-modelo-final-32)).
`02_scraper_detalle_incremental.py` no necesitó cambios (no filtra por tipo, trabaja directo
sobre lo que ya está en `avisos`).

### Migración del histórico de investigación a producción

Nuevo script de una sola vez, `05_modelo_produccion/migracion_historico_a_produccion.py` (no
forma parte del orquestador). Migró **1628 avisos** (`departamento`/`arriendo`) desde
`avisos_gran_concepcion.db` — exactamente los IDs de `datos_ingenieria_variables.csv` (el dataset
que entrenó el modelo), no los ~1956 avisos crudos completos, ya que las ~298 casas del histórico
nunca generarían features ni predicción en producción. Quedan con `estado_publicacion='activo'` y
`fecha_ultimo_chequeo_estado=NULL` (califican de inmediato para el primer re-chequeo real).
Colisiones: se conserva siempre la fila ya existente en producción, nunca se sobreescribe con la
versión del histórico. Backup previo
(`produccion_gran_concepcion.db.bak-pre-migracion-historico`), idempotente.

**Incidente: pérdida silenciosa de filas (Google Drive + SQLite)** — la primera corrida reportó
1628 migrados pero la base terminó con solo 1606 avisos nuevos: 25 filas se perdieron de forma
silenciosa (sin excepción, `PRAGMA integrity_check` seguía en `ok`). Causa más probable: el repo
vive dentro de una carpeta sincronizada por Google Drive, y la migración hizo ~1628 commits
SQLite individuales a lo largo de ~3 minutos — ventana larga donde el sincronizador pudo
interferir con el archivo. Al ser el script idempotente, una segunda corrida completó exactamente
las 25 filas faltantes sin duplicar nada. **Recomendación para scripts futuros** que hagan muchas
escrituras seguidas a estas bases: verificar conteos después de corridas largas, dado que el repo
vive en una carpeta sincronizada.

### Backfill de `gastos_comunes`: reclasificación fila por fila

Contexto y lógica de la corrección en
[README, sección 3.3](README.md#33-el-bug-de-gastos_comunes-impacto-medible-en-el-modelo). Este
es el detalle operacional del backfill: sobre `avisos_gran_concepcion.db` (backup
`.bak-pre-backfill-gastos-comunes`), reclasificando las 1956 filas de `avisos_detalle` según el
texto crudo:

| Categoría | Filas | Acción |
|---|---:|---|
| Bug de escala corregido (`"N.NNN"` → valor real) | 1411 | `×1000` efectivo |
| Placeholder "incluido" → `0` | 45 | incluye 3 casos fuera del rango 1-14 que la inspección manual no había detectado (`"90"`, `"60"`, `"10"`) |
| Ya era `0` explícito | 392 | sin cambios |
| NULL sin resolver (ausencia genuina) | 88 | se dejan `NULL`, no se fuerzan a `0` |
| No parseable (basura de extracción: `","`, `"."`) | 18 | sin cambios, fuera de alcance |
| Outlier sin punto ($35.000, `MLC-2055388319`) | 1 | sin cambios, señalado |
| Outlier implausible (\$1.111.111, `MLC-4068110790`) | 1 | descartado a `NULL` (supera el techo de sanidad de \$500.000) |

De los 190 `NULL` originales, **102 (54%) se resolvieron** vía el fallback de extracción
(insignia de resumen "Gastos comunes desde $X"), verificado en vivo contra el sitio real.

`produccion_gran_concepcion.db` se corrigió por separado con el mismo criterio: los 1628 avisos
migrados ya venían limpios (heredan el texto ya corregido del histórico), y los 3 avisos
preexistentes (scrapeados antes del fix del scraper) se corrigieron directo, respaldados en una
verificación en vivo contra el HTML real.

### Reentrenamiento con datos corregidos

Dataset regenerado desde cero (`01_ingenieria_variables.py` sobre la base ya corregida, no un
parche del CSV): **1628 filas × 44 columnas** (igual tamaño que antes — la corrección de escala
no elimina filas, solo corrige valores). Selección de variables re-corrida: **32 features
finales** (antes 30) — nuevas: `c_ig_com`, `cantidad_clinicas`, `cantidad_centros_comerciales`;
sacada: `cantidad_universidades`. `gastos_comunes` se mantuvo entre las más estables
(`stability_score=0.97`, top 6).

Métricas de test, antes/después de la corrección:

| Métrica | XGBoost antes | XGBoost después | LightGBM antes | LightGBM después |
|---|---:|---:|---:|---:|
| MAE | 50.981 | 49.237 | 50.058 | 48.901 |
| RMSE | 77.646 | 74.455 | 74.359 | 73.890 |
| R² | 0.8250 | 0.8391 | 0.8395 | 0.8416 |
| MAPE | 8.94% | 8.65% | 8.74% | 8.57% |
| MdAPE | 6.85% | 6.50% | 6.67% | 6.16% |

Ambos algoritmos mejoraron en las cinco métricas de test. `seleccionar_algoritmo.py` (criterio
ponderado 50% MAE + 50% RMSE test) sigue eligiendo **LightGBM** como antes, pero el margen se
achicó de 3.03% a 0.72% relativo — XGBoost cerró bastante la brecha con datos corregidos. El
[README](README.md) ya refleja este estado corregido (32 features, LightGBM vigente en
producción); esta entrada documenta el delta puntual frente a la corrida anterior, no una
comparación aparte.

### Producción: pipeline extendido para 3 features nuevas + reentrenamiento

El modelo ganador (LightGBM, 32 features) usa 3 features que el pipeline de producción no
calculaba. Se extendió:
- **`db.py`**: nueva columna `avisos_detalle.c_ig_com` (migración `ALTER TABLE` para bases ya
  existentes, mismo patrón que `intentos_fallidos_detalle`).
- **`03_vulnerabilidad_produccion.py`**: ahora también resuelve `c_ig_com` desde el shapefile
  IGVUST (antes solo `uv_rsh`/`rank_nac`/`pob_rsh_uv`/`p_urbano`).
- **`04_ingenieria_variables_produccion.py`**: agregadas `cantidad_centros_comerciales` y
  `cantidad_clinicas` (mismo patrón que los demás conteos de POI) y `c_ig_com` (mismo fallback por
  comuna que ya usan `rank_nac`/`p_urbano`/`pob_rsh_uv`).

Modelo de producción reentrenado sobre datos corregidos: nueva versión `v0005_..._lightgbm_...`
(algoritmo ganador de la sección anterior). Las 3 predicciones ya existentes en `predicciones`
(hechas con el modelo y el `gastos_comunes` sin corregir) se recalcularon con la versión nueva
**sin pisar las anteriores** — `UNIQUE(id_aviso, version_modelo)` permite que ambas convivan, tal
como está pensado el esquema.
