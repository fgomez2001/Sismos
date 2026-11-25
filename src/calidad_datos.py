"""
Revisión de calidad de datos del catálogo sísmico.

Genera:
  - reporte_calidad.json : conteos de datos crudos, limpieza y alertas
  - calidad_datos.png         : eventos por año y magnitud de completitud (Mc)

Uso:
    python calidad_datos.py                    
    python calidad_datos.py --raw ruta.csv
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import preprocesamiento as pp

BASE = os.environ.get("SISMOS_DIR", r"C:\Users\gomez\Downloads")
OUT_DIR = os.path.join(BASE, "resultados")

TRAIN_END, VAL_END = "2018-12-31", "2021-12-31"   


def dataset_crudo(path):
    df = pd.read_csv(path, sep=";")
    rep = {"filas": int(len(df)), "nulos_por_columna": {c: int(v) for c, v in df.isna().sum().items() if v}}
    sec = df["second"].astype(int)
    dt = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"], hour=df["hour"],
                             minute=df["minute"], second=sec), errors="coerce")
    rep["fechas_invalidas"] = int(dt.isna().sum())
    rep["fuera_de_chile_continental"] = int(
        (~df["lat"].between(-56, -17) | ~df["long"].between(-76, -66)).sum())
    rep["lat_imposible"] = int((~df["lat"].between(-90, 90)).sum())
    rep["magnitud_fuera_de_0_10"] = int((~df["magn1"].between(0, 10)).sum())
    rep["profundidad_negativa_o_gt_700km"] = int((~df["z"].between(0, 700)).sum())
    rep["duplicados_exactos"] = int(pd.DataFrame({"t": dt, "la": df["lat"], "lo": df["long"]}).duplicated().sum())
    return rep


def calcular_mc(mags, bin_w=0.1):
    """Magnitud de completitud estimada con el máximo de la curva frecuencia-magnitud."""
    edges = np.arange(np.floor(mags.min()), mags.max() + bin_w, bin_w)
    hist, edges = np.histogram(mags, bins=edges)
    return float(edges[np.argmax(hist)])


def calcular_ks(a, b):
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / len(a)
    cdf_b = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.abs(cdf_a - cdf_b).max())


def dataset_limpio(df):
    d = df.copy()
    d["dt_hours"] = d["datetime"].diff().dt.total_seconds() / 3600.0
    d["year"] = d["datetime"].dt.year
    per_year = d.groupby("year").agg(eventos=("magnitude", "size"),
                                     mag_media=("magnitude", "mean"),
                                     mag_p05=("magnitude", lambda x: np.percentile(x, 5)))
    per_year["mc_estimada"] = d.groupby("year")["magnitude"].apply(calcular_mc)

    part = np.where(d["datetime"] <= TRAIN_END, "train",
                    np.where(d["datetime"] <= VAL_END, "val", "test"))
    d["split"] = part
    tr, te = d[d.split == "train"], d[d.split == "test"]
    shift = {
        "ks_magnitud_train_vs_test": calcular_ks(tr.magnitude.values, te.magnitude.values),
        "ks_log_dt_train_vs_test": calcular_ks(np.log1p(tr.dt_hours.dropna().values),
                                           np.log1p(te.dt_hours.dropna().values)),
        "ks_profundidad_train_vs_test": calcular_ks(tr.depth.values, te.depth.values),
        "mc_train": calcular_mc(tr.magnitude.values), "mc_test": calcular_mc(te.magnitude.values),
        "eventos_por_split": {k: int(v) for k, v in d.split.value_counts().items()},
    }
    alerts = []
    if abs(shift["mc_test"] - shift["mc_train"]) >= 0.3:
        alerts.append("La magnitud de completitud cambió entre train y test: el catálogo "
                      "no es homogéneo en el tiempo (cambios de red sísmica).")
    if shift["ks_magnitud_train_vs_test"] > 0.1:
        alerts.append("La distribución de magnitudes cambia entre train y test (KS > 0.1).")
    if shift["ks_log_dt_train_vs_test"] > 0.1:
        alerts.append("La distribución de tiempos entre eventos cambia entre train y test (KS > 0.1).")
    big = int((d.loc[d.split == "test", "magnitude"] >= 5).sum())
    if big < 30:
        alerts.append(f"Solo {big} eventos M>=5 en test: las métricas en sismos grandes son muy ruidosas.")
    return per_year, shift, alerts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=pp.RAW_PATH)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    raw = dataset_crudo(args.raw)
    df = pp.cargar_datos(args.raw)
    per_year, shift, alerts = dataset_limpio(df)

    report = {"crudo": raw, "eventos_limpios": int(len(df)),
              "magnitud": {k: float(v) for k, v in df.magnitude.describe().items()},
              "desplazamiento_de_distribucion": shift, "alertas": alerts}
    with open(os.path.join(OUT_DIR, "reporte_calidad.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(per_year.index, per_year["eventos"])
    ax[0].set_title("Eventos por año (post-limpieza)")
    ax[1].plot(per_year.index, per_year["mc_estimada"], marker="o", label="Mc estimada")
    ax[1].plot(per_year.index, per_year["mag_media"], marker="s", label="magnitud media")
    ax[1].set_title("Completitud del catálogo en el tiempo")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "calidad_datos.png"), dpi=120)
    plt.close()

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
