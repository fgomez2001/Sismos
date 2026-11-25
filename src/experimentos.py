"""
Diseño experimental del MVP sísmico.

Pregunta de investigación
    ¿Una LSTM (con/sin atención, con/sin pesos por magnitud) predice la
    magnitud y el log-tiempo del próximo evento mejor que baselines simples?

Qué hace
    1. Calcula baselines sin entrenamiento (media, mediana, persistencia,
       promedio de la ventana).
    2. Entrena cada configuración con varias semillas.
    3. Incluye un CONTROL con etiquetas de train barajadas: si ese control
       "aprende", hay fuga de información en el pipeline.
    4. Guarda evidencia reproducible: JSON (por semilla), CSV (resumen),
       hash del dataset y versiones de librerías.

Uso
    python experimentos.py --seeds 0 1 2
    python experimentos.py --quick          # prueba rápida (3 épocas, 5000 muestras)

Requiere haber corrido preprocesamiento.py y tener torch instalado.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import time

import numpy as np
import sys
if hasattr(sys.stdout, "reconfigure"):          # evita caracteres rotos
    sys.stdout.reconfigure(encoding="utf-8")

BASE = os.environ.get("SISMOS_DIR", r"C:\Users\gomez\Downloads")
DATA_PATH = os.path.join(BASE, "secuencias_sismos.npz")
RESULTS_DIR = os.path.join(BASE, "resultados")



# Métricas 

def metricas_regresion(y_true, y_pred):
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
    }


def mae_sismos_grandes(y_true, y_pred, thr=5.0):
    """MAE solo en eventos grandes (M >= thr). Devuelve (mae, n)."""
    m = y_true >= thr
    if not m.any():
        return float("nan"), 0
    return float(np.mean(np.abs(y_true[m] - y_pred[m]))), int(m.sum())


def hash_archivo(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()



# Baselines (sin entrenamiento)

def predicciones_baseline(data):
    """Devuelve {nombre: (pred_mag, pred_log_dt)} sobre el set de test."""
    cols = [str(c) for c in data["feature_cols"]]
    mean, std = data["scaler_mean"], data["scaler_std"]
    raw = data["X_test"] * std + mean               # des-normaliza la ventana
    i_mag = cols.index("magnitude")
    if "log_dt_hours" in cols:                       # la feature ya viene en log1p
        ldt = raw[:, :, cols.index("log_dt_hours")]
    else:
        ldt = np.log1p(np.clip(raw[:, :, cols.index("dt_hours")], 0, None))
    n = len(raw)

    tr_mag = data["y_mag_train"]
    tr_ldt = np.log1p(data["y_dt_train"])

    last_mag = raw[:, -1, i_mag]
    win_mag = raw[:, :, i_mag].mean(axis=1)
    last_ldt = ldt[:, -1]
    win_ldt = ldt.mean(axis=1)

    return {
        "baseline_media_train": (np.full(n, tr_mag.mean()), np.full(n, tr_ldt.mean())),
        "baseline_mediana_train": (np.full(n, np.median(tr_mag)), np.full(n, np.median(tr_ldt))),
        "baseline_persistencia": (last_mag, last_ldt),
        "baseline_promedio_ventana": (win_mag, win_ldt),
    }



# Entrenamiento de configuraciones

CONFIGS = {
    "lstm_atencion_ponderada": dict(arch="attn", weights=True, shuffle=False),
    "lstm_atencion_sin_pesos": dict(arch="attn", weights=False, shuffle=False),
    "lstm_apilada_sin_pesos": dict(arch="stack", weights=False, shuffle=False),
    "control_etiquetas_barajadas": dict(arch="attn", weights=False, shuffle=True),
}


def fijar_semilla(seed):
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predecir(model, X, device, batch=2048):
    import torch
    model.eval()
    pm, pd_ = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).float().to(device)
            out = model(xb)
            pm.append(out[0].cpu().numpy())
            pd_.append(out[1].cpu().numpy())
    return np.concatenate(pm), np.concatenate(pd_)


def correr_config(name, cfg, seed, data, epochs, patience, max_train):
    from torch.utils.data import DataLoader
    from entrenamiento import (DEVICE, SismoLSTM, SismoLSTMAttention,
                                   SismosDataset, iniciar_bias, entrenar_modelo)
    fijar_semilla(seed)
    Xtr, ymtr, ydtr = data["X_train"], data["y_mag_train"].copy(), data["y_dt_train"].copy()
    if max_train:
        Xtr, ymtr, ydtr = Xtr[:max_train], ymtr[:max_train], ydtr[:max_train]
    if cfg["shuffle"]:                               # control negativo
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(ymtr))
        ymtr, ydtr = ymtr[perm], ydtr[perm]

    tl = DataLoader(SismosDataset(Xtr, ymtr, ydtr), batch_size=512, shuffle=True)
    vl = DataLoader(SismosDataset(data["X_val"], data["y_mag_val"], data["y_dt_val"]),
                    batch_size=1024)
    nf = Xtr.shape[-1]
    model = (SismoLSTMAttention(nf, hidden_size=64, num_layers=1) if cfg["arch"] == "attn"
             else SismoLSTM(nf, hidden_size=128, num_layers=2)).to(DEVICE)
    iniciar_bias(model, ymtr, ydtr)              # arranca en el nivel medio de los targets

    t0 = time.time()
    model = entrenar_modelo(model, tl, vl, epochs=epochs, patience=patience,
                        use_magnitude_weights=cfg["weights"])
    secs = time.time() - t0
    return predecir(model, data["X_test"], DEVICE), secs


def resumir(name, pred_mag, pred_ldt, true_mag, true_ldt):
    m = metricas_regresion(true_mag, pred_mag)
    d = metricas_regresion(true_ldt, pred_ldt)
    big, n_big = mae_sismos_grandes(true_mag, pred_mag)
    return {"modelo": name, "mae_mag": m["mae"], "rmse_mag": m["rmse"], "r2_mag": m["r2"],
            "mae_mag_M5+": big, "n_M5+": n_big,
            "mae_logdt": d["mae"], "rmse_logdt": d["rmse"], "r2_logdt": d["r2"]}


def promediar_semillas(rows):
    """Media ± desviación estándar entre semillas para cada modelo."""
    out = {}
    for name in dict.fromkeys(r["modelo"] for r in rows):
        sub = [r for r in rows if r["modelo"] == name]
        agg = {"modelo": name, "n_semillas": len(sub)}
        for k in sub[0]:
            if k in ("modelo", "n_M5+"):
                continue
            vals = np.array([r[k] for r in sub], dtype=float)
            agg[k] = float(np.nanmean(vals))
            agg[k + "_std"] = float(np.nanstd(vals))
        agg["n_M5+"] = sub[0]["n_M5+"]
        out[name] = agg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max-train", type=int, default=0, help="submuestreo (0 = todo)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-models", action="store_true", help="solo baselines")
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.max_train, args.seeds = 3, 5000, args.seeds[:1]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = np.load(DATA_PATH)
    true_mag, true_ldt = data["y_mag_test"], np.log1p(data["y_dt_test"])

    rows = []
    for name, (pm, pdt) in predicciones_baseline(data).items():
        rows.append(resumir(name, pm, pdt, true_mag, true_ldt))

    env = {"python": platform.python_version(), "numpy": np.__version__,
           "dataset_sha256": hash_archivo(DATA_PATH), "args": vars(args),
           "n_train": int(len(data["X_train"])), "n_val": int(len(data["X_val"])),
           "n_test": int(len(data["X_test"]))}

    if not args.skip_models:
        import torch
        env.update(torch=torch.__version__, cuda=torch.cuda.is_available())
        total = len(CONFIGS) * len(args.seeds)
        done = 0
        for name, cfg in CONFIGS.items():
            for seed in args.seeds:
                done += 1
                print(f"\n=== [{done}/{total}] {name} | seed={seed} ===", flush=True)
                (pm, pdt), secs = correr_config(name, cfg, seed, data, args.epochs,
                                             args.patience, args.max_train)
                r = resumir(name, pm, pdt, true_mag, true_ldt)
                r.update(seed=seed, train_seconds=round(secs, 1))
                rows.append(r)
                print(f"Terminado {name} en {secs / 60:.1f} min | MAE_mag={r['mae_mag']:.3f} "
                      f"R2_mag={r['r2_mag']:.3f}", flush=True)

    summary = promediar_semillas(rows)
    with open(os.path.join(RESULTS_DIR, "resultados_experimentos.json"), "w", encoding="utf-8") as f:
        json.dump({"env": env, "runs": rows, "summary": summary}, f, indent=2, ensure_ascii=False)

    cols = ["modelo", "mae_mag", "rmse_mag", "r2_mag", "mae_mag_M5+", "mae_logdt", "r2_logdt"]
    with open(os.path.join(RESULTS_DIR, "resumen_experimentos.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols + ["mae_mag_std", "r2_mag_std"])
        for s in summary.values():
            w.writerow([s["modelo"]] + [round(s[c], 4) for c in cols[1:]]
                       + [round(s.get("mae_mag_std", 0), 4), round(s.get("r2_mag_std", 0), 4)])

    print("\n" + " | ".join(cols))
    for s in summary.values():
        print(" | ".join([s["modelo"]] + [f"{s[c]:.3f}" for c in cols[1:]]))
    print(f"\nEvidencia guardada en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
