"""
MVP de inferencia: dado el catálogo, toma los últimos SEQ_LEN eventos (hasta una
fecha opcional) y estima la magnitud y el tiempo esperado del SIGUIENTE evento
registrado en el catálogo.

Uso:
    python prediccion.py
    python prediccion.py --until 2023-06-30

"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

import preprocesamiento as pp
from entrenamiento import DEVICE, SismoLSTMAttention

BASE = os.environ.get("SISMOS_DIR", r"C:\Users\gomez\Downloads")
DATA_PATH = os.path.join(BASE, "secuencias_sismos.npz")
MODEL_PATH = os.path.join(BASE, "modelo_sismos.pt")
SEQ_LEN = 30



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=pp.RAW_PATH)
    ap.add_argument("--until", default=None, help="usar solo eventos hasta esta fecha (YYYY-MM-DD)")
    args = ap.parse_args()

    meta = np.load(DATA_PATH)                        
    mean, std = meta["scaler_mean"], meta["scaler_std"]
    cols = [str(c) for c in meta["feature_cols"]]
    assert cols == pp.FEATURE_COLS, "las columnas del modelo no coinciden con preprocesamiento"

    df = pp.agregar_features(pp.cargar_datos(args.raw))
    if args.until:
        df = df[df["datetime"] <= pd.Timestamp(args.until) + pd.Timedelta(days=1)]
    if len(df) < SEQ_LEN:
        raise SystemExit(f"Se necesitan al menos {SEQ_LEN} eventos y hay {len(df)}")

    window = df.tail(SEQ_LEN)
    x = ((window[cols].to_numpy(dtype=np.float32) - mean) / std)[None, ...]

    model = SismoLSTMAttention(n_features=len(cols), hidden_size=64, num_layers=1).to(DEVICE)
   
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    with torch.no_grad():
        mag, log_dt, attn = model(torch.from_numpy(x).float().to(DEVICE))

    hours = float(np.expm1(log_dt.item()))
    top = np.argsort(-attn[0].cpu().numpy())[:3]
    print(f"Última fecha usada : {window['datetime'].iloc[-1]}")
    fecha_pred = window["datetime"].iloc[-1] + pd.Timedelta(hours=hours)
    print(f"Fecha estimada del siguiente evento: {fecha_pred}")
    print(f"Magnitud estimada  : {mag.item():.2f}")
    print(f"Espera estimada    : {hours:.1f} horas ({hours * 60:.0f} min)")
    print("Eventos de la ventana con mas atencion (0 = mas antiguo):",
          [(int(i), round(float(attn[0, i]), 3)) for i in top])
    

if __name__ == "__main__":
    main()
