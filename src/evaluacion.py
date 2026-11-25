
# DE QUE TRATA:
#   Carga el modelo ya entrenado y lo evalua a fondo con los datos de test:
#   saca metricas, hace graficos y revisa en que casos se equivoca mas.
#
# ARCHIVOS:
#   Lee:  secuencias_sismos.npz y modelo_sismos.pt
#   Crea 4 imagenes PNG:
#     - grafico_magnitud.png    (magnitud predicha contra la real)
#     - errores_magnitud.png   (histograma de los errores de magnitud)
#     - serie_magnitud.png (real vs predicho en los ultimos 200 eventos)
#     - grafico_tiempo.png      (log(1+horas) predicho contra el real)
#   Ademas muestra en consola las metricas y el error por rango de magnitud.


"""
Evaluación del modelo LSTM entrenado: métricas, gráficas de predicción vs
real, y análisis de residuos. Corre después de entrenamiento.py.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from entrenamiento import SismoLSTMAttention, SismosDataset, DEVICE
from torch.utils.data import DataLoader

DATA_PATH = r"C:\Users\gomez\Downloads\secuencias_sismos.npz"
MODEL_PATH = r"C:\Users\gomez\Downloads\modelo_sismos.pt"
OUT_DIR = r"C:\Users\gomez\Downloads\resultados"



# Le pasa todo el set de test al modelo y devuelve 4 cosas: la magnitud
# predicha, la real, el tiempo predicho y el real.
@torch.no_grad()
def obtener_predicciones(model, loader):
    model.eval()
    all_pred_mag, all_true_mag = [], []
    all_pred_dt, all_true_dt = [], []
    for X, y_mag, y_dt in loader:
        X = X.to(DEVICE)
        pred_mag, pred_log_dt, _ = model(X)
        all_pred_mag.append(pred_mag.cpu().numpy())
        all_true_mag.append(y_mag.numpy())
        all_pred_dt.append(pred_log_dt.cpu().numpy())
        all_true_dt.append(y_dt.numpy())
    return (
        np.concatenate(all_pred_mag), np.concatenate(all_true_mag),
        np.concatenate(all_pred_dt), np.concatenate(all_true_dt),
    )



# Calcula 3 metricas: el MAE (error promedio), el RMSE (parecido pero castiga
# mas los errores grandes) y el R2 (que porcentaje de la variacion de los
# datos logra explicar el modelo).
def calcular_metricas(y_true, y_pred, name=""):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"[{name}] MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}



# Grafico de puntos con lo predicho contra lo real y una linea roja que es la
# prediccion perfecta. Mientras mas cerca de la linea, mejor.
def grafico_scatter(y_true, y_pred, title, xlabel, ylabel, path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.15, s=8)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", linewidth=1, label="predicción perfecta")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()



# Histograma de los residuos (real menos predicho). Si el modelo no tiene
# sesgo, deberia quedar centrado en cero.
def grafico_residuos(residuals, title, path):
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=60)
    plt.axvline(0, color="r", linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Residuo (real - predicho)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# Compara la curva real y la predicha en los ultimos eventos de test, para ver
# a simple vista si el modelo sigue la tendencia.
def grafico_serie(y_true, y_pred, n, title, ylabel, path):
    plt.figure(figsize=(10, 4))
    plt.plot(y_true[-n:], label="real", linewidth=1)
    plt.plot(y_pred[-n:], label="predicho", linewidth=1, alpha=0.8)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(f"Últimos {n} eventos del set de test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()



# Carga test y el modelo, saca las predicciones, muestra las metricas de
# magnitud y tiempo y las compara con el baseline. Despues genera los 4
# graficos y al final calcula el error por rango de magnitud (<3, 3-4, 4-5,
# 5-6 y 6+) para ver si el modelo falla mas en los sismos grandes.
if __name__ == "__main__":
    data = np.load(DATA_PATH)
    test_ds = SismosDataset(data["X_test"], data["y_mag_test"], data["y_dt_test"])
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    model = SismoLSTMAttention(n_features=data["X_test"].shape[-1], hidden_size=64, num_layers=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))

    pred_mag, true_mag, pred_log_dt, true_log_dt = obtener_predicciones(model, test_loader)

    print("\n--- Métricas en TEST ---")
    metrics_mag = calcular_metricas(true_mag, pred_mag, name="Magnitud")
    metrics_dt = calcular_metricas(true_log_dt, pred_log_dt, name="log(1+horas hasta siguiente evento)")

    # baseline: predecir siempre el promedio de train
    baseline_pred_mag = np.full_like(true_mag, data["y_mag_train"].mean())
    print("\n--- Baseline ingenuo (promedio de train) ---")
    calcular_metricas(true_mag, baseline_pred_mag, name="Magnitud (baseline)")

    # gráficas
    grafico_scatter(
        true_mag, pred_mag,
        title="Magnitud: predicho vs real (test)",
        xlabel="Magnitud real", ylabel="Magnitud predicha",
        path=f"{OUT_DIR}/grafico_magnitud.png",
    )
    grafico_residuos(
        true_mag - pred_mag,
        title="Distribución de residuos - Magnitud",
        path=f"{OUT_DIR}/errores_magnitud.png",
    )
    grafico_serie(
        true_mag, pred_mag, n=200,
        title="Magnitud real vs predicha (últimos 200 eventos de test)",
        ylabel="Magnitud",
        path=f"{OUT_DIR}/serie_magnitud.png",
    )

    grafico_scatter(
        true_log_dt, pred_log_dt,
        title="log(1+horas): predicho vs real (test)",
        xlabel="log(1+horas) real", ylabel="log(1+horas) predicho",
        path=f"{OUT_DIR}/grafico_tiempo.png",
    )

    # métricas por rango de magnitud: ¿el modelo falla más en sismos grandes?
    print("\n--- MAE por rango de magnitud (real) ---")
    bins = [0, 3, 4, 5, 6, 10]
    labels = ["<3", "3-4", "4-5", "5-6", "6+"]
    for lo, hi, label in zip(bins[:-1], bins[1:], labels):
        mask = (true_mag >= lo) & (true_mag < hi)
        if mask.sum() > 0:
            mae_bin = np.mean(np.abs(true_mag[mask] - pred_mag[mask]))
            print(f"  Magnitud {label}: n={mask.sum():5d} | MAE={mae_bin:.3f}")

    print(f"\nGráficas guardadas en {OUT_DIR}/")