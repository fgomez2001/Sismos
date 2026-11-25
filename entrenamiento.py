"""
Modelo de forecasting sísmico (magnitud + tiempo al próximo evento) usando
una LSTM multi-tarea en PyTorch.

Requiere haber corrido antes preprocesamiento.py (genera secuencias_sismos.npz).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_PATH = r"C:\Users\gomez\Downloads\secuencias_sismos.npz"
MODEL_OUT_PATH = r"C:\Users\gomez\Downloads\modelo_sismos.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. DATASET


class SismosDataset(Dataset):
    """Envuelve secuencias (X) y targets (magnitud, log1p(dt_horas))."""

    def __init__(self, X, y_mag, y_dt):
        self.X = torch.from_numpy(X).float()
        self.y_mag = torch.from_numpy(y_mag).float()
        # log1p porque el tiempo entre eventos tiene cola muy larga
        # (la mayoría de los sismos ocurren minutos/horas después del anterior,
        # pero hay esperas de semanas) -> log estabiliza la escala de la pérdida
        self.y_dt = torch.from_numpy(np.log1p(y_dt)).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_mag[idx], self.y_dt[idx]



# 2. MODELO


class SismoLSTMAttention(nn.Module):
    """LSTM + atención aditiva sobre todos los timesteps (en vez de usar
    solo el último estado oculto) + dos cabezas densas (magnitud, log-tiempo).

    La atención permite que el modelo "mire hacia atrás" y pese más los
    eventos pasados que más se parecen al patrón que antecede a un sismo
    grande, en vez de depender únicamente del evento inmediatamente anterior.
    """

    def __init__(self, n_features, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_score = nn.Linear(hidden_size, 1)
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_magnitude = nn.Linear(64, 1)
        self.head_log_dt = nn.Linear(64, 1)

    def forward(self, x):
        # x: [batch, seq_len, n_features]
        outputs, _ = self.lstm(x)                    # [batch, seq_len, hidden]
        scores = self.attn_score(outputs)             # [batch, seq_len, 1]
        weights = torch.softmax(scores, dim=1)        # atención sobre el tiempo
        context = (weights * outputs).sum(dim=1)      # [batch, hidden]
        shared = self.shared(context)
        mag = self.head_magnitude(shared).squeeze(-1)
        log_dt = self.head_log_dt(shared).squeeze(-1)
        return mag, log_dt, weights.squeeze(-1)        # devolvemos pesos para poder inspeccionarlos


class SismoLSTM(nn.Module):
    """LSTM apilada + dos cabezas densas (magnitud y log-tiempo)."""

    def __init__(self, n_features, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_magnitude = nn.Linear(64, 1)
        self.head_log_dt = nn.Linear(64, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]              # último estado de la última capa
        shared = self.shared(last_hidden)
        mag = self.head_magnitude(shared).squeeze(-1)
        log_dt = self.head_log_dt(shared).squeeze(-1)
        return mag, log_dt



# 3. ENTRENAMIENTO


def pesos_magnitud(y_mag: torch.Tensor) -> torch.Tensor:
    """Pesos para dar más importancia a sismos grandes, que son muy pocos
    en el dataset y que el modelo tiende a subestimar (regresión a la
    media). Escala linealmente a partir de M4, con un tope para no dejar
    que un único evento domine el gradiente."""
    weight = 1.0 + 3.0 * torch.clamp(y_mag - 4.0, min=0.0)
    return torch.clamp(weight, max=6.0)


def iniciar_bias(model, y_mag_train, y_dt_train):
    """Inicializa el sesgo de las dos cabezas con la media de los targets de
    train. Sin esto las salidas arrancan cerca de 0 mientras la magnitud media
    es ~3,2 y el modelo gasta muchas iteraciones solo en 'llegar' al nivel
    correcto (con pocas épocas queda con R2 muy negativo)."""
    with torch.no_grad():
        # mediana (no media): la magnitud es asimétrica (Gutenberg-Richter) y el MAE se minimiza con la mediana
        model.head_magnitude.bias.fill_(float(np.median(y_mag_train)))
        model.head_log_dt.bias.fill_(float(np.mean(np.log1p(y_dt_train))))


def entrenar_modelo(
    model, train_loader, val_loader,
    epochs=30, lr=1e-3, patience=5,
    loss_weight_mag=1.0, loss_weight_dt=0.5,
    use_magnitude_weights=False,
    huber_delta=0.2,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    # delta pequeño => la pérdida se comporta como L1 (aprende la mediana condicional),
    # alineada con el MAE que se usa para evaluar. Con delta=1 era casi MSE (media).
    criterion = nn.HuberLoss(reduction="none", delta=huber_delta)  

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X, y_mag, y_dt in train_loader:
            X, y_mag, y_dt = X.to(DEVICE), y_mag.to(DEVICE), y_dt.to(DEVICE)
            optimizer.zero_grad()
            out = model(X)
            pred_mag, pred_log_dt = out[0], out[1]   # funciona con o sin atención (3er valor = pesos)

            per_sample_loss = (
                loss_weight_mag * criterion(pred_mag, y_mag)
                + loss_weight_dt * criterion(pred_log_dt, y_dt)
            )
            if use_magnitude_weights:
                w = pesos_magnitud(y_mag)
                loss = (per_sample_loss * w).mean()
            else:
                loss = per_sample_loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_loader.dataset)

        val_loss, val_mae_mag, val_mae_dt = evaluar(model, val_loader,
                                                       loss_weight_mag, loss_weight_dt,
                                                       use_magnitude_weights, huber_delta)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_MAE_mag={val_mae_mag:.3f} | "
              f"val_MAE_log_dt={val_mae_dt:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping en epoch {epoch} (sin mejora en {patience} epochs)")
                break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluar(model, loader, w_mag=1.0, w_dt=0.5, use_magnitude_weights=False, huber_delta=0.2):
    model.eval()
    criterion = nn.HuberLoss(reduction="none", delta=huber_delta)
    total_loss, total_mae_mag, total_mae_dt, n = 0.0, 0.0, 0.0, 0
    for X, y_mag, y_dt in loader:
        X, y_mag, y_dt = X.to(DEVICE), y_mag.to(DEVICE), y_dt.to(DEVICE)
        out = model(X)
        pred_mag, pred_log_dt = out[0], out[1]
        per_sample_loss = w_mag * criterion(pred_mag, y_mag) + w_dt * criterion(pred_log_dt, y_dt)
        if use_magnitude_weights:
            w = pesos_magnitud(y_mag)
            loss = (per_sample_loss * w).mean()
        else:
            loss = per_sample_loss.mean()
        total_loss += loss.item() * X.size(0)
        total_mae_mag += torch.abs(pred_mag - y_mag).sum().item()
        total_mae_dt += torch.abs(pred_log_dt - y_dt).sum().item()
        n += X.size(0)
    return total_loss / n, total_mae_mag / n, total_mae_dt / n



# 4. BASELINE INGENUO 


def calcular_baselines(data):
    """MAE en test de baselines honestos para la magnitud. El modelo debe
    superar al MEJOR de ellos, no solo a la media."""
    y_tr, y_te = data["y_mag_train"], data["y_mag_test"]
    cols = [str(c) for c in data["feature_cols"]]
    raw = data["X_test"] * data["scaler_std"] + data["scaler_mean"]   # des-normaliza
    mags = raw[:, :, cols.index("magnitude")]
    return {
        "media_train": np.abs(y_te - y_tr.mean()).mean(),
        "mediana_train": np.abs(y_te - np.median(y_tr)).mean(),
        "persistencia (ultimo evento)": np.abs(y_te - mags[:, -1]).mean(),
        "promedio de la ventana": np.abs(y_te - mags.mean(axis=1)).mean(),
    }



if __name__ == "__main__":
    print(f"Usando device: {DEVICE}")

    data = np.load(DATA_PATH)
    train_ds = SismosDataset(data["X_train"], data["y_mag_train"], data["y_dt_train"])
    val_ds = SismosDataset(data["X_val"], data["y_mag_val"], data["y_dt_val"])
    test_ds = SismosDataset(data["X_test"], data["y_mag_test"], data["y_dt_test"])

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)

    n_features = data["X_train"].shape[-1]
    model = SismoLSTMAttention(n_features=n_features, hidden_size=64, num_layers=1).to(DEVICE)
    iniciar_bias(model, data["y_mag_train"], data["y_dt_train"])
    print(model)

    # Sin pesos por magnitud: sesgaban las predicciones hacia arriba y empeoraban MAE/R2.
    # (siguen disponibles con use_magnitude_weights=True si se quiere priorizar sismos grandes)
    model = entrenar_modelo(model, train_loader, val_loader, epochs=30, patience=5,
                        use_magnitude_weights=False)

    test_loss, test_mae_mag, test_mae_dt = evaluar(model, test_loader)
    print(f"\n[TEST] loss={test_loss:.4f} | MAE_magnitud={test_mae_mag:.3f} | "
          f"MAE_log_tiempo={test_mae_dt:.3f}")

    print("\n[BASELINES] MAE de magnitud en test:")
    base = calcular_baselines(data)
    for k, v in base.items():
        print(f"  {k:30s} {v:.3f}")
    best_name = min(base, key=base.get)
    best = base[best_name]
    improvement = (best - test_mae_mag) / best * 100
    print(f"\nMejor baseline: {best_name} ({best:.3f})")
    print(f"Mejora del modelo sobre el mejor baseline: {improvement:.1f}%")
    if improvement <= 0:
        print("AVISO: el modelo NO supera al mejor baseline; no se debe afirmar lo contrario.")

    torch.save(model.state_dict(), MODEL_OUT_PATH)
    print(f"Modelo guardado en {MODEL_OUT_PATH}")
