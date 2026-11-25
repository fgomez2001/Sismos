"""
Preprocesamiento de catálogo sísmico de Chile (2000-2023) para forecasting
de magnitud / tiempo hasta el próximo evento con deep learning.

Pasos:
  1. Limpieza y construcción de datetime real
  2. Feature engineering espacio-temporal
  3. Construcción de secuencias (sliding window) para modelos secuenciales
  4. Split temporal train/val/test
  5. Normalización (fit solo en train)
"""

import numpy as np
import pandas as pd

RAW_PATH = r"C:\Users\gomez\Downloads\Dataset_Earthquake_Chile_2000_nov2023.csv"
OUTPUT_PATH = r"C:\Users\gomez\Downloads\secuencias_sismos.npz"


# 1. CARGA Y LIMPIEZA


def cargar_datos(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # segundo trae decimales -> separamos segundos enteros y microsegundos
    sec_int = df["second"].astype(int)
    microsec = ((df["second"] - sec_int) * 1e6).round().astype(int)

    df["datetime"] = pd.to_datetime(
        dict(
            year=df["year"], month=df["month"], day=df["day"],
            hour=df["hour"], minute=df["minute"], second=sec_int,
        ),
        errors="coerce",
    ) + pd.to_timedelta(microsec, unit="us")

    n_before = len(df)
    df = df.dropna(subset=["datetime"]).copy()
    if len(df) < n_before:
        print(f"Se descartaron {n_before - len(df)} filas con fecha inválida")

    # Filtro geográfico: Chile continental (bounding box aproximado).
    # A partir de ~2020 el catálogo incluye eventos del Territorio Chileno
    # Antártico (lat < -56, long fuera de -76/-66), que tienen una
    # distribución espacial completamente distinta a la de Chile
    # continental. Mezclarlos rompe la normalización de lat/long y genera
    # un fuerte cambio de distribución justo en el período de test.
    # También elimina 1 fila con coordenada físicamente imposible
    # (lat=-94.67, un error de tipeo en el dataset original).
    LAT_MIN, LAT_MAX = -56, -17
    LON_MIN, LON_MAX = -76, -66
    n_before_geo = len(df)
    df = df[
        (df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX)
        & (df["long"] >= LON_MIN) & (df["long"] <= LON_MAX)
    ].copy()
    print(f"Se descartaron {n_before_geo - len(df)} eventos fuera de Chile "
          f"continental (Antártica / errores de coordenadas)")

    df = df.sort_values("datetime").reset_index(drop=True)

    # duplicados exactos (mismo instante, misma ubicación aprox.)
    dup_mask = df.duplicated(subset=["datetime", "lat", "long"], keep="first")
    if dup_mask.sum() > 0:
        print(f"Se descartaron {dup_mask.sum()} eventos duplicados")
        df = df[~dup_mask].reset_index(drop=True)

    df = df.rename(columns={"magn1": "magnitude", "z": "depth"})
    return df[["datetime", "lat", "long", "depth", "magnitude"]]



# 2. FEATURE ENGINEERING


def calcular_distancia(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def calcular_b_value(mags: pd.Series, window: int = 50) -> pd.Series:
    """b-value de Gutenberg-Richter estimado con el método de máxima
    verosimilitud (Aki, 1965) sobre una ventana móvil de N eventos.
    Limitación: usa el mínimo de cada ventana como magnitud de completitud
    (Mc), lo que sesga la estimación; es solo un indicador de contexto."""
    def _calc_b(vals):
        m = np.asarray(vals)
        mmin = m.min()
        mean_excess = (m - mmin).mean()
        if mean_excess <= 0:
            return np.nan
        return np.log10(np.e) / mean_excess
    return mags.rolling(window).apply(_calc_b, raw=True)


def agregar_region(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica la zona tectónica de Chile continental como one-hot.
    Los segmentos de subducción de Chile tienen comportamiento sísmico
    distinto (acoplamiento, historia de grandes terremotos, geometría de
    la placa de Nazca), así que darle esta información al modelo ayuda a
    contextualizar mejor la magnitud esperada."""
    df = df.copy()
    bins = [-56, -44, -37, -30, -24, -17]
    labels = ["austral", "sur", "centro", "norte_chico", "norte_grande"]
    region = pd.cut(df["lat"], bins=bins, labels=labels, include_lowest=True)
    onehot = pd.get_dummies(region, prefix="region").astype(np.float32)
    for col in [f"region_{l}" for l in labels]:
        if col not in onehot.columns:
            onehot[col] = 0.0
    df = pd.concat([df, onehot[[f"region_{l}" for l in labels]]], axis=1)
    return df


def agregar_features(df: pd.DataFrame, b_window: int = 50) -> pd.DataFrame:
    df = df.copy()

    # tiempo entre eventos consecutivos, en horas (guardamos la versión cruda
    # para el target de entrenamiento) y su log1p para usar como FEATURE de
    # entrada, ya que la distribución cruda tiene cola muy larga (pausas de
    # semanas) que sin transformar produce outliers extremos tras normalizar
    df["dt_hours"] = df["datetime"].diff().dt.total_seconds() / 3600.0
    df["log_dt_hours"] = np.log1p(df["dt_hours"])

    # distancia al evento anterior
    df["dist_km"] = calcular_distancia(
        df["lat"].shift(1), df["long"].shift(1), df["lat"], df["long"]
    )

    # encoding cíclico de estacionalidad
    df["month_sin"] = np.sin(2 * np.pi * df["datetime"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["datetime"].dt.month / 12)
    df["hour_sin"] = np.sin(2 * np.pi * df["datetime"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["datetime"].dt.hour / 24)

    # b-value móvil (contexto de actividad sísmica reciente)
    df["b_value"] = calcular_b_value(df["magnitude"], window=b_window)

    # magnitud y profundidad del evento anterior (autoregresivo simple)
    df["prev_magnitude"] = df["magnitude"].shift(1)
    df["prev_depth"] = df["depth"].shift(1)

    df = df.dropna().reset_index(drop=True)
    df = agregar_region(df)
    return df



# 3. SECUENCIAS (SLIDING WINDOW)

FEATURE_COLS = [
    "lat", "long", "depth", "magnitude",
    "log_dt_hours", "dist_km",
    "month_sin", "month_cos", "hour_sin", "hour_cos",
    "b_value", "prev_magnitude", "prev_depth",
    "region_austral", "region_sur", "region_centro", "region_norte_chico", "region_norte_grande",
]


def crear_secuencias(df: pd.DataFrame, seq_len: int = 30):
    """Convierte el catálogo en (X, y_magnitude, y_dt_hours).
    X: [n_samples, seq_len, n_features] -> últimos `seq_len` eventos
    y: valores del evento inmediatamente siguiente a la ventana
    """
    feats = df[FEATURE_COLS].values.astype(np.float32)
    mags = df["magnitude"].values.astype(np.float32)
    dts = df["dt_hours"].values.astype(np.float32)

    n = len(df) - seq_len
    X = np.zeros((n, seq_len, len(FEATURE_COLS)), dtype=np.float32)
    y_mag = np.zeros(n, dtype=np.float32)
    y_dt = np.zeros(n, dtype=np.float32)
    target_datetime = []

    for i in range(n):
        X[i] = feats[i : i + seq_len]
        y_mag[i] = mags[i + seq_len]
        y_dt[i] = dts[i + seq_len]
        target_datetime.append(df["datetime"].iloc[i + seq_len])

    return X, y_mag, y_dt, pd.Series(target_datetime)



# 4. SPLIT TEMPORAL


def dividir_datos(target_dates: pd.Series, train_end="2018-12-31", val_end="2021-12-31"):
    train_idx = target_dates[target_dates <= train_end].index.values
    val_idx = target_dates[(target_dates > train_end) & (target_dates <= val_end)].index.values
    test_idx = target_dates[target_dates > val_end].index.values
    return train_idx, val_idx, test_idx



# 5. NORMALIZACIÓN (fit solo en train)

def ajustar_scaler(X_train: np.ndarray):
    """Devuelve media/std por feature calculadas sobre train, listas para
    aplicar a val/test (evita fuga de información)."""
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def aplicar_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / std


if __name__ == "__main__":
    df = cargar_datos()
    print(f"Eventos tras limpieza: {len(df)}")

    df = agregar_features(df)
    print(f"Eventos tras feature engineering (se pierden los primeros por rolling/shift): {len(df)}")
    print(df[FEATURE_COLS + ["datetime"]].head())

    SEQ_LEN = 30
    X, y_mag, y_dt, target_dates = crear_secuencias(df, seq_len=SEQ_LEN)
    print(f"\nSecuencias construidas: X={X.shape}, y_mag={y_mag.shape}, y_dt={y_dt.shape}")

    train_idx, val_idx, test_idx = dividir_datos(target_dates)
    print(f"\nTrain: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    mean, std = ajustar_scaler(X[train_idx])
    X_train = aplicar_scaler(X[train_idx], mean, std)
    X_val = aplicar_scaler(X[val_idx], mean, std)
    X_test = aplicar_scaler(X[test_idx], mean, std)

    np.savez(
        OUTPUT_PATH,
        X_train=X_train, y_mag_train=y_mag[train_idx], y_dt_train=y_dt[train_idx],
        X_val=X_val, y_mag_val=y_mag[val_idx], y_dt_val=y_dt[val_idx],
        X_test=X_test, y_mag_test=y_mag[test_idx], y_dt_test=y_dt[test_idx],
        feature_cols=np.array(FEATURE_COLS), scaler_mean=mean, scaler_std=std,
    )
    print(f"\nGuardado en {OUTPUT_PATH}")
