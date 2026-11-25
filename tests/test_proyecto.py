"""
Pruebas funcionales del MVP sísmico.

Usa un catálogo SINTÉTICO (Gutenberg-Richter, b=1) con errores inyectados a
propósito (Antártica, lat imposible, duplicados, fecha inválida), así que no
depende del CSV real ni de la GPU.  Las pruebas que necesitan PyTorch se
saltan solas si torch no está instalado.
"""
import numpy as np
import pandas as pd
import pytest

import preprocesamiento as pp


# Catálogo sintético con "suciedad" conocida

N_VALID = 3000
N_ANTARTICA = 4
N_DUP = 3


def hacer_catalogo(n=N_VALID, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2000-01-01").value
    t1 = pd.Timestamp("2023-10-31").value
    ts = pd.to_datetime(np.sort(rng.integers(t0, t1, n)))
    df = pd.DataFrame({
        "year": ts.year.values, "month": ts.month.values, "day": ts.day.values,
        "hour": ts.hour.values, "minute": ts.minute.values,
        "second": ts.second.values + rng.random(n).round(2),
        "lat": rng.uniform(-45, -18, n), "long": rng.uniform(-74, -68, n),
        "z": rng.uniform(5, 150, n),
        "magn1": 2.5 + rng.exponential(1 / np.log(10), n),   # b = 1
    })
    # --- suciedad inyectada ---
    antartica = df.iloc[100:100 + N_ANTARTICA].copy()
    antartica["lat"] = -62.0
    antartica["long"] = -58.0
    antartica["second"] = 30.5            # evita colisión con los originales
    imposible = df.iloc[200:201].copy()
    imposible["lat"] = -94.67
    imposible["second"] = 31.5
    dups = df.iloc[300:300 + N_DUP].copy()   # copia exacta -> duplicado
    invalida = df.iloc[400:401].copy()
    invalida["month"], invalida["day"] = 2, 30   # 30 de febrero
    df = pd.concat([df, antartica, imposible, dups, invalida], ignore_index=True)
    df.insert(0, "Unnamed: 0", range(len(df)))
    return df


@pytest.fixture(scope="module")
def csv_crudo(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "catalogo.csv"
    hacer_catalogo().to_csv(path, sep=";", index=False)
    return str(path)


@pytest.fixture(scope="module")
def datos_limpios(csv_crudo):
    return pp.cargar_datos(csv_crudo)


@pytest.fixture(scope="module")
def datos_features(datos_limpios):
    return pp.agregar_features(datos_limpios)



# T1-T3  Limpieza

def test_T1_limpieza_elimina_errores_inyectados(datos_limpios):
    assert datos_limpios["lat"].between(-56, -17).all()
    assert datos_limpios["long"].between(-76, -66).all()
    assert not datos_limpios.duplicated(subset=["datetime", "lat", "long"]).any()
    assert datos_limpios["datetime"].notna().all()
    assert len(datos_limpios) == N_VALID, "deben sobrevivir solo los eventos válidos"


def test_T2_catalogo_ordenado_en_el_tiempo(datos_limpios):
    assert datos_limpios["datetime"].is_monotonic_increasing


def test_T3_columnas_de_salida(datos_limpios):
    assert list(datos_limpios.columns) == ["datetime", "lat", "long", "depth", "magnitude"]



# T4-T6  Features

def test_T4_haversine_un_grado_de_latitud():
    d = pp.calcular_distancia(0.0, 0.0, 1.0, 0.0)
    assert d == pytest.approx(111.19, abs=0.1)


def test_T5_features_sin_nan_ni_inf(datos_features):
    X = datos_features[pp.FEATURE_COLS].to_numpy(dtype=float)
    assert np.isfinite(X).all()


def test_T6_region_onehot_suma_uno(datos_features):
    cols = [c for c in pp.FEATURE_COLS if c.startswith("region_")]
    assert (datos_features[cols].sum(axis=1) == 1).all()



# T7-T9  Secuencias y ausencia de fuga de información (leakage)

SEQ = 10


@pytest.fixture(scope="module")
def secuencias(datos_features):
    return pp.crear_secuencias(datos_features, seq_len=SEQ)


def test_T7_forma_de_las_secuencias(secuencias, datos_features):
    X, y_mag, y_dt, dates = secuencias
    assert X.shape == (len(datos_features) - SEQ, SEQ, len(pp.FEATURE_COLS))
    assert len(y_mag) == len(y_dt) == len(dates) == len(X)


def test_T8_el_target_nunca_esta_dentro_de_la_ventana(secuencias, datos_features):
    X, y_mag, y_dt, dates = secuencias
    i_mag = pp.FEATURE_COLS.index("magnitude")
    for i in [0, 7, len(X) // 2, len(X) - 1]:
        # el target es el evento i+SEQ; la última entrada es el evento i+SEQ-1
        assert y_mag[i] == pytest.approx(datos_features["magnitude"].iloc[i + SEQ])
        assert X[i, -1, i_mag] == pytest.approx(datos_features["magnitude"].iloc[i + SEQ - 1])
        assert dates.iloc[i] >= datos_features["datetime"].iloc[i + SEQ - 1]
    # ventanas consecutivas se desplazan exactamente 1 evento
    assert np.allclose(X[1:, :-1], X[:-1, 1:])


def test_T9_split_temporal_disjunto_y_ordenado(secuencias):
    _, _, _, dates = secuencias
    tr, va, te = pp.dividir_datos(dates)
    assert not (set(tr) & set(va)) and not (set(va) & set(te)) and not (set(tr) & set(te))
    assert dates[tr].max() <= pd.Timestamp("2018-12-31")
    assert pd.Timestamp("2018-12-31") < dates[va].min() and dates[va].max() <= pd.Timestamp("2021-12-31")
    assert dates[te].min() > pd.Timestamp("2021-12-31")
    assert len(tr) > 0 and len(va) > 0 and len(te) > 0



# T10  Normalización solo con train

def test_T10_scaler_se_ajusta_solo_con_train(secuencias):
    X, _, _, dates = secuencias
    tr, _, te = pp.dividir_datos(dates)
    mean, std = pp.ajustar_scaler(X[tr])
    Z = pp.aplicar_scaler(X[tr], mean, std).reshape(-1, X.shape[-1])
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-3)
    mean_all, _ = pp.ajustar_scaler(X)
    assert not np.allclose(mean, mean_all), "el scaler no debe haber visto test"



# T11-T14  Modelo (requieren torch)

def _importar_torch():
    """Importa torch y el módulo de entrenamiento; salta SOLO el test si falta torch."""
    torch = pytest.importorskip("torch", reason="PyTorch no instalado")
    import entrenamiento as tm
    return torch, tm


def test_T11_formas_de_salida_de_ambos_modelos():
    torch, tm = _importar_torch()
    x = torch.randn(8, 30, 18)
    mag, ldt, w = tm.SismoLSTMAttention(18, hidden_size=16)(x)
    assert mag.shape == ldt.shape == (8,) and w.shape == (8, 30)
    assert torch.allclose(w.sum(dim=1), torch.ones(8), atol=1e-5)  # atención suma 1
    mag, ldt = tm.SismoLSTM(18, hidden_size=16)(x)
    assert mag.shape == ldt.shape == (8,)


def test_T12_pesos_por_magnitud():
    torch, tm = _importar_torch()
    w = tm.pesos_magnitud(torch.tensor([3.0, 4.0, 5.0, 7.0]))
    assert w.tolist() == pytest.approx([1.0, 1.0, 4.0, 6.0])   # tope en 6


def test_T13_el_modelo_puede_sobreajustar_un_lote_pequeno():
    torch, tm = _importar_torch()
    torch.manual_seed(0)
    model = tm.SismoLSTMAttention(18, hidden_size=32)
    x, ym, yd = torch.randn(64, 30, 18), torch.randn(64), torch.randn(64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = torch.nn.HuberLoss()
    losses = []
    for _ in range(150):
        opt.zero_grad()
        pm, pd_, _ = model(x)
        loss = crit(pm, ym) + crit(pd_, yd)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0], "si no baja, hay un bug en el modelo o el loop"


def test_T15_sesgo_inicial_de_las_cabezas():
    torch, tm = _importar_torch()
    model = tm.SismoLSTMAttention(18, hidden_size=16)
    ym = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    yd = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    tm.iniciar_bias(model, ym, yd)
    assert model.head_magnitude.bias.item() == pytest.approx(4.0)
    assert model.head_log_dt.bias.item() == pytest.approx(np.log1p(yd).mean())


def test_T14_metricas_de_regresion():
    _importar_torch()
    import experimentos as ex
    y = np.array([3.0, 4.0, 5.0, 6.0])
    assert ex.metricas_regresion(y, y)["r2"] == pytest.approx(1.0)
    assert ex.metricas_regresion(y, np.full(4, y.mean()))["r2"] == pytest.approx(0.0)


if __name__ == "__main__":

    import sys
    sys.exit(pytest.main(["-v", __file__]))
