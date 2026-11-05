"""
tests/test_juego.py — Pruebas del simulador de jugadas, las reglas del drive, el entorno y los DTs.

Correr con:   python -m pytest tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C  # noqa: E402
from entorno.drive_env import DIM_OBS, DriveEnv  # noqa: E402
from entrenadores import crear_dt, jugar_drive  # noqa: E402
from juego.defensa import asignar_cobertura  # noqa: E402
from juego.drive import EstadoDrive, aplicar_jugada, patear, puntos_esperados  # noqa: E402
from juego.jugada import ESPACIO_ACCION, NOMBRES_FORMACIONES, Jugada, elegibles_de  # noqa: E402
from juego.jugadores import FORMACIONES_DEFENSA, armar_situacion  # noqa: E402
from juego.simulador import ResultadoJugada, simular_jugada  # noqa: E402

PASE = Jugada("Shotgun (3 receptores)", "pase", ["flat", "bloqueo", "go", "slant", "out"], 3, 0, 2)
CARRERA = Jugada("Formación I", "carrera", ["bloqueo"] * 5, 0, 0, 2)


def test_todas_las_formaciones_tienen_5_elegibles():
    for f in NOMBRES_FORMACIONES:
        assert len(elegibles_de(f)) == 5, f


def test_accion_ida_y_vuelta():
    rng = np.random.default_rng(0)
    for _ in range(200):
        accion = np.array([rng.integers(n) for n in ESPACIO_ACCION])
        assert (Jugada.desde_accion(accion).a_accion() == accion).all()


def test_cada_defensor_tiene_asignacion():
    for personal in FORMACIONES_DEFENSA:
        for cob in C.COBERTURAS:
            sit = armar_situacion("Empty (5 receptores)", personal, 30)
            asig = asignar_cobertura(cob, sit.jugadores, sit.x_scrimmage, sit.pelota[1])
            defensores = [i for i, j in enumerate(sit.jugadores) if j.equipo == "defensa"]
            assert sorted(asig) == defensores, (personal, cob)


def test_jugada_reproducible_con_semilla():
    sit = lambda: armar_situacion("Shotgun (3 receptores)", "Nickel (4-2-5)", 30)
    a = simular_jugada(PASE, sit(), "Cover 3", np.random.default_rng(42))
    b = simular_jugada(PASE, sit(), "Cover 3", np.random.default_rng(42))
    assert (a.tipo, a.x_final, a.descripcion) == (b.tipo, b.x_final, b.descripcion)


def test_resultados_validos():
    rng = np.random.default_rng(1)
    tipos = set()
    for k in range(300):
        cob = C.COBERTURAS[k % 5]
        jugada = PASE if k % 2 else CARRERA
        sit = armar_situacion(jugada.formacion, "4-3", 30)
        r = simular_jugada(jugada, sit, cob, rng)
        tipos.add(r.tipo)
        assert r.tipo in ("pase_completo", "pase_incompleto", "intercepcion", "sack", "carrera")
        assert -30 < r.yardas < 110
    assert {"pase_completo", "pase_incompleto", "carrera"} <= tipos


def test_grabar_cuadros_para_animar():
    sit = armar_situacion("Shotgun (3 receptores)", "Nickel (4-2-5)", 30)
    r = simular_jugada(PASE, sit, "Cover 1", np.random.default_rng(3), grabar=True)
    assert len(r.cuadros) >= 3 and r.cuadros[0]["P"].shape == (22, 2)
    from juego.animacion import animar_jugada
    gif = animar_jugada(sit, r, "prueba", cada=4)
    assert gif[:3] == b"GIF"


def _resultado(tipo, x, td=False):
    return ResultadoJugada(tipo=tipo, x_final=x, y_final=C.MITAD_ANCHO, yardas=0, touchdown=td)


def test_reglas_del_drive():
    e = EstadoDrive(yarda=25, down=1, distancia=10)
    x0 = C.GOL_IZQ + 25
    d = aplicar_jugada(e, _resultado("carrera", x0 + 12))          # primer down
    assert d.primer_down and d.estado.down == 1 and d.estado.yarda == 37
    d = aplicar_jugada(e, _resultado("carrera", x0 + 4))           # 2da y 6
    assert d.estado.down == 2 and d.estado.distancia == 6
    d = aplicar_jugada(EstadoDrive(25, 4, 3), _resultado("carrera", x0 + 1))   # falla en 4ta
    assert d.terminado and d.motivo == "downs"
    d = aplicar_jugada(e, _resultado("pase_completo", C.GOL_DER + 1, td=True))
    assert d.terminado and d.puntos == 7
    d = aplicar_jugada(EstadoDrive(3, 1, 10), _resultado("sack", C.GOL_IZQ - 1))   # safety
    assert d.motivo == "safety" and d.puntos == -2
    d = aplicar_jugada(e, _resultado("intercepcion", x0 + 20))
    assert d.motivo == "intercepcion" and d.valor_final < 0


def test_patadas():
    rng = np.random.default_rng(0)
    assert patear(EstadoDrive(yarda=95), rng).motivo in ("field_goal", "fg_errado")   # FG de 22 yardas
    d = patear(EstadoDrive(yarda=20), rng)
    assert d.motivo == "punt" and "su 40" in d.texto
    assert patear(EstadoDrive(yarda=70), rng).motivo in ("field_goal", "fg_errado")   # FG de 47 yardas
    assert patear(EstadoDrive(yarda=50), rng).motivo == "punt"                         # FG de 67: fuera de rango


def test_puntos_esperados_crecen_con_la_yarda():
    assert puntos_esperados(80) > puntos_esperados(50) > puntos_esperados(20)
    assert puntos_esperados(50, down=3) < puntos_esperados(50, down=1)


def test_entorno_gymnasium():
    from gymnasium.utils.env_checker import check_env
    env = DriveEnv()
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (DIM_OBS,)


def test_drives_terminan_siempre():
    env = DriveEnv(rival="agresivo")
    for nombre in ("aleatorio", "clasico"):
        dt = crear_dt(nombre, semilla=0)
        for s in range(30):
            d = jugar_drive(dt, env, s)
            assert d["jugadas"] <= C.MAX_JUGADAS_DRIVE
            assert d["motivo"] in ("touchdown", "field_goal", "fg_errado", "punt", "intercepcion",
                                   "downs", "safety", "limite_jugadas")


def test_clasico_supera_al_azar():
    """Sanidad: con reglas razonables se anota más que eligiendo al azar."""
    env = DriveEnv(rival="equilibrado")
    puntos = {n: np.mean([jugar_drive(crear_dt(n, semilla=0), env, 9000 + s)["puntos"] for s in range(150)])
              for n in ("aleatorio", "clasico")}
    assert puntos["clasico"] > puntos["aleatorio"], puntos


def test_dt_entrenado_sin_modelo_avisa(tmp_path):
    with pytest.raises(FileNotFoundError):
        crear_dt("entrenado", tmp_path / "no_existe.zip")


def test_entrenado_con_modelo_al_azar_funciona():
    """Sin entrenar: una red con pesos al azar tiene que poder jugar un drive (verifica la conexión)."""
    from stable_baselines3 import PPO
    from entrenadores import DTEntrenado
    env = DriveEnv()
    dt = DTEntrenado(modelo=PPO("MlpPolicy", env, device="cpu", seed=0))
    assert "puntos" in jugar_drive(dt, env, 0)
