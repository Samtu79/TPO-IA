"""
tests/test_cancha.py — Pruebas de la cancha y las formaciones.

Correr con:   python -m pytest tests -q
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C  # noqa: E402
from juego.jugadores import FORMACIONES_ATAQUE, FORMACIONES_DEFENSA, HASHES, armar_situacion  # noqa: E402

COMBINACIONES = list(itertools.product(FORMACIONES_ATAQUE, FORMACIONES_DEFENSA, HASHES, (1, 10, 25, 50, 80, 99)))


def test_once_contra_once():
    for ataque, defensa, h, yarda in COMBINACIONES:
        s = armar_situacion(ataque, defensa, yarda, h)
        assert sum(j.equipo == "ataque" for j in s.jugadores) == C.JUGADORES_POR_EQUIPO
        assert sum(j.equipo == "defensa" for j in s.jugadores) == C.JUGADORES_POR_EQUIPO


def test_todos_dentro_de_la_cancha_y_de_su_lado():
    for ataque, defensa, h, yarda in COMBINACIONES:
        s = armar_situacion(ataque, defensa, yarda, h)
        for j in s.jugadores:
            assert 0 < j.x < C.LARGO and 0 < j.y < C.ANCHO, (ataque, defensa, h, yarda, j)
            if j.equipo == "ataque":
                assert j.x < s.x_scrimmage, j
            else:
                assert j.x > s.x_scrimmage, j


def test_sin_jugadores_superpuestos():
    for ataque, defensa, h, yarda in COMBINACIONES:
        js = armar_situacion(ataque, defensa, yarda, h).jugadores
        for a, b in itertools.combinations(js, 2):
            d = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
            assert d >= 2 * C.RADIO_JUGADOR * 0.95, (ataque, defensa, h, yarda, a.id, b.id, round(d, 2))


def test_ids_unicos():
    s = armar_situacion("Empty (5 receptores)", "Dime (4-1-6)")
    ids = [j.id for j in s.jugadores]
    assert len(ids) == len(set(ids))


def test_gol_para_ir():
    s = armar_situacion("Formación I", "4-3", yarda=95, yardas_para_primero=10)
    assert s.x_primer_down == C.GOL_DER and "gol para ir" in s.descripcion


def test_dibujo_no_falla():
    from juego.dibujo import crear_figura, dibujar_cancha, dibujar_jugadores, dibujar_pelota, plt
    s = armar_situacion("Pistol", "3-4", 40, "izquierdo")
    fig, ax = crear_figura()
    dibujar_cancha(ax)
    dibujar_jugadores(ax, s.jugadores)
    dibujar_pelota(ax, *s.pelota)
    fig.canvas.draw()
    plt.close(fig)


def test_app_abre_sin_errores():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
