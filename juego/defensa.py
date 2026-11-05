"""
juego/defensa.py — El rival: un coordinador defensivo con reglas fijas (NO aprende).

Antes de cada jugada elige:
  1. el PERSONAL (formación defensiva), según la situación: yardas cortas -> más gente
     cerca de la línea; yardas largas -> más defensores de pase;
  2. la COBERTURA (oculta para el ataque), según su perfil (equilibrado, agresivo, conservador).

Las coberturas:
  Cover 0  todos en marca personal, el resto va a buscar al QB (blitz). Sin ayuda profunda.
  Cover 1  marca personal + 1 safety profundo en el medio.
  Cover 2  zona: 2 safeties cubren cada mitad profunda, los esquineros cuidan los costados cortos.
  Cover 3  zona: la profundidad se divide en 3 tercios (2 esquineros + 1 safety).
  Cover 4  zona: la profundidad se divide en 4 cuartos. Muy difícil pasar profundo.

Cada defensor recibe una asignación:
  ("presion",)          va a buscar al QB
  ("marca", i)          sigue al jugador de ataque número i
  ("zona", (x, y))      cuida un punto de la cancha y defiende al que entre a su zona
"""

from __future__ import annotations

import numpy as np

import config as C
from juego.jugadores import Jugador

LINEA_DEFENSIVA = ("DE", "DT", "NT", "OLB")
APOYADORES = ("LB", "ILB")


def elegir_defensa(down: int, distancia: int, perfil: str, rng: np.random.Generator) -> tuple[str, str]:
    """Personal y cobertura que elige el rival para esta jugada."""
    if distancia <= 2:
        opciones, pesos = ["4-3", "3-4", "Nickel (4-2-5)"], [0.5, 0.4, 0.1]
    elif distancia >= 8:
        opciones, pesos = ["Nickel (4-2-5)", "Dime (4-1-6)", "4-3"], [0.5, 0.35, 0.15]
    else:
        opciones, pesos = ["4-3", "3-4", "Nickel (4-2-5)", "Dime (4-1-6)"], [0.3, 0.2, 0.4, 0.1]
    personal = opciones[rng.choice(len(opciones), p=pesos)]

    prob = dict(C.PERFILES_RIVAL[perfil])
    if distancia <= 2:                       # yardas cortas: más agresivo
        prob["Cover 0"] *= 2.0
        prob["Cover 1"] *= 1.5
    elif distancia >= 10 or down == 3 and distancia >= 7:
        prob["Cover 4"] *= 1.8               # yardas largas: cuidar lo profundo
        prob["Cover 3"] *= 1.3
    nombres = list(prob)
    p = np.array([prob[n] for n in nombres])
    cobertura = nombres[rng.choice(len(nombres), p=p / p.sum())]
    return personal, cobertura


def cantidad_profundos(cobertura: str) -> int:
    """Cuántos safeties quedan profundos en esa cobertura (lo que 'muestra' antes del snap)."""
    return {"Cover 0": 0, "Cover 1": 1, "Cover 2": 2, "Cover 3": 1, "Cover 4": 2}[cobertura]


def alinear_safeties(defensa: list[Jugador], cobertura_mostrada: str, x_los: float, y_pelota: float) -> None:
    """Acomoda a los safeties antes del snap según la cobertura que quiere MOSTRAR (puede ser un disfraz)."""
    safeties = [j for j in defensa if j.posicion == "S"]
    profundos = cantidad_profundos(cobertura_mostrada)
    x_prof = min(x_los + 13, C.LARGO - 1.5)
    for k, s in enumerate(sorted(safeties, key=lambda j: j.y)):
        if profundos == 2:
            s.x, s.y = x_prof, (C.ANCHO * 0.3 if k == 0 else C.ANCHO * 0.7)
        elif profundos == 1 and k == 1:
            s.x, s.y = x_prof, y_pelota
        else:                                 # safety "bajado" cerca de la línea
            s.x, s.y = x_los + 7, y_pelota + (-6 if k == 0 else 6)
        s.y = float(np.clip(s.y, 1.5, C.ANCHO - 1.5))


def _repartir_zonas(defensores: list[int], puntos: list[tuple[float, float]], jugadores: list[Jugador],
                    asignacion: dict) -> list[int]:
    """Asigna cada punto de zona al defensor libre más cercano. Devuelve los que sobran."""
    libres = list(defensores)
    for p in puntos:
        if not libres:
            break
        k = min(libres, key=lambda i: (jugadores[i].x - p[0]) ** 2 + (jugadores[i].y - p[1]) ** 2)
        asignacion[k] = ("zona", p)
        libres.remove(k)
    return libres


def asignar_cobertura(cobertura: str, jugadores: list[Jugador], x_los: float, y_pelota: float) -> dict[int, tuple]:
    """
    Asignación de cada defensor (índice dentro de 'jugadores') para la cobertura dada.
    'jugadores' es la lista completa (11 de ataque y 11 de defensa).
    """
    A = C.ANCHO
    xmax = C.LARGO - 1.0
    ataque = [i for i, j in enumerate(jugadores) if j.equipo == "ataque"]
    defensa = [i for i, j in enumerate(jugadores) if j.equipo == "defensa"]
    elegibles = [i for i in ataque if jugadores[i].posicion in ("WR", "TE", "RB", "FB")]
    asignacion: dict[int, tuple] = {}

    linea = [i for i in defensa if jugadores[i].posicion in LINEA_DEFENSIVA]
    for i in linea:
        asignacion[i] = ("presion",)
    resto = [i for i in defensa if i not in linea]
    corners = [i for i in resto if jugadores[i].posicion in ("CB", "NB")]
    safeties = [i for i in resto if jugadores[i].posicion == "S"]
    lbs = [i for i in resto if jugadores[i].posicion in APOYADORES]

    def prof(dx):
        return min(x_los + dx, xmax)

    debajo = [(prof(4), 9.0), (prof(4), A - 9.0), (prof(7), y_pelota - 7), (prof(7), y_pelota + 7), (prof(8), y_pelota)]
    debajo = [(x, float(np.clip(y, 2, A - 2))) for x, y in debajo]

    if cobertura in ("Cover 0", "Cover 1"):
        # marca personal: cada receptor, el defensor libre más cercano (primero esquineros, después safeties, LBs)
        disponibles = corners + safeties + lbs
        if cobertura == "Cover 1" and safeties:
            libre = max(safeties, key=lambda i: jugadores[i].x)       # el safety más profundo queda de "centerfield"
            asignacion[libre] = ("zona", (prof(16), y_pelota))
            disponibles.remove(libre)
        receptores = sorted(elegibles, key=lambda i: abs(jugadores[i].y - y_pelota), reverse=True)  # de afuera hacia adentro
        for r in receptores:
            if not disponibles:
                break
            d = min(disponibles, key=lambda i: (jugadores[i].y - jugadores[r].y) ** 2 + 0.3 * (jugadores[i].x - jugadores[r].x) ** 2)
            asignacion[d] = ("marca", r)
            disponibles.remove(d)
        for d in disponibles:                                          # los que sobran: blitz
            asignacion[d] = ("presion",) if cobertura == "Cover 0" else ("zona", (prof(8), y_pelota))
        return asignacion

    zonas_cortas = debajo
    if cobertura == "Cover 2":
        profundos = [(prof(15), A * 0.25), (prof(15), A * 0.75)]
        sobran = _repartir_zonas(safeties, profundos, jugadores, asignacion)
        sobran = _repartir_zonas(corners[:2], debajo[:2], jugadores, asignacion) + sobran + corners[2:]
        zonas_cortas = debajo[2:]          # los costados cortos ya los cuidan los esquineros
    elif cobertura == "Cover 3":
        profundos = [(prof(14), A / 6), (prof(14), A * 5 / 6)]
        sobran = _repartir_zonas(corners[:2], profundos, jugadores, asignacion)
        sobran += _repartir_zonas(safeties, [(prof(16), A / 2)], jugadores, asignacion) + corners[2:]
    else:  # Cover 4
        sobran = _repartir_zonas(corners[:2], [(prof(13), A / 8), (prof(13), A * 7 / 8)], jugadores, asignacion)
        sobran += _repartir_zonas(safeties, [(prof(14), A * 3 / 8), (prof(14), A * 5 / 8)], jugadores, asignacion)
        sobran += corners[2:]
    # los demás cuidan las zonas cortas (costados, ganchos, medio); si sobra alguien, presiona
    for d in _repartir_zonas(sobran + lbs, zonas_cortas, jugadores, asignacion):
        asignacion[d] = ("presion",)
    return asignacion
