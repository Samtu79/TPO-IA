"""
juego/jugada.py — Qué decide el DT en cada jugada, y cómo se traduce a/desde la acción del agente.

La acción del agente es un vector de 10 números enteros (espacio "MultiDiscrete"):

  posición  qué elige                          opciones
  [0]       formación ofensiva                 5 (Shotgun, I, Singleback, Pistol, Empty)
  [1]       tipo de jugada                     3 (pase, carrera, patear)
  [2..6]    ruta de cada uno de los 5 elegibles 10 (go, post, corner, dig, comeback, curl, out, slant, flat, bloqueo)
  [7]       a quién va el pase                 5 (uno de los 5 elegibles)
  [8]       cuándo lanza                       3 (rápido, medio, largo)
  [9]       por dónde corre (si es carrera)    5 huecos

Los "elegibles" son los 5 jugadores que pueden recibir un pase (WR, TE, RB, FB),
en el orden en que aparecen en la formación.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from juego.jugadores import FORMACIONES_ATAQUE
from juego.rutas import NOMBRES_RUTAS

NOMBRES_FORMACIONES = list(FORMACIONES_ATAQUE)
TIPOS = ["pase", "carrera", "patear"]
MOMENTOS = ["rápido", "medio", "largo"]
HUECOS = ["afuera por la izquierda", "adentro por la izquierda", "por el medio",
          "adentro por la derecha", "afuera por la derecha"]
DESPLAZAMIENTO_HUECOS = [6.5, 2.7, 0.0, -2.7, -6.5]   # yardas desde la pelota (izquierda del ataque = arriba)
POSICIONES_ELEGIBLES = ("WR", "TE", "RB", "FB")
N_ELEGIBLES = 5

ESPACIO_ACCION = [len(NOMBRES_FORMACIONES), len(TIPOS)] + [len(NOMBRES_RUTAS)] * N_ELEGIBLES + \
                 [N_ELEGIBLES, len(MOMENTOS), len(HUECOS)]


def elegibles_de(formacion: str) -> list[str]:
    """Posiciones de los elegibles de una formación, en orden (ej. ['TE', 'WR', 'WR', 'WR', 'RB'])."""
    return [p for p, *_ in FORMACIONES_ATAQUE[formacion] if p in POSICIONES_ELEGIBLES]


@dataclass
class Jugada:
    formacion: str
    tipo: str                 # "pase", "carrera" o "patear"
    rutas: list[str]          # una por elegible
    objetivo: int             # índice del elegible al que va el pase
    momento: int              # 0 rápido, 1 medio, 2 largo
    hueco: int                # índice en HUECOS

    @classmethod
    def desde_accion(cls, accion) -> "Jugada":
        a = [int(v) for v in np.asarray(accion).ravel()]
        return cls(NOMBRES_FORMACIONES[a[0]], TIPOS[a[1]], [NOMBRES_RUTAS[k] for k in a[2:7]], a[7], a[8], a[9])

    def a_accion(self) -> np.ndarray:
        return np.array([NOMBRES_FORMACIONES.index(self.formacion), TIPOS.index(self.tipo)]
                        + [NOMBRES_RUTAS.index(r) for r in self.rutas]
                        + [self.objetivo, self.momento, self.hueco])

    def describir(self) -> str:
        """Descripción en lenguaje de pizarra, ej. 'Shotgun · Pase rápido al WR2 (slant)'."""
        if self.tipo == "patear":
            return "Patada (field goal si está en rango; si no, punt)"
        pos = elegibles_de(self.formacion)
        etiquetas, cuenta = [], {}
        for p in pos:
            cuenta[p] = cuenta.get(p, 0) + 1
            etiquetas.append(f"{p}{cuenta[p]}" if pos.count(p) > 1 else p)
        if self.tipo == "carrera":
            corredor = "RB" if "RB" in pos else "QB"
            return f"{self.formacion} · Carrera del {corredor} {HUECOS[self.hueco]}"
        rutas = ", ".join(f"{e} {r}" for e, r in zip(etiquetas, self.rutas))
        return (f"{self.formacion} · Pase {MOMENTOS[self.momento]} al {etiquetas[self.objetivo]} "
                f"({self.rutas[self.objetivo]}) · Rutas: {rutas}")
