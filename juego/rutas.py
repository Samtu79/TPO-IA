"""
juego/rutas.py — El "árbol de rutas": los recorridos que puede correr un receptor.

Cada ruta es una lista de puntos de paso relativos a donde arranca el receptor:
    (yardas hacia adelante, yardas hacia AFUERA)
"Afuera" es hacia el lateral más cercano; un número negativo es hacia el medio.
Así la misma ruta sirve para un receptor de arriba o de abajo (se espeja sola).

Al terminar los puntos de paso, el receptor:
  - "sigue" corriendo en la misma dirección, o
  - "para" y se queda esperando el pase (rutas que "se sientan" en un hueco).
"""

from __future__ import annotations

import numpy as np

import config as C

# nombre: (puntos de paso, qué hace al final, descripción)
RUTAS: dict[str, tuple[list[tuple[float, float]], str, str]] = {
    "go":       ([(40, 0)], "sigue", "derecho a lo profundo"),
    "post":     ([(11, 0), (30, -10)], "sigue", "profunda hacia el medio (los postes)"),
    "corner":   ([(11, 0), (25, 10)], "sigue", "profunda hacia la esquina"),
    "dig":      ([(11, 0), (12, -14)], "sigue", "a 11 yardas cruza hacia adentro"),
    "comeback": ([(13, 0), (11, 3)], "para", "profunda y vuelve hacia el lateral"),
    "curl":     ([(10, 0), (8.5, -1.5)], "para", "a 10 yardas se da vuelta"),
    "out":      ([(6, 0), (6.5, 8)], "sigue", "a 6 yardas sale hacia el lateral"),
    "slant":    ([(3, 0), (9, -6)], "sigue", "diagonal rápida hacia adentro"),
    "flat":     ([(1.5, 3), (2.5, 12)], "sigue", "corta, hacia el costado"),
    "bloqueo":  ([], "bloquea", "se queda a proteger al QB"),
}
NOMBRES_RUTAS = list(RUTAS)          # el orden importa: es el número de acción del agente


def camino_absoluto(ruta: str, x0: float, y0: float, y_pelota: float) -> np.ndarray:
    """Convierte una ruta en puntos (x, y) de la cancha para un receptor que arranca en (x0, y0)."""
    puntos, _, _ = RUTAS[ruta]
    afuera = -1.0 if y0 < y_pelota - 0.01 else 1.0      # abajo de la pelota: afuera = hacia abajo
    camino = [(x0, y0)]
    for adelante, lateral in puntos:
        x = min(x0 + adelante, C.LARGO + 5)               # puede pasarse de la cancha (pase incompleto)
        y = float(np.clip(y0 + afuera * lateral, 1.0, C.ANCHO - 1.0))
        camino.append((x, y))
    return np.array(camino, dtype=float)


def termina_parado(ruta: str) -> bool:
    return RUTAS[ruta][1] == "para"
