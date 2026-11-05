"""
juego/drive.py — Reglas del drive: downs, primeros downs, touchdowns, patadas y cambios de posesión.

Un DRIVE es una serie de jugadas del mismo equipo hasta que:
  - anota (touchdown = 7, field goal = 3),
  - pierde la pelota (intercepción, no consigue el primer down en 4to, safety), o
  - patea un punt.

También define los PUNTOS ESPERADOS (EP) de cada situación, base de la recompensa (EPA).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config as C
from juego.jugadores import HASHES


@dataclass
class EstadoDrive:
    yarda: int = 25           # desde la propia línea de gol (1 a 99)
    down: int = 1
    distancia: int = 10       # yardas para el primer down
    hash: str = "centro"
    jugadas: int = 0

    def describir(self) -> str:
        ordinal = {1: "1ra", 2: "2da", 3: "3ra", 4: "4ta"}[self.down]
        lugar = (f"propia {self.yarda}" if self.yarda < 50 else
                 "mitad de cancha" if self.yarda == 50 else f"rival {100 - self.yarda}")
        dist = "gol" if self.yarda + self.distancia >= 100 else str(self.distancia)
        return f"{ordinal} y {dist} en {lugar}"


def puntos_esperados(yarda: float, down: int = 1, distancia: float = 10) -> float:
    """Cuántos puntos 'vale' tener la pelota en esta situación (modelo lineal simple)."""
    ep = (C.EP_BASE + C.EP_POR_YARDA * yarda - C.EP_POR_DOWN * (down - 1)
          - C.EP_POR_YARDA_EXTRA * max(distancia - 10, 0))
    return float(np.clip(ep, -2.0, 6.5))


def valor_estado(e: EstadoDrive) -> float:
    return puntos_esperados(e.yarda, e.down, e.distancia)


def valor_para_el_rival(yarda_nuestra: float) -> float:
    """Si perdemos la pelota en nuestra yarda Y, el rival arranca en SU yarda 100 - Y: eso vale en contra."""
    return -puntos_esperados(100 - yarda_nuestra)


def hash_siguiente(y_final: float) -> str:
    """En la NFL la pelota se ubica entre los hashes; acá la llevamos al hash más cercano (o al centro)."""
    if y_final <= (HASHES["derecho"] + C.MITAD_ANCHO) / 2:
        return "derecho"
    if y_final >= (HASHES["izquierdo"] + C.MITAD_ANCHO) / 2:
        return "izquierdo"
    return "centro"


@dataclass
class Desenlace:
    """Cómo quedó el drive después de una jugada."""
    estado: EstadoDrive | None    # None si el drive terminó
    terminado: bool
    motivo: str = ""              # touchdown, field_goal, fg_errado, punt, intercepcion, downs, safety
    puntos: float = 0.0           # puntos anotados (o en contra, en un safety)
    valor_final: float = 0.0      # valor del final del drive (puntos o valor que le dejamos al rival)
    primer_down: bool = False
    texto: str = ""


def aplicar_jugada(e: EstadoDrive, resultado) -> Desenlace:
    """Aplica el resultado de una jugada de pase o carrera a la situación del drive."""
    x_los = C.GOL_IZQ + e.yarda
    if resultado.tipo == "intercepcion":
        yarda_nuestra = resultado.x_final - C.GOL_IZQ   # dónde la interceptaron (en nuestras yardas)
        if yarda_nuestra >= 100:               # interceptada en la zona de anotación: touchback (rival en su 20)
            yarda_nuestra = 80
        return Desenlace(None, True, "intercepcion", 0.0, valor_para_el_rival(yarda_nuestra),
                         texto="Intercepción: la pelota pasa al rival")
    if resultado.touchdown:
        return Desenlace(None, True, "touchdown", C.PUNTOS["td"], C.PUNTOS["td"], texto="¡TOUCHDOWN!")

    ganancia = resultado.x_final - x_los if resultado.tipo != "pase_incompleto" else 0.0
    nueva_x = x_los + ganancia
    if nueva_x <= C.GOL_IZQ:                   # tackleado en su propia zona de anotación
        return Desenlace(None, True, "safety", C.PUNTOS["safety"], C.PUNTOS["safety"] - 0.5,
                         texto="¡Safety! (2 puntos para el rival)")
    yarda = int(np.clip(round(nueva_x - C.GOL_IZQ), 1, 99))
    hash_n = hash_siguiente(resultado.y_final) if resultado.tipo != "pase_incompleto" else e.hash
    if ganancia >= e.distancia:
        nuevo = EstadoDrive(yarda, 1, min(10, 100 - yarda), hash_n, e.jugadas + 1)
        return Desenlace(nuevo, False, primer_down=True, texto=f"¡Primer down! {nuevo.describir()}")
    if e.down == 4:
        return Desenlace(None, True, "downs", 0.0, valor_para_el_rival(yarda),
                         texto="No llegó en 4to down: la pelota pasa al rival")
    nuevo = EstadoDrive(yarda, e.down + 1, int(max(1, round(e.distancia - ganancia))), hash_n, e.jugadas + 1)
    return Desenlace(nuevo, False, texto=nuevo.describir())


def patear(e: EstadoDrive, rng: np.random.Generator) -> Desenlace:
    """Field goal si está en rango; si no, punt."""
    distancia_fg = (100 - e.yarda) + 17        # 10 de la zona de anotación + 7 de la ubicación del pateador
    if distancia_fg <= C.RANGO_FG_MAXIMO:
        prob = float(np.clip(0.99 - 0.0115 * max(distancia_fg - 25, 0) - 0.0006 * max(distancia_fg - 45, 0) ** 2,
                             0.05, 0.99))
        if rng.random() < prob:
            return Desenlace(None, True, "field_goal", C.PUNTOS["fg"], C.PUNTOS["fg"],
                             texto=f"Field goal de {distancia_fg} yardas: ¡adentro! (+3)")
        return Desenlace(None, True, "fg_errado", 0.0, valor_para_el_rival(max(e.yarda - 7, 1)),
                         texto=f"Field goal de {distancia_fg} yardas: errado")
    caida = e.yarda + C.DISTANCIA_PUNT_NETA
    yarda_rival = 20 if caida >= 100 else 100 - caida   # touchback: el rival arranca en su 20
    return Desenlace(None, True, "punt", 0.0, -puntos_esperados(yarda_rival),
                     texto=f"Punt: el rival arranca en su {yarda_rival}")
