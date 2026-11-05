"""
juego/jugadores.py — Jugadores (círculos) y formaciones iniciales de ataque y defensa.

Cada formación es una lista de "instrucciones de ubicación" respecto de la pelota:
    (posición, dx, dy, referencia)
  - dx: yardas hacia adelante (+) o hacia atrás (-) de la línea de scrimmage.
  - dy: yardas hacia arriba (+) o hacia abajo (-).
  - referencia:
      "pelota"      -> dy se mide desde la pelota
      "lateral_inf" -> dy es la distancia al lateral de abajo (receptores abiertos)
      "lateral_sup" -> dy es la distancia al lateral de arriba
      "marca_ext_inf" / "marca_ext_sup"   -> se alinea con el receptor más abierto de ese lado
      "marca_slot_inf" / "marca_slot_sup" -> se alinea con el segundo receptor de ese lado
                                             (si no hay, usa dy desde la pelota)

Posiciones (abreviaturas de la NFL):
  Ataque:  QB mariscal · RB corredor · FB fullback · WR receptor abierto · TE ala cerrada
           C centro · G guardia · T tackle
  Defensa: DE ala defensiva · DT tackle defensivo · NT nose tackle · LB / ILB / OLB apoyadores
           CB esquinero · NB esquinero de slot · S safety
"""

from __future__ import annotations

from dataclasses import dataclass, field

import config as C

LINEA_OFENSIVA = [("C", -1.0, 0.0, "pelota"), ("G", -1.0, -1.8, "pelota"), ("G", -1.0, 1.8, "pelota"),
                  ("T", -1.0, -3.6, "pelota"), ("T", -1.0, 3.6, "pelota")]
LINEA_DEFENSIVA_4 = [("DE", 1.0, -4.8, "pelota"), ("DT", 1.0, -1.3, "pelota"),
                     ("DT", 1.0, 1.3, "pelota"), ("DE", 1.0, 4.8, "pelota")]
SAFETIES = [("S", 12.0, -9.0, "pelota"), ("S", 12.0, 9.0, "pelota")]
CORNERS = [("CB", 6.0, 0.0, "marca_ext_inf"), ("CB", 6.0, 0.0, "marca_ext_sup")]

FORMACIONES_ATAQUE: dict[str, list] = {
    "Shotgun (3 receptores)": LINEA_OFENSIVA + [
        ("QB", -5.0, 0.0, "pelota"), ("RB", -5.0, 1.8, "pelota"), ("TE", -1.0, 5.4, "pelota"),
        ("WR", -1.0, 8.0, "lateral_inf"), ("WR", -1.5, -9.5, "pelota"), ("WR", -1.5, 8.0, "lateral_sup")],
    "Formación I": LINEA_OFENSIVA + [
        ("QB", -2.7, 0.0, "pelota"), ("FB", -4.5, 0.0, "pelota"), ("RB", -7.0, 0.0, "pelota"),
        ("TE", -1.0, 5.4, "pelota"), ("WR", -1.0, 9.0, "lateral_inf"), ("WR", -1.5, 9.0, "lateral_sup")],
    "Singleback (2 alas cerradas)": LINEA_OFENSIVA + [
        ("QB", -2.7, 0.0, "pelota"), ("RB", -6.5, 0.0, "pelota"), ("TE", -1.0, -5.4, "pelota"),
        ("TE", -1.0, 5.4, "pelota"), ("WR", -1.5, 9.0, "lateral_inf"), ("WR", -1.5, 9.0, "lateral_sup")],
    "Pistol": LINEA_OFENSIVA + [
        ("QB", -4.0, 0.0, "pelota"), ("RB", -7.0, 0.0, "pelota"), ("TE", -1.0, 5.4, "pelota"),
        ("WR", -1.0, 8.0, "lateral_inf"), ("WR", -1.5, -9.5, "pelota"), ("WR", -1.5, 8.0, "lateral_sup")],
    "Empty (5 receptores)": LINEA_OFENSIVA + [
        ("QB", -5.0, 0.0, "pelota"), ("WR", -1.0, 7.0, "lateral_inf"), ("WR", -1.5, -10.0, "pelota"),
        ("TE", -1.0, 5.4, "pelota"), ("WR", -1.5, 10.5, "pelota"), ("WR", -1.0, 7.0, "lateral_sup")],
}

FORMACIONES_DEFENSA: dict[str, list] = {
    "4-3": LINEA_DEFENSIVA_4 + [
        ("LB", 5.0, -4.0, "pelota"), ("LB", 5.5, 0.0, "pelota"), ("LB", 5.0, 4.0, "pelota")] + CORNERS + SAFETIES,
    "3-4": [("DE", 1.0, -3.0, "pelota"), ("NT", 1.0, 0.0, "pelota"), ("DE", 1.0, 3.0, "pelota"),
            ("OLB", 1.5, -6.8, "pelota"), ("ILB", 5.0, -2.0, "pelota"), ("ILB", 5.0, 2.0, "pelota"),
            ("OLB", 1.5, 6.8, "pelota")] + CORNERS + SAFETIES,
    "Nickel (4-2-5)": LINEA_DEFENSIVA_4 + [
        ("LB", 5.0, -2.5, "pelota"), ("LB", 5.0, 2.5, "pelota"), ("NB", 5.0, -8.0, "marca_slot_inf")]
        + CORNERS + SAFETIES,
    "Dime (4-1-6)": LINEA_DEFENSIVA_4 + [
        ("LB", 5.5, 0.0, "pelota"), ("NB", 5.0, -8.0, "marca_slot_inf"), ("NB", 5.0, 8.0, "marca_slot_sup")]
        + CORNERS + SAFETIES,
}

HASHES = {"izquierdo": C.HASH_SUPERIOR, "centro": C.MITAD_ANCHO, "derecho": C.HASH_INFERIOR}
# (el ataque mira hacia la derecha de la pantalla, así que "su izquierda" es arriba)


@dataclass
class Jugador:
    id: str           # identificador único, ej. "A-WR2" o "D-CB1"
    equipo: str       # "ataque" o "defensa"
    posicion: str     # abreviatura: QB, WR, CB, ...
    x: float
    y: float


@dataclass
class Situacion:
    """Todo lo que hace falta para dibujar (y más adelante simular) una jugada antes del snap."""
    jugadores: list[Jugador]
    x_scrimmage: float
    x_primer_down: float
    pelota: tuple[float, float]
    descripcion: str = ""
    extra: dict = field(default_factory=dict)


def _limitar(x: float, y: float) -> tuple[float, float]:
    """Que ningún círculo quede afuera de la cancha."""
    r = C.RADIO_JUGADOR
    return min(max(x, r), C.LARGO - r), min(max(y, r + 0.2), C.ANCHO - r - 0.2)


def _ubicar_ataque(formacion: list, x_los: float, y_pelota: float) -> list[Jugador]:
    jugadores, contador = [], {}
    for pos, dx, dy, ref in formacion:
        y = {"pelota": y_pelota + dy, "lateral_inf": dy, "lateral_sup": C.ANCHO - dy}[ref]
        contador[pos] = contador.get(pos, 0) + 1
        jugadores.append(Jugador(f"A-{pos}{contador[pos]}", "ataque", pos, *_limitar(x_los + dx, y)))
    return jugadores


def _receptores_por_lado(ataque: list[Jugador], y_pelota: float) -> dict[str, list[Jugador]]:
    """Receptores (WR/TE) de cada lado, del más abierto al más cerrado."""
    receptores = [j for j in ataque if j.posicion in ("WR", "TE")]
    inf = sorted([j for j in receptores if j.y < y_pelota], key=lambda j: j.y)
    sup = sorted([j for j in receptores if j.y > y_pelota], key=lambda j: -j.y)
    return {"inf": inf, "sup": sup}


def _ubicar_defensa(formacion: list, x_los: float, y_pelota: float, ataque: list[Jugador]) -> list[Jugador]:
    lados = _receptores_por_lado(ataque, y_pelota)
    jugadores, contador = [], {}
    for pos, dx, dy, ref in formacion:
        y = y_pelota + dy
        if ref.startswith("marca_"):
            _, tipo, lado = ref.split("_")            # ej. "marca_ext_inf"
            indice = 0 if tipo == "ext" else 1
            if len(lados[lado]) > indice:
                y = lados[lado][indice].y
            elif tipo == "ext":                       # sin receptor de ese lado: se abre igual
                y = 10.0 if lado == "inf" else C.ANCHO - 10.0
        x = max(x_los + dx, x_los + C.RADIO_JUGADOR + 0.1)   # nunca en offside
        if pos == "S":
            x = min(x, C.LARGO - 1.5)                 # cerca del gol rival: los safeties quedan en la zona
        contador[pos] = contador.get(pos, 0) + 1
        jugadores.append(Jugador(f"D-{pos}{contador[pos]}", "defensa", pos, *_limitar(x, y)))
    return jugadores


def armar_situacion(formacion_ataque: str, formacion_defensa: str, yarda: int = 25,
                    hash_pelota: str = "centro", yardas_para_primero: int = 10) -> Situacion:
    """
    Arma la situación antes del snap.
      yarda: dónde está la pelota, medida desde la PROPIA línea de gol del ataque (1 a 99).
             25 = "propia 25"; 70 = "rival 30".
      hash_pelota: "izquierdo", "centro" o "derecho" (desde el punto de vista del ataque).
    """
    if not 1 <= yarda <= 99:
        raise ValueError("La yarda tiene que estar entre 1 y 99")
    x_los = C.GOL_IZQ + yarda
    y_pelota = HASHES[hash_pelota]
    ataque = _ubicar_ataque(FORMACIONES_ATAQUE[formacion_ataque], x_los, y_pelota)
    defensa = _ubicar_defensa(FORMACIONES_DEFENSA[formacion_defensa], x_los, y_pelota, ataque)
    x_primero = min(x_los + yardas_para_primero, C.GOL_DER)   # gol para ir: la amarilla es la línea de gol
    lugar = f"propia {yarda}" if yarda < 50 else ("mitad de cancha" if yarda == 50 else f"rival {100 - yarda}")
    gol = " (gol para ir)" if yarda + yardas_para_primero >= 100 else ""
    return Situacion(ataque + defensa, x_los, x_primero, (x_los, y_pelota),
                     f"{formacion_ataque} vs. {formacion_defensa} · pelota en {lugar}, hash {hash_pelota}"
                     f" · {yardas_para_primero} para el primero{gol}")

