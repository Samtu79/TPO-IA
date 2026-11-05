"""
juego/simulador.py — Simula UNA jugada, desde el snap hasta que la pelota queda muerta.

Cada paso son 0,1 segundos. En cada paso:
  - los receptores corren su ruta;
  - la línea ofensiva (y los que se quedan a bloquear) frena a los que van a buscar al QB;
  - la defensa cubre según su asignación (marca personal o zona) y reacciona al pase;
  - el QB lanza en el momento elegido (o antes si lo apuran); si lo alcanzan antes: sack;
  - con la pelota en el aire, se decide si es completo, incompleto o intercepción según la
    separación del defensor más cercano y la distancia del pase;
  - el que tiene la pelota corre esquivando defensores hasta que lo tacklean, sale de la
    cancha o anota.

Hay azar a propósito (velocidades, bloqueos, recepciones, tackles): la misma jugada no
siempre sale igual, como en la realidad. Con la misma semilla, sí sale igual.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import config as C
from juego.defensa import asignar_cobertura
from juego.jugada import DESPLAZAMIENTO_HUECOS, HUECOS, Jugada
from juego.jugadores import Situacion
from juego.rutas import camino_absoluto, termina_parado

LINEA_OFENSIVA = ("C", "G", "T")


@dataclass
class ResultadoJugada:
    tipo: str                 # pase_completo, pase_incompleto, intercepcion, sack, carrera
    x_final: float            # dónde quedó la pelota (coordenada x de la cancha)
    y_final: float
    yardas: float             # ganancia respecto de la línea de scrimmage
    touchdown: bool = False
    descripcion: str = ""
    rushers: int = 0          # cuántos defensores fueron a buscar al QB (para "leer" al rival)
    x_recepcion: float | None = None   # dónde se recibió el pase (para separar yardas en el aire / después)
    cuadros: list = field(default_factory=list)   # posiciones de cada paso (solo si grabar=True)
    caminos: dict = field(default_factory=dict)   # ruta planeada de cada receptor (para dibujar)


def _unitario(dx: float, dy: float) -> tuple[float, float]:
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else (0.0, 0.0)


class _Jugada:
    """Estado interno de una jugada en curso."""

    def __init__(self, jugada: Jugada, sit: Situacion, cobertura: str, rng: np.random.Generator, grabar: bool):
        self.j, self.sit, self.rng, self.grabar = jugada, sit, rng, grabar
        self.J = sit.jugadores
        self.n = len(self.J)
        self.P = np.array([[p.x, p.y] for p in self.J], dtype=float)
        self.pos = [p.posicion for p in self.J]
        self.vmax = np.array([C.VELOCIDADES[p] * (1 + rng.normal(0, C.RUIDO_VELOCIDAD)) for p in self.pos])
        self.acel = np.array([C.ACELERACIONES[p] for p in self.pos])
        self.V = np.zeros_like(self.P)        # velocidad actual de cada jugador (yd/s): hay inercia
        self.x_los, self.y_pelota = sit.x_scrimmage, sit.pelota[1]
        self.ataque = [i for i, p in enumerate(self.J) if p.equipo == "ataque"]
        self.defensa = [i for i, p in enumerate(self.J) if p.equipo == "defensa"]
        self.qb = next(i for i in self.ataque if self.pos[i] == "QB")
        self.elegibles = [i for i in self.ataque if self.pos[i] in ("WR", "TE", "RB", "FB")]
        self.asig = asignar_cobertura(cobertura, self.J, self.x_los, self.y_pelota)
        self.t = 0.0
        self.historial: list[np.ndarray] = []
        self.cuadros: list[dict] = []
        self.pelota = "qb"                    # "qb", "aire" o el índice del portador
        self.vel_portador = np.zeros(2)
        self.trabados: dict[int, tuple[int, float, float]] = {}   # defensor -> (bloqueador, velocidad, hasta)
        self.caminos: dict[int, np.ndarray] = {}
        self.avance_ruta: dict[int, int] = {}
        self.x_recepcion: float | None = None

    # ---------------- utilidades ----------------
    def dist(self, a: int, b: int) -> float:
        return math.hypot(self.P[a, 0] - self.P[b, 0], self.P[a, 1] - self.P[b, 1])

    def grabar_cuadro(self, texto: str = "", pelota_xy=None):
        if not self.grabar:
            return
        if pelota_xy is None:
            i = self.qb if self.pelota == "qb" else self.pelota
            pelota_xy = tuple(self.P[i])
        self.cuadros.append({"t": round(self.t, 2), "P": self.P.copy(), "pelota": pelota_xy, "texto": texto})

    def mover(self, objetivos: np.ndarray, velocidades: np.ndarray):
        """
        Mueve a todos a la vez hacia su objetivo. Hay INERCIA: nadie cambia de dirección ni
        frena de golpe, cada uno acelera según su posición. Así aparece la separación en los
        quiebres de las rutas (el receptor sabe a dónde va; el defensor tiene que reaccionar).
        """
        d = objetivos - self.P
        largo = np.linalg.norm(d, axis=1)
        rapidez = np.minimum(velocidades, largo * C.GANANCIA_LLEGADA)      # frena al llegar
        deseada = np.where(largo[:, None] > 1e-9, d / np.maximum(largo, 1e-9)[:, None] * rapidez[:, None], 0.0)
        cambio = deseada - self.V
        n = np.linalg.norm(cambio, axis=1)
        factor = np.minimum(1.0, self.acel * C.DT_JUGADA / np.maximum(n, 1e-9))
        self.V += cambio * factor[:, None]
        self.P += self.V * C.DT_JUGADA

    # ---------------- rutas ----------------
    def preparar_rutas(self):
        for k, i in enumerate(self.elegibles):
            ruta = self.j.rutas[k]
            if ruta != "bloqueo":
                self.caminos[i] = camino_absoluto(ruta, self.P[i, 0], self.P[i, 1], self.y_pelota)
                self.avance_ruta[i] = 1

    def objetivo_ruta(self, i: int, P: np.ndarray, avance: dict) -> tuple[float, float]:
        """Hacia dónde corre un receptor en su ruta (y avanza el punto de paso si llegó)."""
        camino = self.caminos[i]
        k = avance[i]
        while k < len(camino) and math.hypot(*(camino[k] - P[i])) < 0.6:
            k += 1
        avance[i] = k
        if k < len(camino):
            return tuple(camino[k])
        ruta = self.j.rutas[self.elegibles.index(i)]
        if termina_parado(ruta):
            return tuple(P[i])
        dx, dy = _unitario(*(camino[-1] - camino[-2]))
        return P[i, 0] + 10 * dx, float(np.clip(P[i, 1] + 10 * dy, 1.0, C.ANCHO - 1.0))

    def posicion_futura(self, i: int, segundos: float) -> np.ndarray:
        """Dónde va a estar el receptor i dentro de 'segundos' si sigue su ruta (el QB le lanza ahí)."""
        P = self.P.copy()
        p, v = P[i].copy(), self.V[i].copy()
        avance = dict(self.avance_ruta)
        for _ in range(int(round(segundos / C.DT_JUGADA))):
            P[i] = p
            ox, oy = self.objetivo_ruta(i, P, avance)
            d = np.array([ox, oy]) - p
            largo = float(np.linalg.norm(d))
            rapidez = min(self.vmax[i], largo * C.GANANCIA_LLEGADA)
            deseada = d / largo * rapidez if largo > 1e-9 else np.zeros(2)
            cambio = deseada - v
            n = float(np.linalg.norm(cambio))
            v += cambio * min(1.0, self.acel[i] * C.DT_JUGADA / max(n, 1e-9))
            p += v * C.DT_JUGADA
        return p

    # ---------------- bloqueos ----------------
    def armar_proteccion(self):
        """Pase: cada defensor que presiona queda trabado por el bloqueador libre más cercano."""
        bloqueadores = [i for i in self.ataque if self.pos[i] in LINEA_OFENSIVA]
        bloqueadores += [i for k, i in enumerate(self.elegibles) if self.j.rutas[k] == "bloqueo"]
        rushers = [d for d in self.defensa if self.asig[d][0] == "presion"]
        rushers.sort(key=lambda d: self.dist(d, self.qb))
        libres = list(bloqueadores)
        for d in rushers:
            if not libres:
                break
            b = min(libres, key=lambda b: self.dist(b, d))
            self.trabados[d] = (b, self.rng.uniform(*C.VEL_RUSHER_BLOQUEADO), math.inf)
            libres.remove(b)
        self.bloqueadores_libres = libres

    def armar_bloqueo_carrera(self, portador: int):
        """Carrera: todos los de ataque (menos el corredor y el QB) bloquean al defensor más cercano."""
        bloqueadores = [i for i in self.ataque if i not in (portador, self.qb)]
        libres_def = list(self.defensa)
        for b in sorted(bloqueadores, key=lambda b: min(self.dist(b, d) for d in self.defensa)):
            if not libres_def:
                break
            d = min(libres_def, key=lambda d: self.dist(b, d))
            if self.dist(b, d) < 9:
                hasta = self.t + self.rng.uniform(*C.DURACION_BLOQUEO_CARRERA)
                self.trabados[d] = (b, C.VEL_DEFENSOR_BLOQUEADO, hasta)
                libres_def.remove(d)

    # ---------------- comportamiento ----------------
    def objetivo_defensor_cobertura(self, d: int) -> tuple[float, float]:
        tipo = self.asig[d][0]
        if tipo == "marca":
            r = self.asig[d][1]
            atraso = int(round(C.REACCION_MARCA / C.DT_JUGADA))
            ref = self.historial[-atraso] if len(self.historial) >= atraso else self.historial[0]
            return ref[r, 0] + 0.8, ref[r, 1]               # se pone un poco "por arriba" del receptor
        if tipo == "zona":
            lx, ly = self.asig[d][1]
            if lx - self.x_los >= 12:
                # Zona PROFUNDA: nadie le pasa por detrás. Acompaña al receptor más profundo de su
                # sector, manteniéndose un par de yardas por arriba (más lejos del QB que él).
                sector = [r for r in self.caminos if abs(self.P[r, 1] - ly) < C.ANCHO_SECTOR_PROFUNDO
                          and self.P[r, 0] > lx - C.RADIO_ZONA]
                if sector:
                    r = max(sector, key=lambda r: self.P[r, 0])
                    return max(lx, self.P[r, 0] + 2.0), 0.5 * (self.P[r, 1] + ly)
                return lx, ly
            cerca =[r for r in self.caminos if math.hypot(self.P[r, 0] - lx, self.P[r, 1] - ly) < C.RADIO_ZONA]
            if cerca:
                r = min(cerca, key=lambda r: math.hypot(self.P[r, 0] - lx, self.P[r, 1] - ly))
                return self.P[r, 0], self.P[r, 1]
            return lx, ly
        return self.P[self.qb, 0], self.P[self.qb, 1]

    def objetivo_persecucion(self, d: int, portador: int) -> tuple[float, float]:
        """Ir a cortar al que tiene la pelota (apuntando un poco adelante de donde está)."""
        dist = self.dist(d, portador)
        adelanto = min(dist / max(self.vmax[d], 1e-6), 1.0)
        return self.P[portador, 0] + self.vel_portador[0] * adelanto, self.P[portador, 1] + self.vel_portador[1] * adelanto

    def objetivo_corredor(self, i: int, hueco_xy=None) -> tuple[float, float]:
        """El corredor va hacia adelante, esquivando a los defensores libres que tiene enfrente."""
        x, y = self.P[i]
        if hueco_xy is not None and x < self.x_los - 0.3:
            return hueco_xy
        ux, uy = 1.0, 0.0
        for d in self.defensa:
            if d in self.trabados and self.trabados[d][2] > self.t:
                continue
            rx, ry = self.P[d, 0] - x, self.P[d, 1] - y
            dist = math.hypot(rx, ry)
            if dist < 7 and rx > -1:
                fuerza = 1.6 / max(dist, 0.8)
                ux -= rx / max(dist, 1e-6) * fuerza * 0.3
                uy -= ry / max(dist, 1e-6) * fuerza
        if y < 3:
            uy += 0.8
        elif y > C.ANCHO - 3:
            uy -= 0.8
        ux = max(ux, 0.25)                       # nunca hacia atrás
        ux, uy = _unitario(ux, uy)
        return x + 5 * ux, y + 5 * uy

    # ---------------- final ----------------
    def resultado(self, tipo: str, x: float, y: float, texto: str, td: bool = False) -> ResultadoJugada:
        self.grabar_cuadro(texto)
        rushers = sum(1 for d in self.defensa if self.asig[d][0] == "presion")
        return ResultadoJugada(tipo=tipo, x_final=x, y_final=y, yardas=x - self.x_los, touchdown=td,
                               descripcion=texto, rushers=rushers, x_recepcion=self.x_recepcion,
                               cuadros=self.cuadros, caminos=dict(self.caminos))


def _etiqueta(sim: _Jugada, i: int) -> str:
    """Ej. 'WR2' o 'RB' (con número solo si hay más de uno de esa posición)."""
    pos = sim.pos[i]
    iguales = [k for k in sim.ataque if sim.pos[k] == pos]
    return f"{pos}{iguales.index(i) + 1}" if len(iguales) > 1 else pos


def _fase_portador(sim: _Jugada, portador: int, texto_base: str, tipo: str, hueco_xy=None,
                   en_carrera: bool = False) -> ResultadoJugada:
    """El que tiene la pelota corre hasta que lo tacklean, sale de la cancha, anota o se acaba el tiempo."""
    sim.pelota = portador
    inicio = sim.t
    lectura = sim.t + C.LECTURA_CARRERA   # en una carrera, la defensa tarda un poco en "leer" la jugada
    while sim.t < C.DURACION_MAX_JUGADA:
        sim.historial.append(sim.P.copy())
        objetivos = sim.P.copy()
        vel = sim.vmax.copy()
        ox, oy = sim.objetivo_corredor(portador, hueco_xy)
        objetivos[portador] = (ox, oy)
        if tipo == "pase_completo":       # después de recibir hay que darse vuelta y arrancar
            arranque = min(1.0, 0.3 + 0.7 * (sim.t - inicio) / C.ARRANQUE_CON_PELOTA)
            vel[portador] *= C.VEL_CON_PELOTA * arranque
        for d in sim.defensa:
            trabado = sim.trabados.get(d)
            if trabado and trabado[2] > sim.t:
                b, v, _ = trabado
                objetivos[d] = sim.objetivo_persecucion(d, portador)
                vel[d] = v
                ux, uy = _unitario(sim.P[portador, 0] - sim.P[d, 0], sim.P[portador, 1] - sim.P[d, 1])
                objetivos[b] = (sim.P[d, 0] + 1.7 * ux, sim.P[d, 1] + 1.7 * uy)   # el bloqueador se pone adelante
            elif en_carrera and sim.t < lectura and sim.asig[d][0] != "presion":
                objetivos[d] = sim.objetivo_defensor_cobertura(d)
            else:
                objetivos[d] = sim.objetivo_persecucion(d, portador)
        for a in sim.ataque:
            if a != portador and not any(t[0] == a and t[2] > sim.t for t in sim.trabados.values()):
                objetivos[a] = (sim.P[portador, 0] + 3, sim.P[a, 1])            # acompaña la jugada
                vel[a] *= 0.5
        antes = sim.P[portador].copy()
        sim.mover(objetivos, vel)
        sim.vel_portador = (sim.P[portador] - antes) / C.DT_JUGADA
        sim.t += C.DT_JUGADA
        x, y = sim.P[portador]
        if x >= C.GOL_DER:
            return sim.resultado(tipo, x, y, f"{texto_base} ¡TOUCHDOWN!", td=True)
        if y <= 0 or y >= C.ANCHO:
            return sim.resultado(tipo, x, y, f"{texto_base} sale por el costado: {x - sim.x_los:+.0f} yardas")
        for d in sim.defensa:
            trabado = sim.trabados.get(d)
            if trabado and trabado[2] > sim.t:
                continue
            if sim.dist(d, portador) < C.DISTANCIA_TACKLE and sim.rng.random() < C.PROB_TACKLE_POR_PASO:
                return sim.resultado(tipo, x, y, f"{texto_base} tackleado: {x - sim.x_los:+.0f} yardas")
        sim.grabar_cuadro(texto_base)
    x, y = sim.P[portador]
    return sim.resultado(tipo, x, y, f"{texto_base} {x - sim.x_los:+.0f} yardas")


def _simular_carrera(sim: _Jugada) -> ResultadoJugada:
    rbs = [i for i in sim.ataque if sim.pos[i] == "RB"]
    portador = rbs[0] if rbs else sim.qb
    hueco_xy = (sim.x_los + 0.5, sim.y_pelota + DESPLAZAMIENTO_HUECOS[sim.j.hueco])
    sim.grabar_cuadro("Snap")
    # entrega: el corredor recibe la pelota apenas arranca
    sim.t += C.DT_JUGADA * 3
    sim.armar_bloqueo_carrera(portador)
    texto = f"Carrera del {_etiqueta(sim, portador)} {HUECOS[sim.j.hueco]}:"
    return _fase_portador(sim, portador, texto, "carrera", hueco_xy, en_carrera=True)


def _simular_pase(sim: _Jugada) -> ResultadoJugada:
    sim.preparar_rutas()
    # si el receptor que marcaba un defensor se queda a bloquear, ese defensor va a presionar
    for d in sim.defensa:
        if sim.asig[d][0] == "marca" and sim.asig[d][1] not in sim.caminos:
            sim.asig[d] = ("presion",)
    sim.armar_proteccion()
    bajo_centro = sim.P[sim.qb, 0] > sim.x_los - 3.5
    retroceso = (sim.x_los - 7.0 if bajo_centro else sim.x_los - 6.5, sim.y_pelota)
    objetivo = sim.elegibles[sim.j.objetivo]
    momento = C.TIEMPOS_LANZAMIENTO[sim.j.momento]
    sim.grabar_cuadro("Snap")

    # ---- antes del pase ----
    apurado = False
    while True:
        sim.historial.append(sim.P.copy())
        objetivos = sim.P.copy()
        vel = sim.vmax.copy()
        objetivos[sim.qb] = retroceso
        for i in sim.caminos:
            objetivos[i] = sim.objetivo_ruta(i, sim.P, sim.avance_ruta)
        for d in sim.defensa:
            objetivos[d] = sim.objetivo_defensor_cobertura(d)
            if d in sim.trabados:
                b, v, _ = sim.trabados[d]
                vel[d] = v
                ux, uy = _unitario(sim.P[sim.qb, 0] - sim.P[d, 0], sim.P[sim.qb, 1] - sim.P[d, 1])
                objetivos[b] = (sim.P[d, 0] + 1.7 * ux, sim.P[d, 1] + 1.7 * uy)
        for b in sim.bloqueadores_libres:
            objetivos[b] = (sim.x_los - 2.0, sim.P[b, 1])
        sim.mover(objetivos, vel)
        sim.t += C.DT_JUGADA
        for d in list(sim.trabados):                     # a veces el rusher le gana al bloqueador
            if sim.rng.random() < C.PROB_GANAR_BLOQUEO:
                sim.bloqueadores_libres.append(sim.trabados.pop(d)[0])

        presion = min((sim.dist(d, sim.qb) for d in sim.defensa if sim.asig[d][0] == "presion"), default=99)
        if presion < C.DISTANCIA_SACK:
            x, y = sim.P[sim.qb]
            return sim.resultado("sack", x, y, f"¡SACK! {x - sim.x_los:+.0f} yardas")
        apuran = presion < C.PRESION_LANZA_YA and sim.t >= 0.6 and sim.rng.random() < C.PROB_LANZAR_BAJO_PRESION
        if sim.t >= momento or apuran:
            apurado = sim.t < momento
            break
        sim.grabar_cuadro("")

    # ---- el pase: se lanza adelante del receptor, a donde va a estar ----
    origen = sim.P[sim.qb].copy()
    destino = sim.P[objetivo].copy()
    for _ in range(3):
        vuelo = math.hypot(*(destino - origen)) / C.VELOCIDAD_PASE
        destino = sim.posicion_futura(objetivo, vuelo) if objetivo in sim.caminos else sim.P[objetivo].copy()
    vuelo = max(math.hypot(*(destino - origen)) / C.VELOCIDAD_PASE, C.DT_JUGADA)
    t_lanzamiento, llegada = sim.t, sim.t + vuelo
    receptor = _etiqueta(sim, objetivo)
    ruta = sim.j.rutas[sim.j.objetivo]
    texto = f"Pase al {receptor} ({ruta})" + (" apurado" if apurado else "")

    # ---- la pelota en el aire ----
    while sim.t < llegada - 1e-9:
        sim.historial.append(sim.P.copy())
        objetivos = sim.P.copy()
        vel = sim.vmax.copy()
        for i in sim.caminos:
            objetivos[i] = sim.objetivo_ruta(i, sim.P, sim.avance_ruta)
        reaccionan = sim.t - t_lanzamiento >= C.REACCION_PELOTA
        for d in sim.defensa:
            if sim.asig[d][0] == "presion" or not reaccionan:
                objetivos[d] = sim.objetivo_defensor_cobertura(d)
            else:
                objetivos[d] = destino                       # todos van a la pelota
                vel[d] *= C.VEL_IR_A_LA_PELOTA
            if d in sim.trabados:
                vel[d] = sim.trabados[d][1]
        sim.mover(objetivos, vel)
        sim.t += C.DT_JUGADA
        f = min((sim.t - t_lanzamiento) / vuelo, 1.0)
        sim.grabar_cuadro(texto, pelota_xy=tuple(origen + (destino - origen) * f))

    # ---- ¿completo, incompleto o intercepción? ----
    cx, cy = destino
    if not (0 < cy < C.ANCHO) or cx > C.LARGO:
        return sim.resultado("pase_incompleto", sim.x_los, sim.y_pelota, f"{texto}: incompleto (afuera)")
    if math.hypot(*(sim.P[objetivo] - destino)) > 2.0:
        return sim.resultado("pase_incompleto", sim.x_los, sim.y_pelota, f"{texto}: incompleto (no llegó)")
    separacion = min(math.hypot(sim.P[d, 0] - cx, sim.P[d, 1] - cy) for d in sim.defensa)
    largo = math.hypot(cx - origen[0], cy - origen[1])
    p_comp = float(np.clip(C.COMPLETO_BASE - C.COMPLETO_CAIDA_POR_YARDA * max(largo - 5, 0), 0.3, 0.95))
    if separacion < C.SEPARACION_LIBRE:
        p_comp *= C.COMPLETO_PEGADO + (1 - C.COMPLETO_PEGADO) * separacion / C.SEPARACION_LIBRE
    if apurado:
        p_comp *= C.PRECISION_BAJO_PRESION
    p_int = C.INTERCEPCION_MAX * max(0.0, 1 - separacion / 1.5) +         C.INTERCEPCION_POR_YARDA * largo * (1.0 if separacion < 2.5 else 0.3)
    u = sim.rng.random()
    if u < p_int:
        return sim.resultado("intercepcion", cx, cy, f"{texto}: ¡INTERCEPTADO!")
    if u >= p_int + p_comp:
        return sim.resultado("pase_incompleto", sim.x_los, sim.y_pelota, f"{texto}: incompleto")
    if cx >= C.GOL_DER:
        sim.pelota = objetivo
        return sim.resultado("pase_completo", cx, cy, f"{texto}: completo en la zona de anotación. ¡TOUCHDOWN!", td=True)
    sim.trabados.clear()
    sim.x_recepcion = cx
    return _fase_portador(sim, objetivo, f"{texto}: completo,", "pase_completo")


def simular_jugada(jugada: Jugada, situacion: Situacion, cobertura: str,
                   rng: np.random.Generator | None = None, grabar: bool = False) -> ResultadoJugada:
    """
    Simula una jugada de pase o carrera (las patadas se resuelven en juego/drive.py).
    situacion: posiciones antes del snap (de juego.jugadores.armar_situacion).
    grabar=True guarda un "cuadro" por paso para poder animarla.
    """
    if jugada.tipo not in ("pase", "carrera"):
        raise ValueError("simular_jugada solo simula pases y carreras")
    sim = _Jugada(jugada, situacion, cobertura, rng or np.random.default_rng(), grabar)
    return _simular_pase(sim) if jugada.tipo == "pase" else _simular_carrera(sim)
