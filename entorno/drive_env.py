"""
entorno/drive_env.py — Entorno Gymnasium: el agente es el DT (coordinador ofensivo).

  EPISODIO = un drive (de la primera jugada hasta anotar, perder la pelota o patear).
  PASO     = una jugada: el DT elige formación, tipo, rutas, a quién pasa, cuándo y por dónde corre.

Lo que el DT VE antes de cada jugada (observación, 19 números):
  - down (1 a 4), yardas para el primero, yarda de la cancha, hash;
  - el personal defensivo (4-3, 3-4, Nickel, Dime);
  - cuántos safeties muestra profundos (0, 1 o 2): una PISTA de la cobertura, que puede ser un disfraz;
  - cuántos defensores hay "en la caja" (cerca de la pelota);
  - cuánto viene haciendo blitz este rival en el drive (tendencia);
  - cuántas jugadas lleva el drive.
Lo que NO ve: la cobertura real ni el perfil del rival -> entorno parcialmente observable.

Recompensa: EPA (cuántos puntos esperados agregó la jugada) + extras opcionales de config.py.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import config as C
from juego.defensa import alinear_safeties, cantidad_profundos, elegir_defensa
from juego.drive import EstadoDrive, aplicar_jugada, patear, valor_estado
from juego.jugada import ESPACIO_ACCION, Jugada
from juego.jugadores import FORMACIONES_DEFENSA, HASHES, armar_situacion
from juego.simulador import simular_jugada

PERSONALES = list(FORMACIONES_DEFENSA)
HASH_LISTA = list(HASHES)
DIM_OBS = 19
FORMACION_REFERENCIA = "Shotgun (3 receptores)"   # para "ver" la defensa antes de elegir formación


def armar_observacion(e: EstadoDrive, personal: str, profundos_visibles: int, en_caja: int,
                      tendencia_blitz: float) -> np.ndarray:
    """Vector de 19 números (todos chicos) que describe la situación antes del snap."""
    o = np.zeros(DIM_OBS, dtype=np.float32)
    o[e.down - 1] = 1.0                                  # [0:4] down
    o[4] = min(e.distancia, 20) / 10                     # [4]   yardas para el primero
    o[5] = e.yarda / 100                                 # [5]   dónde está la pelota
    o[6 + HASH_LISTA.index(e.hash)] = 1.0                # [6:9] hash
    o[9 + PERSONALES.index(personal)] = 1.0              # [9:13] personal defensivo
    o[13 + profundos_visibles] = 1.0                     # [13:16] safeties profundos que se ven
    o[16] = en_caja / 11                                 # [16]  defensores en la caja
    o[17] = tendencia_blitz                              # [17]  % de blitz del rival en este drive
    o[18] = min(e.jugadas, 20) / 10                      # [18]  jugadas del drive
    return o


def situacion_snap(formacion: str, personal: str, mostrada: str, e: EstadoDrive):
    """Posiciones antes del snap: la formación ofensiva contra el personal defensivo, con los safeties
    parados según la cobertura que la defensa MUESTRA."""
    sit = armar_situacion(formacion, personal, e.yarda, e.hash, e.distancia)
    defensores = [j for j in sit.jugadores if j.equipo == "defensa"]
    alinear_safeties(defensores, mostrada, sit.x_scrimmage, sit.pelota[1])
    return sit


class DriveEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, rival: str = "mezcla", grabar: bool = False):
        """
        rival: "equilibrado", "agresivo", "conservador" o "mezcla" (cada drive, uno al azar).
        grabar: guarda los cuadros de cada jugada para animarlas (solo para la app; más lento).
        """
        super().__init__()
        if rival != "mezcla" and rival not in C.PERFILES_RIVAL:
            raise ValueError(f"Rival desconocido: {rival}")
        self.rival, self.grabar = rival, grabar
        self.action_space = spaces.MultiDiscrete(ESPACIO_ACCION)
        self.observation_space = spaces.Box(low=0.0, high=2.5, shape=(DIM_OBS,), dtype=np.float32)
        self.estado = EstadoDrive()

    # ------------------------------------------------------------------
    def _preparar_defensa(self):
        """El rival elige personal y cobertura, y decide si 'disfraza' lo que muestra."""
        e = self.estado
        personal, cobertura = elegir_defensa(e.down, e.distancia, self.perfil, self.np_random)
        mostrada = cobertura
        if self.np_random.random() < C.PROB_DISFRAZ:
            otras = [c for c in C.COBERTURAS if cantidad_profundos(c) != cantidad_profundos(cobertura)]
            mostrada = otras[self.np_random.integers(len(otras))]
        self.defensa = {"personal": personal, "cobertura": cobertura, "mostrada": mostrada}
        ref = self.situacion(FORMACION_REFERENCIA)
        y_pelota = ref.pelota[1]
        self.en_caja = sum(1 for j in ref.jugadores if j.equipo == "defensa"
                           and j.x - ref.x_scrimmage <= 7 and abs(j.y - y_pelota) <= 8)

    def situacion(self, formacion: str):
        """Posiciones antes del snap para la formación ofensiva elegida y la defensa actual."""
        return situacion_snap(formacion, self.defensa["personal"], self.defensa["mostrada"], self.estado)

    def _obs(self) -> np.ndarray:
        tendencia = float(np.mean(self.blitzes)) if self.blitzes else 0.0
        return armar_observacion(self.estado, self.defensa["personal"],
                                 cantidad_profundos(self.defensa["mostrada"]), self.en_caja, tendencia)

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        perfiles = list(C.PERFILES_RIVAL)
        self.perfil = options.get("rival") or (self.rival if self.rival != "mezcla"
                                               else perfiles[self.np_random.integers(len(perfiles))])
        yarda = options.get("yarda") or int(self.np_random.integers(C.YARDA_INICIO_DRIVE[0], C.YARDA_INICIO_DRIVE[1] + 1))
        self.estado = EstadoDrive(int(yarda), 1, 10, HASH_LISTA[self.np_random.integers(3)], 0)
        self.blitzes: list[bool] = []
        self.stats = {"yardas": 0.0, "primeros_downs": 0, "sacks": 0, "intercepciones": 0, "pases": 0,
                      "completos": 0, "carreras": 0, "valor_inicial": valor_estado(self.estado)}
        self._preparar_defensa()
        return self._obs(), {"perfil_rival": self.perfil}

    def step(self, accion):
        jugada = Jugada.desde_accion(accion)
        e = self.estado
        v0 = valor_estado(e)
        info = {"antes": e.describir(), "jugada": jugada.describir(), **self.defensa, "perfil_rival": self.perfil}
        R = C.RECOMPENSA_EXTRA
        extra = R["jugada"]
        resultado = None

        if jugada.tipo == "patear":
            des = patear(e, self.np_random)
        else:
            sit = self.situacion(jugada.formacion)
            resultado = simular_jugada(jugada, sit, self.defensa["cobertura"], self.np_random, self.grabar)
            des = aplicar_jugada(e, resultado)
            self.blitzes.append(resultado.rushers >= 5)
            s = self.stats
            if jugada.tipo == "pase":
                s["pases"] += 1
                s["completos"] += resultado.tipo == "pase_completo"
            else:
                s["carreras"] += 1
            s["sacks"] += resultado.tipo == "sack"
            s["intercepciones"] += resultado.tipo == "intercepcion"
            s["yardas"] += 0.0 if resultado.tipo in ("pase_incompleto", "intercepcion") else resultado.yardas
            s["primeros_downs"] += des.primer_down
            extra += (R["sack"] * (resultado.tipo == "sack") + R["intercepcion"] * (resultado.tipo == "intercepcion")
                      + R["primer_down"] * des.primer_down)
            info["resultado"] = {"tipo": resultado.tipo, "yardas": round(resultado.yardas, 1),
                                 "rushers": resultado.rushers, "touchdown": resultado.touchdown}
            if self.grabar:                      # para animar la jugada en la app
                info["situacion"], info["simulacion"] = sit, resultado

        if des.terminado:
            recompensa = des.valor_final - v0
            extra += R["td"] * (des.motivo == "touchdown")
        else:
            recompensa = valor_estado(des.estado) - v0
        info["texto"] = (resultado.descripcion + " → " if resultado else "") + des.texto

        terminado = des.terminado
        truncado = not terminado and des.estado.jugadas >= C.MAX_JUGADAS_DRIVE
        if terminado or truncado:
            info["drive"] = {"puntos": des.puntos, "motivo": des.motivo or "limite_jugadas",
                             "jugadas": e.jugadas + 1, "perfil_rival": self.perfil, **self.stats}
        else:
            self.estado = des.estado
            self._preparar_defensa()
        return self._obs(), float(recompensa + extra), terminado, truncado, info
