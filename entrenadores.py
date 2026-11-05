"""
entrenadores.py — Los DTs que se comparan.

  - DTAleatorio:  elige jugadas al azar (pero patea en 4to down). Es la vara mínima.
  - DTClasico:    reglas "de manual": yardas cortas -> correr; largas -> pasar profundo; 4to -> patear.
                  No mira la defensa. Es el DT "tradicional" contra el que se compara el agente.
  - DTEntrenado:  el agente que aprendió con PPO (models/dt_nfl.zip).

Todos tienen el mismo método: decidir(obs, env) -> acción (vector de 10 números).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import config as C
from juego.jugada import Jugada

NOMBRES = {"aleatorio": "DT al azar", "clasico": "DT clásico (reglas)", "entrenado": "DT con IA (PPO)"}


def _situacion(obs) -> tuple[int, int, int]:
    """Down, distancia y yarda a partir de la observación."""
    down = int(np.argmax(obs[0:4])) + 1
    return down, int(round(obs[4] * 10)), int(round(obs[5] * 100))


def _patea_o_va(down: int, dist: int, yarda: int) -> bool:
    """¿Conviene patear en 4to? (regla clásica: solo se va por ella con 1 yarda y lejos de casa)."""
    return down == 4 and not (dist <= 1 and yarda >= 40)


class DTAleatorio:
    nombre = "aleatorio"

    def __init__(self, semilla: int | None = None):
        self.rng = np.random.default_rng(semilla)

    def decidir(self, obs, env=None) -> np.ndarray:
        down, dist, yarda = _situacion(obs)
        accion = np.array([self.rng.integers(n) for n in env.action_space.nvec])
        accion[1] = 2 if down == 4 else self.rng.integers(2)
        return accion


class DTClasico:
    nombre = "clasico"

    def __init__(self, semilla: int | None = None):
        self.rng = np.random.default_rng(semilla)

    def decidir(self, obs, env=None) -> np.ndarray:
        down, dist, yarda = _situacion(obs)
        r = self.rng.random()
        if _patea_o_va(down, dist, yarda):
            j = Jugada("Formación I", "patear", ["bloqueo"] * 5, 0, 0, 2)
        elif dist <= 3 or (down == 1 and r < 0.5):
            # carrera con 2 alas cerradas, por adentro
            j = Jugada("Singleback (2 alas cerradas)", "carrera", ["bloqueo"] * 5, 0, 0, int(self.rng.integers(1, 4)))
        elif dist <= 7:
            # pase rápido: slants y flats.  Elegibles Shotgun: RB, TE, WR abajo, WR slot, WR arriba
            j = Jugada("Shotgun (3 receptores)", "pase", ["bloqueo", "out", "slant", "flat", "slant"],
                       int(self.rng.choice([2, 4])), 0, 2)
        else:
            # yardas largas: rutas intermedias y profundas
            j = Jugada("Shotgun (3 receptores)", "pase", ["bloqueo", "dig", "comeback", "dig", "post"],
                       int(self.rng.choice([1, 2, 4])), 1, 2)
        return j.a_accion()


class DTEntrenado:
    nombre = "entrenado"

    def __init__(self, ruta_modelo: str | Path | None = None, modelo=None):
        if modelo is None:
            from stable_baselines3 import PPO
            ruta = Path(ruta_modelo or C.MODELO_POR_DEFECTO)
            if not ruta.exists():
                raise FileNotFoundError(f"No hay modelo entrenado en '{ruta}'. Entrenalo con:  python train.py")
            modelo = PPO.load(str(ruta), device="cpu")
        self.modelo = modelo

    def decidir(self, obs, env=None) -> np.ndarray:
        accion, _ = self.modelo.predict(obs, deterministic=True)
        return np.asarray(accion)


def crear_dt(nombre: str, ruta_modelo=None, semilla: int | None = None):
    if nombre == "aleatorio":
        return DTAleatorio(semilla)
    if nombre == "clasico":
        return DTClasico(semilla)
    if nombre == "entrenado":
        return DTEntrenado(ruta_modelo)
    raise ValueError(f"DT desconocido: {nombre}")


def jugar_drive(dt, env, semilla: int, opciones: dict | None = None) -> dict:
    """Juega un drive completo y devuelve el resumen (puntos, motivo, yardas, etc.)."""
    obs, _ = env.reset(seed=semilla, options=opciones)
    while True:
        obs, _, terminado, truncado, info = env.step(dt.decidir(obs, env))
        if terminado or truncado:
            return info["drive"]
