"""
train.py — Entrena al DT con PPO (aprendizaje por refuerzo profundo).

Ejemplos:
    python train.py                                  # 300.000 jugadas contra rivales mezclados
    python train.py --timesteps 1000000              # más entrenamiento
    python train.py --rival agresivo                 # que se especialice contra un rival que hace mucho blitz
    python train.py --continuar models/dt_nfl.zip    # seguir entrenando un modelo ya guardado

Cada "timestep" es UNA jugada. Mientras entrena muestra cuántos puntos por drive consigue:
si aprende, ese número sube.

¿Por qué aprendizaje por refuerzo? Nadie le dice qué jugada es la correcta: solo sabe cuántos
puntos esperados ganó o perdió con cada jugada (EPA). Por prueba y error, en cientos de miles de
jugadas simuladas, descubre qué jugadas funcionan en cada situación y contra cada defensa.

¿Por qué PPO? Sirve para acciones compuestas (la acción son 10 elecciones a la vez: formación,
tipo, 5 rutas, a quién pasa, cuándo y por dónde corre), es estable y corre bien en CPU.

¿Por qué una red neuronal? La situación combina números continuos (yarda, distancia) con
categorías (personal, safeties): una tabla tendría millones de casillas. La red generaliza
a situaciones parecidas que nunca vio exactas.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import config as C
from entorno.drive_env import DriveEnv

ARCHIVO_CURVA = C.CARPETA_RESULTADOS / "curva_aprendizaje.csv"
COLUMNAS = ["timesteps", "drives", "puntos_por_drive", "pct_td", "pct_fg", "pct_perdidas", "yardas_por_jugada",
            "intercepciones_por_drive", "sacks_por_drive", "pct_pases", "minutos"]


class CurvaAprendizaje(BaseCallback):
    """Después de cada actualización de PPO, guarda cómo le fue en los drives que terminaron."""

    def __init__(self, inicio: float):
        super().__init__()
        self.inicio = inicio
        self.drives: list[dict] = []

    def _on_step(self) -> bool:
        for hecho, info in zip(self.locals["dones"], self.locals["infos"]):
            if hecho and "drive" in info:
                self.drives.append(info["drive"])
        return True

    def _on_rollout_end(self) -> None:
        ds, self.drives = self.drives, []
        if not ds:
            return
        motivos = Counter(d["motivo"] for d in ds)
        jugadas = sum(d["jugadas"] for d in ds)
        pases = sum(d["pases"] for d in ds)
        n = len(ds)
        fila = {
            "timesteps": self.num_timesteps, "drives": n,
            "puntos_por_drive": round(float(np.mean([d["puntos"] for d in ds])), 3),
            "pct_td": round(100 * motivos["touchdown"] / n, 1),
            "pct_fg": round(100 * motivos["field_goal"] / n, 1),
            "pct_perdidas": round(100 * (motivos["intercepcion"] + motivos["downs"]) / n, 1),
            "yardas_por_jugada": round(sum(d["yardas"] for d in ds) / max(jugadas, 1), 2),
            "intercepciones_por_drive": round(sum(d["intercepciones"] for d in ds) / n, 3),
            "sacks_por_drive": round(sum(d["sacks"] for d in ds) / n, 3),
            "pct_pases": round(100 * pases / max(pases + sum(d["carreras"] for d in ds), 1), 1),
            "minutos": round((time.time() - self.inicio) / 60, 2),
        }
        nuevo = not ARCHIVO_CURVA.exists()
        with open(ARCHIVO_CURVA, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNAS)
            if nuevo:
                w.writeheader()
            w.writerow(fila)
        print(f"jugadas {self.num_timesteps:>9,} | drives {n:>4} | puntos/drive {fila['puntos_por_drive']:5.2f} | "
              f"TD {fila['pct_td']:5.1f}% | yardas/jugada {fila['yardas_por_jugada']:5.2f} | "
              f"pases {fila['pct_pases']:5.1f}% | {fila['minutos']:.1f} min", flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Entrena al DT de NFL con PPO")
    p.add_argument("--timesteps", type=int, default=C.TIMESTEPS_POR_DEFECTO, help="Cantidad de jugadas a entrenar")
    p.add_argument("--rival", default="mezcla", choices=["mezcla", *C.PERFILES_RIVAL])
    p.add_argument("--entornos", type=int, default=C.ENTORNOS_PARALELOS, help="Simulaciones en paralelo")
    p.add_argument("--sin-procesos", action="store_true",
                   help="Simula todo en un solo proceso (más lento, útil si hay problemas en Windows/Colab)")
    p.add_argument("--continuar", default=None, help="Modelo .zip desde el cual seguir entrenando")
    p.add_argument("--salida", default=str(C.MODELO_POR_DEFECTO), help="Dónde guardar el modelo")
    p.add_argument("--semilla", type=int, default=0)
    args = p.parse_args()

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    C.CARPETA_MODELOS.mkdir(exist_ok=True)
    C.CARPETA_RESULTADOS.mkdir(exist_ok=True)

    clase = DummyVecEnv if args.sin_procesos else SubprocVecEnv
    venv = make_vec_env(DriveEnv, n_envs=args.entornos, seed=args.semilla, vec_env_cls=clase,
                        env_kwargs={"rival": args.rival})
    print(f"Entrenando al DT: {args.timesteps:,} jugadas · rival: {args.rival} · {args.entornos} simulaciones en paralelo")
    if args.continuar:
        print(f"Continuando desde {args.continuar}")
        modelo = PPO.load(args.continuar, env=venv, device="cpu")
    else:
        if ARCHIVO_CURVA.exists():
            ARCHIVO_CURVA.unlink()           # curva nueva para un entrenamiento desde cero
        modelo = PPO("MlpPolicy", venv, device="cpu", seed=args.semilla, verbose=0,
                     policy_kwargs={"net_arch": C.RED_NEURONAL}, **C.PPO_PARAMS)

    inicio = time.time()
    callbacks = CallbackList([
        CurvaAprendizaje(inicio),
        CheckpointCallback(save_freq=max(50_000 // args.entornos, 1),
                           save_path=str(C.CARPETA_MODELOS / "checkpoints"), name_prefix="dt_nfl"),
    ])
    modelo.learn(total_timesteps=args.timesteps, callback=callbacks, reset_num_timesteps=not args.continuar)
    modelo.save(args.salida)
    venv.close()
    print(f"\nListo en {(time.time() - inicio) / 60:.1f} minutos. Modelo guardado en {args.salida}")
    print("Siguiente paso:  python evaluate.py")


if __name__ == "__main__":
    main()
