"""
evaluate.py — Compara a los DTs en los MISMOS drives y arma el "libro de jugadas" del agente.

Ejemplos:
    python evaluate.py                        # al azar vs clásico vs IA, contra los 3 rivales
    python evaluate.py --drives 1000          # más drives = resultados más confiables
    python evaluate.py --dts aleatorio clasico   # sin modelo entrenado

Guarda en resultados/:
    comparacion.csv / comparacion.png         -> puntos por drive, % de touchdowns, etc.
    curva_aprendizaje.png                     -> si existe la curva del entrenamiento
    libro_de_jugadas.csv                      -> qué jugada elige la IA en cada situación
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np
import pandas as pd

import config as C
from entorno.drive_env import PERSONALES, DriveEnv, armar_observacion
from entrenadores import NOMBRES, crear_dt, jugar_drive
from juego.drive import EstadoDrive
from juego.jugada import Jugada

COLORES = {"aleatorio": "#8a8984", "clasico": "#eb6834", "entrenado": "#2a78d6"}
TINTA, TINTA_2, GRILLA = "#0b0b0b", "#52514e", "#e4e3df"
METRICAS = [("puntos_por_drive", "Puntos por drive", True), ("pct_td", "Drives con touchdown (%)", True),
            ("yardas_por_jugada", "Yardas por jugada", True), ("pct_perdidas", "Pérdidas de balón (%)", False)]


def evaluar(dts: list[str], rivales: list[str], drives: int, semilla: int, ruta_modelo=None) -> pd.DataFrame:
    filas = []
    for nombre in dts:
        try:
            dt = crear_dt(nombre, ruta_modelo, semilla=0)
        except FileNotFoundError as e:
            print(f"[aviso] Se omite '{nombre}': {e}")
            continue
        for rival in rivales:
            env = DriveEnv(rival=rival)
            ds = [jugar_drive(dt, env, semilla + k) for k in range(drives)]
            motivos = Counter(d["motivo"] for d in ds)
            jugadas = sum(d["jugadas"] for d in ds)
            pases = sum(d["pases"] for d in ds)
            filas.append({
                "dt": nombre, "nombre": NOMBRES[nombre], "rival": rival, "drives": drives,
                "puntos_por_drive": round(float(np.mean([d["puntos"] for d in ds])), 2),
                "pct_td": round(100 * motivos["touchdown"] / drives, 1),
                "pct_fg": round(100 * motivos["field_goal"] / drives, 1),
                "pct_perdidas": round(100 * (motivos["intercepcion"] + motivos["downs"]) / drives, 1),
                "yardas_por_jugada": round(sum(d["yardas"] for d in ds) / max(jugadas, 1), 2),
                "pct_pases_completos": round(100 * sum(d["completos"] for d in ds) / max(pases, 1), 1),
                "intercepciones_por_drive": round(sum(d["intercepciones"] for d in ds) / drives, 3),
                "sacks_por_drive": round(sum(d["sacks"] for d in ds) / drives, 3),
                "pct_pases": round(100 * pases / max(pases + sum(d["carreras"] for d in ds), 1), 1),
            })
            print(f"  {NOMBRES[nombre]:<22} vs rival {rival:<12} {filas[-1]['puntos_por_drive']:5.2f} puntos/drive")
    return pd.DataFrame(filas)


def libro_de_jugadas(dt) -> pd.DataFrame:
    """Qué jugada elige el DT en cada situación típica (pelota en la propia 35, hash al centro)."""
    filas = []
    for down in (1, 2, 3, 4):
        for distancia in (1, 3, 7, 10, 15):
            if down == 1 and distancia != 10:
                continue
            for personal in PERSONALES:
                for profundos in (1, 2):
                    e = EstadoDrive(35, down, distancia, "centro", down - 1)
                    obs = armar_observacion(e, personal, profundos, 6, 0.0)
                    jugada = Jugada.desde_accion(dt.decidir(obs, None))
                    filas.append({"situacion": e.describir(), "down": down, "distancia": distancia,
                                  "personal_rival": personal, "safeties_profundos": profundos,
                                  "tipo": jugada.tipo, "formacion": jugada.formacion, "jugada": jugada.describir()})
    return pd.DataFrame(filas)


def graficar_comparacion(res: pd.DataFrame, ruta) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dts = list(dict.fromkeys(res["dt"]))
    rivales = list(dict.fromkeys(res["rival"]))
    fig, ejes = plt.subplots(1, len(METRICAS), figsize=(4.4 * len(METRICAS), 4.2))
    ancho = 0.8 / len(dts)
    x = np.arange(len(rivales))
    for ax, (col, titulo, mas_es_mejor) in zip(ejes, METRICAS):
        for k, dt in enumerate(dts):
            sub = res[res["dt"] == dt].set_index("rival").reindex(rivales)
            ax.bar(x + (k - (len(dts) - 1) / 2) * ancho, sub[col].values, ancho * 0.92,
                   color=COLORES[dt], label=NOMBRES[dt])
        ax.set_title(titulo + ("  ↑ mejor" if mas_es_mejor else "  ↓ mejor"), fontsize=10, color=TINTA, loc="left")
        ax.set_xticks(x, [f"rival {r}" for r in rivales], color=TINTA_2, fontsize=8)
        ax.tick_params(axis="y", colors=TINTA_2, labelsize=8)
        ax.grid(axis="y", color=GRILLA, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRILLA)
    ejes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=len(dts), frameon=False, fontsize=9,
                   labelcolor=TINTA_2)
    fig.suptitle("Comparación de DTs en los mismos drives", fontsize=13, color=TINTA, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120, bbox_inches="tight")
    plt.close(fig)


def graficar_curva(ruta_csv, ruta_png) -> bool:
    if not ruta_csv.exists():
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(ruta_csv)
    suave = lambda s: s.rolling(5, min_periods=1).mean()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(df["timesteps"] / 1e3, suave(df["puntos_por_drive"]), color="#2a78d6", lw=2)
    a1.set_title("Puntos por drive (sube = aprende)", loc="left", fontsize=10, color=TINTA)
    for col, color, nombre in (("pct_td", "#1baf7a", "Drives con TD"), ("pct_perdidas", "#e34948", "Pérdidas de balón"),
                               ("pct_pases", "#eb6834", "Jugadas de pase")):
        a2.plot(df["timesteps"] / 1e3, suave(df[col]), color=color, lw=2, label=nombre)
    a2.set_title("% durante el entrenamiento", loc="left", fontsize=10, color=TINTA)
    a2.legend(frameon=False, fontsize=8)
    for ax in (a1, a2):
        ax.set_xlabel("miles de jugadas de entrenamiento", color=TINTA_2, fontsize=9)
        ax.grid(color=GRILLA, lw=0.8)
        ax.tick_params(colors=TINTA_2, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Compara DTs y arma el libro de jugadas")
    p.add_argument("--dts", nargs="+", default=["aleatorio", "clasico", "entrenado"], choices=list(NOMBRES))
    p.add_argument("--rivales", nargs="+", default=list(C.PERFILES_RIVAL), choices=list(C.PERFILES_RIVAL))
    p.add_argument("--drives", type=int, default=300, help="Drives por DT y por rival")
    p.add_argument("--semilla", type=int, default=50_000)
    p.add_argument("--modelo", default=None)
    args = p.parse_args()

    C.CARPETA_RESULTADOS.mkdir(exist_ok=True)
    print(f"Jugando {args.drives} drives por DT y por rival (mismos drives para todos)...")
    res = evaluar(args.dts, args.rivales, args.drives, args.semilla, args.modelo)
    res.to_csv(C.CARPETA_RESULTADOS / "comparacion.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + res.drop(columns=["dt", "drives"]).to_string(index=False))
    graficar_comparacion(res, C.CARPETA_RESULTADOS / "comparacion.png")
    if graficar_curva(C.CARPETA_RESULTADOS / "curva_aprendizaje.csv", C.CARPETA_RESULTADOS / "curva_aprendizaje.png"):
        print("\nCurva de aprendizaje: resultados/curva_aprendizaje.png")
    if "entrenado" in set(res["dt"]):
        libro = libro_de_jugadas(crear_dt("entrenado", args.modelo))
        libro.to_csv(C.CARPETA_RESULTADOS / "libro_de_jugadas.csv", index=False)
        print("Libro de jugadas del agente: resultados/libro_de_jugadas.csv")
    print("Comparación: resultados/comparacion.csv y comparacion.png")


if __name__ == "__main__":
    main()
