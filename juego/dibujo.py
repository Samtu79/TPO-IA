"""
juego/dibujo.py — Dibuja la cancha de NFL en 2D y a los jugadores como círculos.

Todo se dibuja con matplotlib, localmente (no necesita internet). Lo mismo
sirve para la app, para guardar imágenes y, más adelante, para animar jugadas.

Uso típico:
    fig, ax = crear_figura()
    dibujar_cancha(ax)
    dibujar_lineas_de_jugada(ax, x_scrimmage, x_primer_down)
    dibujar_jugadores(ax, jugadores)
    dibujar_pelota(ax, x, y)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # dibujar sin ventana (necesario para Streamlit)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle  # noqa: E402

import config as C  # noqa: E402

MARGEN = 3.0  # yardas de "afuera" que se ven alrededor de la cancha


def crear_figura(ancho_pulgadas: float = 14.0, dpi: int = 100, ventana: tuple[float, float] | None = None):
    """
    Figura con un eje que ocupa todo el lienzo, en escala 1:1 (una yarda mide lo mismo en x que en y).
    ventana=(x0, x1) muestra solo ese tramo de la cancha (para hacer zoom en una jugada).
    """
    x0, x1 = ventana if ventana else (-MARGEN, C.LARGO + MARGEN)
    largo, ancho = x1 - x0, C.ANCHO + 2 * MARGEN
    # Figura propia, fuera de pyplot: pyplot es global y no es seguro con varias ejecuciones de la app a la vez
    # (una podía cerrar la figura de otra a mitad del dibujo y dejarla sin lienzo).
    fig = Figure(figsize=(ancho_pulgadas, ancho_pulgadas * ancho / largo), dpi=dpi)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(C.COLOR_FONDO)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(-MARGEN, C.ANCHO + MARGEN)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _puntos_por_yarda(ax) -> float:
    """Cuántos puntos tipográficos mide una yarda en pantalla (para escalar los textos)."""
    fig = ax.figure
    ancho_pts = ax.get_position().width * fig.get_figwidth() * 72
    x0, x1 = ax.get_xlim()
    return ancho_pts / (x1 - x0)


def dibujar_cancha(ax) -> None:
    """Cancha reglamentaria: pasto a franjas, zonas de anotación, líneas, números, hash marks y postes."""
    L, A = C.LARGO, C.ANCHO
    ppy = _puntos_por_yarda(ax)

    # Fondo y franjas de pasto cada 5 yardas
    ax.add_patch(Rectangle((-MARGEN, -MARGEN), L + 2 * MARGEN, A + 2 * MARGEN, color=C.COLOR_FONDO, zorder=0))
    for i, x in enumerate(range(int(C.GOL_IZQ), int(C.GOL_DER), 5)):
        color = C.COLOR_PASTO_1 if i % 2 == 0 else C.COLOR_PASTO_2
        ax.add_patch(Rectangle((x, 0), 5, A, color=color, zorder=1))

    # Zonas de anotación
    for x0, color in ((0, C.COLOR_ZONA_IZQ), (C.GOL_DER, C.COLOR_ZONA_DER)):
        ax.add_patch(Rectangle((x0, 0), C.ZONA_ANOTACION, A, color=color, zorder=1))
    for xc, rot in ((C.ZONA_ANOTACION / 2, 90), (L - C.ZONA_ANOTACION / 2, -90)):
        ax.text(xc, A / 2, "ZONA DE ANOTACIÓN", rotation=rot, ha="center", va="center",
                color=C.COLOR_LINEAS, alpha=0.85, fontsize=2.6 * ppy, fontweight="bold", zorder=2)

    lineas = dict(color=C.COLOR_LINEAS, zorder=3, solid_capstyle="butt")

    # Borde (laterales y líneas de fondo)
    ax.plot([0, L, L, 0, 0], [0, 0, A, A, 0], lw=0.35 * ppy, **lineas)

    # Líneas de yarda cada 5 (las de gol, más gruesas)
    for x in range(int(C.GOL_IZQ), int(C.GOL_DER) + 1, 5):
        grosor = 0.3 * ppy if x in (C.GOL_IZQ, C.GOL_DER) else 0.12 * ppy
        ax.plot([x, x], [0, A], lw=grosor, **lineas)

    # Marcas de cada yarda: junto a los laterales y en los hash marks
    for x in range(int(C.GOL_IZQ) + 1, int(C.GOL_DER)):
        if x % 5 == 0:
            continue
        for y0, y1 in ((0.3, 1.0), (A - 1.0, A - 0.3),
                       (C.HASH_INFERIOR - 0.33, C.HASH_INFERIOR + 0.33),
                       (C.HASH_SUPERIOR - 0.33, C.HASH_SUPERIOR + 0.33)):
            ax.plot([x, x], [y0, y1], lw=0.08 * ppy, **lineas)

    # Líneas de conversión (yarda 2 de cada lado)
    for x in (C.GOL_IZQ + C.LINEA_CONVERSION, C.GOL_DER - C.LINEA_CONVERSION):
        ax.plot([x, x], [A / 2 - 0.5, A / 2 + 0.5], lw=0.12 * ppy, **lineas)

    # Números cada 10 yardas, con la flecha que apunta al gol más cercano
    tam = C.ALTO_NUMEROS * ppy * 1.35
    for k in range(10, 100, 10):
        x = C.GOL_IZQ + k
        numero = str(k if k <= 50 else 100 - k)
        abajo = C.NUMEROS_DESDE_LATERAL + C.ALTO_NUMEROS / 2
        arriba = A - abajo
        ax.text(x, abajo, numero, ha="center", va="center", color=C.COLOR_LINEAS,
                fontsize=tam, fontweight="bold", family="DejaVu Sans", zorder=3)
        ax.text(x, arriba, numero, ha="center", va="center", color=C.COLOR_LINEAS, rotation=180,
                fontsize=tam, fontweight="bold", family="DejaVu Sans", zorder=3)
        if k != 50:
            sentido = -1 if k < 50 else 1            # hacia qué gol apunta la flecha
            xf = x + sentido * 2.6
            for yf in (abajo + 0.35, arriba - 0.35):
                ax.add_patch(Polygon([(xf + sentido * 0.6, yf), (xf, yf + 0.35), (xf, yf - 0.35)],
                                     closed=True, color=C.COLOR_LINEAS, zorder=3))

    # Postes (vistos desde arriba: un travesaño detrás de cada línea de fondo)
    medio_travesanio = (18.5 / 3) / 2
    for x in (-0.7, L + 0.7):
        ax.plot([x, x], [A / 2 - medio_travesanio, A / 2 + medio_travesanio],
                color="#ffd400", lw=0.25 * ppy, zorder=3, solid_capstyle="round")


def dibujar_lineas_de_jugada(ax, x_scrimmage: float, x_primer_down: float | None = None) -> None:
    """Línea de scrimmage (azul) y línea de primer down (amarilla), como en las transmisiones."""
    ppy = _puntos_por_yarda(ax)
    ax.plot([x_scrimmage, x_scrimmage], [0, C.ANCHO], color=C.COLOR_SCRIMMAGE, lw=0.3 * ppy, alpha=0.9, zorder=4)
    if x_primer_down is not None and x_primer_down < C.GOL_DER:
        ax.plot([x_primer_down, x_primer_down], [0, C.ANCHO], color=C.COLOR_PRIMER_DOWN,
                lw=0.3 * ppy, alpha=0.9, zorder=4)


def dibujar_jugadores(ax, jugadores, etiquetas: bool = True) -> None:
    """Cada jugador es un círculo del color de su equipo, con su posición escrita adentro."""
    ppy = _puntos_por_yarda(ax)
    for j in jugadores:
        color = C.COLOR_ATAQUE if j.equipo == "ataque" else C.COLOR_DEFENSA
        ax.add_patch(Circle((j.x, j.y), C.RADIO_JUGADOR, facecolor=color, edgecolor=C.COLOR_BORDE_JUGADOR,
                            lw=0.08 * ppy, zorder=6))
        if etiquetas:
            ax.text(j.x, j.y, j.posicion, ha="center", va="center", color="white", fontweight="bold",
                    fontsize=(0.75 if len(j.posicion) <= 2 else 0.6) * ppy, zorder=7)


def dibujar_pelota(ax, x: float, y: float) -> None:
    """La pelota: un óvalo marrón con las costuras."""
    ppy = _puntos_por_yarda(ax)
    ax.add_patch(Ellipse((x, y), 0.9, 0.55, facecolor=C.COLOR_PELOTA, edgecolor="white", lw=0.04 * ppy, zorder=8))
    ax.plot([x - 0.18, x + 0.18], [y, y], color="white", lw=0.04 * ppy, zorder=9)
