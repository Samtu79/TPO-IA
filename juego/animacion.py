"""
juego/animacion.py — Dibuja una jugada simulada cuadro por cuadro (estilo pizarra de TV).

Para que sea rápido, matplotlib dibuja UNA vez la cancha (con las rutas planeadas) y UNA vez cada
jugador (su círculo con la posición escrita). En cada cuadro solo se pegan esas imágenes en su lugar,
junto con las estelas, la pelota y el reloj: unos pocos milisegundos por cuadro.

  - Animador:        prepara la cancha y dibuja el cuadro que se le pida (la app los muestra en vivo).
  - generar_cuadros: entrega todos los cuadros, uno por uno.
  - animar_jugada:   junta todos los cuadros en un GIF.
  - imagen_previa:   la formación antes del snap, con las rutas planeadas (la "pizarra" del DT).

Con cancha_entera=True se ve la cancha completa; si no, solo el tramo donde pasa la jugada.
"""

from __future__ import annotations

import io
from functools import lru_cache

import imageio.v2 as imageio
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse
from PIL import Image, ImageDraw, ImageFont

import config as C
from juego.dibujo import (_puntos_por_yarda, crear_figura, dibujar_cancha, dibujar_jugadores, dibujar_lineas_de_jugada,
                          dibujar_pelota)
from juego.rutas import camino_absoluto

ELEGIBLES = ("WR", "TE", "RB", "FB")     # los que pueden correr una ruta (mismo orden que en el simulador)
FUENTE = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))  # viene con matplotlib


def _ventana(situacion, resultado) -> tuple[float, float]:
    """Tramo de cancha a mostrar: de un poco antes de la línea hasta un poco después de donde terminó."""
    xs = [situacion.x_scrimmage, resultado.x_final] + [c["P"][:, 0].max() for c in resultado.cuadros[-1:]]
    x0 = max(-3.0, situacion.x_scrimmage - 15)
    x1 = min(C.LARGO + 3, max(max(xs) + 10, x0 + 55))
    return x0, x1


def _figura(situacion, ventana, ancho_pulgadas: float, dpi: int):
    """Cancha con las líneas de la jugada. Devuelve la figura, el eje y cuántos puntos mide una yarda."""
    fig, ax = crear_figura(ancho_pulgadas, dpi=dpi, ventana=ventana)
    dibujar_cancha(ax)
    dibujar_lineas_de_jugada(ax, situacion.x_scrimmage, situacion.x_primer_down)
    return fig, ax, _puntos_por_yarda(ax)


def _carteles(ax, ppy: float, titulo: str, centrados: bool):
    """Título arriba y reloj/estado abajo. Con la cancha entera van al centro (los costados los tapan los menús)."""
    x = (C.LARGO / 2) if centrados else (ax.get_xlim()[0] + 1)
    alineado = "center" if centrados else "left"
    estilo = dict(fontsize=(1.4 if centrados else 0.9) * ppy, fontweight="bold", va="center", ha=alineado, zorder=10)
    ax.text(x, C.ANCHO + 1.4, titulo, color="white", **estilo)
    return ax.text(x, -1.5, "", color=C.COLOR_PRIMER_DOWN, **estilo)


def _rutas_planeadas(ax, ppy: float, caminos) -> None:
    """Rutas planeadas: líneas punteadas, como en la pizarra."""
    for camino in caminos:
        ax.plot(camino[:, 0], camino[:, 1], ls=":", color="white", alpha=0.45, lw=0.12 * ppy, zorder=5)


def _a_imagen(fig) -> np.ndarray:
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


@lru_cache(maxsize=256)
def _sprite(forma: str, color: str, texto: str, px_por_yarda: float, dpi: float, ppy: float) -> Image.Image:
    """Imagen chica con transparencia (un jugador o la pelota), dibujada con matplotlib a la escala de la cancha."""
    medio = (C.RADIO_JUGADOR if forma == "jugador" else 0.5) * 1.3      # yardas del centro al borde de la imagen
    lado = int(np.ceil(2 * medio * px_por_yarda))
    medio = lado / (2 * px_por_yarda)                                   # que la escala sea exacta
    fig = Figure(figsize=(lado / dpi, lado / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-medio, medio)
    ax.set_ylim(-medio, medio)
    ax.axis("off")
    if forma == "jugador":
        ax.add_patch(Circle((0, 0), C.RADIO_JUGADOR, facecolor=color, edgecolor="white", lw=0.08 * ppy))
        ax.text(0, 0, texto, ha="center", va="center", color="white", fontweight="bold",
                fontsize=(0.75 if len(texto) <= 2 else 0.6) * ppy)
    else:
        ax.add_patch(Ellipse((0, 0), 0.9, 0.55, facecolor=C.COLOR_PELOTA, edgecolor="white", lw=0.04 * ppy))
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy())


class Animador:
    """Prepara la cancha de una jugada una sola vez y dibuja cualquier cuadro que se le pida.

    Cada cuadro sale solo de las posiciones de ese momento, así que se pueden saltear cuadros
    (la app lo hace si la PC se atrasa, para no perder la velocidad real).
    Cada cuadro equivale a C.DT_JUGADA * cada segundos de jugada.
    """

    def __init__(self, situacion, resultado, titulo: str = "", cada: int = 1, ancho_pulgadas: float = 10.0,
                 dpi: int = 80, cancha_entera: bool = False):
        self.cuadros = resultado.cuadros[::cada] + resultado.cuadros[-1:]
        self.n = len(self.cuadros)
        ventana = None if cancha_entera else _ventana(situacion, resultado)
        fig, ax, ppy = _figura(situacion, ventana, ancho_pulgadas, dpi)
        _rutas_planeadas(ax, ppy, resultado.caminos.values())
        estado = _carteles(ax, ppy, titulo, centrados=cancha_entera)
        fig.canvas.draw()
        self.fondo = Image.fromarray(_a_imagen(fig))

        # de yardas de la cancha a píxeles de la imagen (la fila 0 es la de arriba)
        alto = self.fondo.height
        (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
        escala_x, escala_y = x1 - x0, y1 - y0
        a_px = lambda P: np.column_stack([x0 + escala_x * P[..., 0].ravel(),
                                          alto - (y0 + escala_y * P[..., 1].ravel())]).reshape(P.shape)
        self.posiciones = a_px(np.array([c["P"] for c in self.cuadros]))          # (cuadros, jugadores, 2)
        self.pelotas = a_px(np.array([c["pelota"] for c in self.cuadros], dtype=float))

        colores = [C.COLOR_ATAQUE if j.equipo == "ataque" else C.COLOR_DEFENSA for j in situacion.jugadores]
        clave = (round(float(escala_x), 3), float(fig.dpi), round(float(ppy), 3))
        self.sprites = [_sprite("jugador", c, j.posicion, *clave) for c, j in zip(colores, situacion.jugadores)]
        self.sprite_pelota = _sprite("pelota", "", "", *clave)
        self.colores_estela = [tuple(int(255 * v) for v in to_rgb(c)) + (128,) for c in colores]   # 50 % transparente
        self.ancho_estela = max(1, round(0.1 * ppy * fig.dpi / 72))

        # reloj y estado de la jugada, abajo
        self.fuente = ImageFont.truetype(FUENTE, max(8, round(estado.get_fontsize() * fig.dpi / 72)))
        self.pos_estado = tuple(a_px(np.array(estado.get_position(), dtype=float)))
        self.ancla_estado = "mm" if cancha_entera else "lm"
        self.color_estado = tuple(int(255 * v) for v in to_rgb(C.COLOR_PRIMER_DOWN))
        self.textos_estado, ultimo = [], ""          # el último evento anunciado hasta cada cuadro
        for c in self.cuadros:
            ultimo = c["texto"] or ultimo
            self.textos_estado.append(ultimo)

    def _pegar(self, imagen: Image.Image, sprite: Image.Image, x: float, y: float) -> None:
        imagen.paste(sprite, (round(x - sprite.width / 2), round(y - sprite.height / 2)), sprite)

    def cuadro(self, k: int) -> np.ndarray:
        """Imagen RGB del cuadro k."""
        imagen = self.fondo.copy()
        if k > 0:                                    # estelas: el camino recorrido hasta ahora
            capa = Image.new("RGBA", imagen.size, (0, 0, 0, 0))
            lapiz = ImageDraw.Draw(capa)
            for i, color in enumerate(self.colores_estela):
                lapiz.line([tuple(p) for p in self.posiciones[: k + 1, i]], fill=color, width=self.ancho_estela,
                           joint="curve")
            imagen.paste(capa, (0, 0), capa)
        for sprite, (x, y) in zip(self.sprites, self.posiciones[k]):
            self._pegar(imagen, sprite, x, y)
        self._pegar(imagen, self.sprite_pelota, *self.pelotas[k])
        ImageDraw.Draw(imagen).text(self.pos_estado, f"{self.cuadros[k]['t']:.1f} s   {self.textos_estado[k]}",
                                    font=self.fuente, fill=self.color_estado, anchor=self.ancla_estado)
        return np.asarray(imagen)


def generar_cuadros(situacion, resultado, titulo: str = "", cada: int = 1, ancho_pulgadas: float = 10.0,
                    dpi: int = 80, cancha_entera: bool = False):
    """Va entregando todos los cuadros (imágenes RGB), uno por uno. Necesita un resultado con grabar=True."""
    animador = Animador(situacion, resultado, titulo, cada, ancho_pulgadas, dpi, cancha_entera)
    for k in range(animador.n):
        yield animador.cuadro(k)


def animar_jugada(situacion, resultado, titulo: str = "", cada: int = 1, ancho_pulgadas: float = 10.0,
                  pausa_final_s: float = 1.5) -> bytes:
    """Devuelve los bytes de un GIF con la jugada. Necesita un resultado simulado con grabar=True."""
    imagenes = list(generar_cuadros(situacion, resultado, titulo, cada, ancho_pulgadas))
    imagenes += [imagenes[-1]] * int(pausa_final_s / (C.DT_JUGADA * cada))
    buf = io.BytesIO()
    imageio.mimsave(buf, imagenes, format="GIF", duration=1000 * C.DT_JUGADA * cada, loop=0)
    return buf.getvalue()


def imagen_previa(situacion, jugada=None, titulo: str = "", pie: str = "", ancho_pulgadas: float = 16.0,
                  dpi: int = 90) -> np.ndarray:
    """La cancha entera antes del snap: las formaciones y, si se pasa la jugada, las rutas planeadas."""
    fig, ax, ppy = _figura(situacion, None, ancho_pulgadas, dpi)
    if jugada is not None and jugada.tipo == "pase":
        elegibles = [j for j in situacion.jugadores if j.equipo == "ataque" and j.posicion in ELEGIBLES]
        y_pelota = situacion.pelota[1]
        _rutas_planeadas(ax, ppy, [camino_absoluto(ruta, j.x, j.y, y_pelota)
                                   for j, ruta in zip(elegibles, jugada.rutas) if ruta != "bloqueo"])
    dibujar_jugadores(ax, situacion.jugadores)
    dibujar_pelota(ax, *situacion.pelota)
    _carteles(ax, ppy, titulo, centrados=True).set_text(pie)
    fig.canvas.draw()
    return _a_imagen(fig)
