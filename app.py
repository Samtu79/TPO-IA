"""
app.py — DT-IA: un "DT" de NFL que diseña jugadas aprendiendo por refuerzo.

Abrir con:
    streamlit run app.py

La pantalla es la cancha entera, ocupando toda la ventana:
  - Antes de cada jugada se ve la formación de los dos equipos con las rutas planeadas (la "pizarra").
  - Al jugar, la jugada se dibuja en vivo, a velocidad real, a medida que se genera.

Encima flotan menús semitransparentes. Cada uno se oculta (y se vuelve a abrir) tocando su título:
  - Drive (izquierda): quién dirige (al azar, clásico, la IA entrenada o VOS diseñando cada jugada),
    el rival, la jugada elegida, los botones y el resultado de la última jugada.
  - A la derecha: Formaciones, Aprendizaje, Comparación y Libro de jugadas (resultados del entrenamiento).

Funciona 100 % local y sin internet. Si no hay modelo entrenado, avisa y deja usar el resto.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import config as C
from entorno.drive_env import FORMACION_REFERENCIA, DriveEnv, situacion_snap
from entrenadores import NOMBRES, crear_dt
from juego.animacion import Animador, generar_cuadros, imagen_previa
from juego.defensa import cantidad_profundos
from juego.drive import EstadoDrive
from juego.jugada import HUECOS, MOMENTOS, NOMBRES_FORMACIONES, TIPOS, Jugada, elegibles_de
from juego.jugadores import FORMACIONES_ATAQUE, FORMACIONES_DEFENSA, HASHES, armar_situacion
from juego.rutas import NOMBRES_RUTAS, RUTAS
from juego.simulador import simular_jugada

st.set_page_config(page_title="DT-IA · NFL", page_icon="🏈", layout="wide", initial_sidebar_state="collapsed")

CADA_EN_VIVO = 1                 # se dibuja 1 de cada N pasos de la simulación (subilo a 2 si la PC no llega a velocidad real)
ANCHO_IMAGEN, DPI_IMAGEN = 16, 90   # imagen de la cancha: 16 x 90 = 1440 píxeles de ancho
CALIDAD_JPEG = 85
RUTAS_POR_DEFECTO = ["flat", "dig", "go", "slant", "post"]
NOMBRES_RIVAL = {"equilibrado": "Equilibrado", "agresivo": "Agresivo (mucho blitz)",
                 "conservador": "Conservador (cuida lo profundo)"}

# ---------------------------------------------------------------------------
# Estilo: la cancha de fondo en toda la ventana y los menús flotando encima, semitransparentes
# ---------------------------------------------------------------------------
st.html(f"""
<style>
  [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
  [data-testid="stStatusWidget"], [data-testid="stSidebar"], footer {{ display: none !important; }}
  [data-testid="stApp"] {{ background: {C.COLOR_FONDO}; }}
  [data-testid="stMainBlockContainer"] {{ padding: 0 !important; max-width: none !important; }}
  /* mientras la app se actualiza, no oscurecer lo que ya está en pantalla */
  .stale-element, [data-stale="true"] {{ opacity: 1 !important; transition: none !important; }}

  /* la cancha: fija, ocupando toda la ventana (sin deformarse) */
  .st-key-cancha {{ position: fixed; inset: 0; z-index: 0; }}
  .st-key-cancha img {{ width: 100vw !important; height: 100vh !important; max-width: none !important;
                        object-fit: contain; }}
  .st-key-cancha [data-testid="stElementToolbar"] {{ display: none !important; }}

  /* menús flotantes */
  .st-key-panel_drive, .st-key-panel_analisis {{
    position: fixed; top: 14px; z-index: 20; max-height: calc(100vh - 28px); overflow-y: auto;
    transition: width .2s ease; scrollbar-width: thin;
  }}
  .st-key-panel_drive {{ left: 14px; width: min(390px, calc(100vw - 28px)); }}
  .st-key-panel_analisis {{ right: 14px; width: min(560px, calc(100vw - 28px)); }}
  /* cerrados ocupan poco: solo se ve el título */
  .st-key-panel_drive:not(:has(details[open])) {{ width: 230px; }}
  .st-key-panel_analisis:not(:has(details[open])) {{ width: 240px; }}

  .st-key-panel_drive [data-testid="stExpander"] details,
  .st-key-panel_analisis [data-testid="stExpander"] details {{
    background: rgba(8, 14, 22, 0.40); backdrop-filter: blur(5px); -webkit-backdrop-filter: blur(5px);
    border: 1px solid rgba(255, 255, 255, 0.14); border-radius: 14px; box-shadow: 0 8px 28px rgba(0, 0, 0, 0.3);
  }}
  .st-key-panel_drive [data-testid="stExpander"] summary,
  .st-key-panel_analisis [data-testid="stExpander"] summary {{ font-weight: 600; }}
  .st-key-panel_drive, .st-key-panel_analisis {{ text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6); }}
  .st-key-panel_drive [data-baseweb="select"] > div,
  .st-key-panel_analisis [data-baseweb="select"] > div {{ background: rgba(255, 255, 255, 0.07); }}
</style>
""")


# ---------------------------------------------------------------------------
# Modelo entrenado (si existe)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando el DT entrenado...")
def cargar_entrenado(ruta: str, _marca: float):
    return crear_dt("entrenado", ruta)


def dt_entrenado():
    ruta = C.MODELO_POR_DEFECTO
    if not ruta.exists():
        return None, ("Todavía no hay un DT entrenado (`models/dt_nfl.zip`). Entrenalo con `python train.py`. "
                      "Mientras tanto podés usar los otros DTs o dirigir vos.")
    try:
        return cargar_entrenado(str(ruta), ruta.stat().st_mtime), None
    except Exception as e:   # modelo de otra versión o corrupto: la app sigue funcionando
        return None, f"No se pudo cargar el modelo ({type(e).__name__}: {e})."


entrenado, aviso_modelo = dt_entrenado()
OPCIONES_DT = {"vos": "🧑 Vos (diseñás cada jugada)", "aleatorio": NOMBRES["aleatorio"], "clasico": NOMBRES["clasico"]}
if entrenado:
    OPCIONES_DT["entrenado"] = NOMBRES["entrenado"]

ss = st.session_state
ss.setdefault("dt", "entrenado" if entrenado else "clasico")
ss.setdefault("rival", "equilibrado")
ss.setdefault("yarda_inicio", 25)
ss.setdefault("semilla", 1)
ss.setdefault("n_drive", 0)
ss.setdefault("vista", "previa")          # "previa": formación antes del snap · "jugada": la última jugada
if ss.dt not in OPCIONES_DT:
    ss.dt = "clasico"
# valores iniciales de los menús (se leen antes de dibujar la cancha, que va primero)
for clave, valor in {"h_form": NOMBRES_FORMACIONES[0], "h_tipo": TIPOS[0], "h_obj": 0, "h_mom": 1, "h_hueco": 2,
                     **{f"h_ruta_{k}": r for k, r in enumerate(RUTAS_POR_DEFECTO)},
                     "ver_form": False, "v_at": list(FORMACIONES_ATAQUE)[0], "v_def": list(FORMACIONES_DEFENSA)[0],
                     "v_yarda": 25, "v_hash": "centro"}.items():
    ss.setdefault(clave, valor)


# ---------------------------------------------------------------------------
# Lógica del drive
# ---------------------------------------------------------------------------
def nuevo_drive():
    env = DriveEnv(rival=ss.rival, grabar=True)
    ss.n_drive += 1                            # cada drive nuevo es distinto (misma semilla = misma secuencia)
    ss.obs, _ = env.reset(seed=int(ss.semilla) * 1000 + ss.n_drive, options={"yarda": int(ss.yarda_inicio)})
    ss.env = env
    ss.registro = []
    ss.ultima = None
    ss.resumen = None
    ss.vista = "previa"


def jugar(accion):
    """Simula la jugada. El dibujo se hace después, en vivo, en la cancha (imagen None = falta dibujarla)."""
    env = ss.env
    obs, recompensa, terminado, truncado, info = env.step(accion)
    ss.ultima = {"imagen": None, "info": info, "recompensa": recompensa}
    ss.vista = "jugada"
    ss.registro.append({"Situación": info["antes"], "Jugada": info["jugada"],
                        "Defensa (real)": f"{info['personal']} · {info['cobertura']}",
                        "Resultado": info["texto"], "EPA": round(recompensa, 2)})
    ss.obs = obs
    if terminado or truncado:
        ss.resumen = info["drive"]


def decidir_dt():
    if ss.dt == "entrenado":
        return entrenado.decidir(ss.obs, ss.env)
    return crear_dt(ss.dt, semilla=len(ss.registro) + int(ss.semilla) * 100).decidir(ss.obs, ss.env)


def jugada_de_vos() -> Jugada:
    """La jugada que estás diseñando en el menú (se lee de los valores guardados de los menús)."""
    tipo = ss.h_tipo
    rutas = [ss.get(f"h_ruta_{k}", RUTAS_POR_DEFECTO[k]) for k in range(5)] if tipo == "pase" else ["bloqueo"] * 5
    return Jugada(ss.h_form, tipo, rutas, ss.h_obj if tipo == "pase" else 0, ss.h_mom if tipo == "pase" else 1,
                  ss.h_hueco if tipo == "carrera" else 2)


def ver_previa():
    ss.vista = "previa"


# ---------------------------------------------------------------------------
# Dibujo de la cancha
# ---------------------------------------------------------------------------
def a_jpeg(imagen: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(imagen).save(buf, format="JPEG", quality=CALIDAD_JPEG)
    return buf.getvalue()


@st.cache_data(show_spinner=False, max_entries=64)
def imagen_cancha(formacion: str, personal: str, mostrada: str | None, yarda: int, down: int, distancia: int,
                  hash_pelota: str, accion: tuple | None, titulo: str, pie: str) -> bytes:
    """La cancha entera antes del snap (con las rutas de la jugada, si se pasa)."""
    if mostrada:     # una situación del drive: los safeties se paran según lo que muestra la defensa
        sit = situacion_snap(formacion, personal, mostrada, EstadoDrive(yarda, down, distancia, hash_pelota))
    else:            # solo para ver formaciones
        sit = armar_situacion(formacion, personal, yarda, hash_pelota, distancia)
    jugada = Jugada.desde_accion(np.array(accion)) if accion else None
    return a_jpeg(imagen_previa(sit, jugada, titulo, pie, ANCHO_IMAGEN, DPI_IMAGEN))


def mostrar_en_vivo(lienzo, info) -> bytes:
    """Dibuja la jugada cuadro por cuadro a medida que se genera, a velocidad real. Devuelve el último cuadro.

    Siempre dibuja el cuadro que corresponde al tiempo transcurrido: si la PC se atrasa, saltea cuadros
    en vez de mostrar la jugada en cámara lenta. El último cuadro se muestra siempre.
    """
    paso = C.DT_JUGADA * CADA_EN_VIVO
    animador = Animador(info["situacion"], info["simulacion"], titulo=info["antes"], cada=CADA_EN_VIVO,
                        ancho_pulgadas=ANCHO_IMAGEN, dpi=DPI_IMAGEN, cancha_entera=True)
    inicio, k = time.perf_counter(), 0     # el reloj arranca con el primer cuadro (preparar la cancha tarda un poco)
    while True:
        imagen = a_jpeg(animador.cuadro(k))
        lienzo.image(imagen, width="stretch")
        if k == animador.n - 1:
            return imagen
        k = min(max(k + 1, int((time.perf_counter() - inicio) / paso)), animador.n - 1)
        espera = inicio + k * paso - time.perf_counter()
        if espera > 0:                     # dibujar fue más rápido que la jugada real: espera
            time.sleep(espera)


@st.cache_resource(show_spinner=False)
def calentar_dibujo() -> bool:
    """Dibuja una jugada de prueba sin mostrarla, una sola vez por proceso.

    La primera animación tarda casi el doble (matplotlib carga fuentes y arma cachés): así la primera
    jugada de verdad ya sale a velocidad real.
    """
    sit = armar_situacion("Shotgun (3 receptores)", "Nickel (4-2-5)", 25)
    jugada = Jugada("Shotgun (3 receptores)", "pase", ["flat", "bloqueo", "go", "slant", "out"], 3, 0, 2)
    simulacion = simular_jugada(jugada, sit, "Cover 3", np.random.default_rng(0), grabar=True)
    for imagen in generar_cuadros(sit, simulacion, ancho_pulgadas=ANCHO_IMAGEN, dpi=DPI_IMAGEN, cancha_entera=True):
        a_jpeg(imagen)
    return True


if "env" not in ss:
    nuevo_drive()
env = ss.env
if ss.resumen:
    propuesta = None
elif ss.dt == "vos":
    propuesta = jugada_de_vos()
else:
    propuesta = Jugada.desde_accion(decidir_dt())

# ---------------------------------------------------------------------------
# La cancha (va primero: mientras la jugada se dibuja en vivo, el resultado todavía no aparece en el menú)
# ---------------------------------------------------------------------------
with st.container(key="cancha"):
    lienzo = st.empty()
    u = ss.ultima
    if ss.ver_form:
        ss.cancha_actual = imagen_cancha(ss.v_at, ss.v_def, None, ss.v_yarda, 1, 10, ss.v_hash, None,
                                         "Formaciones", f"{ss.v_at}  contra  {ss.v_def}")
        lienzo.image(ss.cancha_actual, width="stretch")
    elif ss.vista == "jugada" and u and "simulacion" in u["info"]:
        if u.get("imagen") is None:
            if "cancha_actual" in ss:          # mientras se prepara el primer cuadro, queda la cancha que se veía
                lienzo.image(ss.cancha_actual, width="stretch")
            u["imagen"] = mostrar_en_vivo(lienzo, u["info"])
        else:
            lienzo.image(u["imagen"], width="stretch")
        ss.cancha_actual = u["imagen"]
    else:
        e, d = env.estado, env.defensa
        if ss.vista == "jugada" and u:            # patada (punt o field goal): no hay jugada para animar
            pie = u["info"]["texto"]
        else:
            pie = (f"Antes del snap · la defensa muestra {d['personal']} con "
                   f"{cantidad_profundos(d['mostrada'])} safety(s) profundo(s)")
        con_formacion = propuesta is not None and propuesta.tipo != "patear"
        formacion = propuesta.formacion if con_formacion else FORMACION_REFERENCIA
        accion = tuple(int(a) for a in propuesta.a_accion()) if con_formacion else None
        ss.cancha_actual = imagen_cancha(formacion, d["personal"], d["mostrada"], e.yarda, e.down, e.distancia,
                                         e.hash, accion, e.describir(), pie)
        lienzo.image(ss.cancha_actual, width="stretch")

# ---------------------------------------------------------------------------
# Menú del drive (izquierda)
# ---------------------------------------------------------------------------
with st.container(key="panel_drive"):
    with st.expander("🏈 DT-IA · Drive", expanded=True):
        if ss.resumen:
            r = ss.resumen
            emoji = {"touchdown": "🎉", "field_goal": "🥅"}.get(r["motivo"], "🏁")
            st.success(f"{emoji} **Fin del drive: {r['motivo'].replace('_', ' ')}** · {r['puntos']:+.0f} puntos · "
                       f"{r['jugadas']} jugadas · {r['yardas']:.0f} yardas")
            if st.button("🔄 Jugar otro drive", type="primary", width="stretch"):
                nuevo_drive()
                st.rerun()
        else:
            e, d = env.estado, env.defensa
            st.markdown(f"#### {e.describir()}")
            st.caption(f"La defensa muestra **{d['personal']}** con **{cantidad_profundos(d['mostrada'])} safety(s) "
                       f"profundo(s)** · {env.en_caja} en la caja")
            st.selectbox("¿Quién dirige el ataque?", list(OPCIONES_DT), format_func=OPCIONES_DT.get, key="dt",
                         on_change=ver_previa)

            if ss.dt == "vos":
                st.selectbox("Formación", NOMBRES_FORMACIONES, key="h_form", on_change=ver_previa)
                tipo = st.radio("Tipo", TIPOS, horizontal=True, key="h_tipo", on_change=ver_previa)
                if tipo == "pase":
                    pos = elegibles_de(ss.h_form)
                    cuenta, etiquetas = {}, []
                    for p in pos:
                        cuenta[p] = cuenta.get(p, 0) + 1
                        etiquetas.append(f"{p}{cuenta[p]}" if pos.count(p) > 1 else p)
                    ayuda = " · ".join(f"{n}: {RUTAS[n][2]}" for n in NOMBRES_RUTAS)
                    for fila in ((0, 1, 2), (3, 4)):
                        cols = st.columns(3)
                        for col, k in zip(cols, fila):
                            col.selectbox(etiquetas[k], NOMBRES_RUTAS, key=f"h_ruta_{k}", help=ayuda,
                                          on_change=ver_previa)
                    c1, c2 = st.columns(2)
                    c1.selectbox("Pase a", range(5), key="h_obj", on_change=ver_previa,
                                 format_func=lambda k: f"{etiquetas[k]} ({ss.get(f'h_ruta_{k}')})")
                    c2.selectbox("Lanza", range(3), format_func=lambda k: MOMENTOS[k], key="h_mom")
                elif tipo == "carrera":
                    st.selectbox("Por dónde", range(5), format_func=lambda k: HUECOS[k], key="h_hueco",
                                 on_change=ver_previa)
                if st.button("▶️ ¡Hike!", type="primary", width="stretch"):
                    jugar(jugada_de_vos().a_accion())
                    st.rerun()
            else:
                st.markdown(f"**Elige:** {propuesta.describir()}")
                b1, b2 = st.columns(2)
                if b1.button("▶️ Jugar", type="primary", width="stretch"):
                    jugar(propuesta.a_accion())
                    st.rerun()
                if b2.button("⏩ Drive entero", width="stretch", help="Juega hasta el final; en vivo se ve la última"):
                    with st.spinner("Jugando el drive..."):
                        while not ss.resumen:
                            jugar(decidir_dt())
                    st.rerun()

        if ss.ultima:
            info = ss.ultima["info"]
            st.divider()
            st.markdown(f"**Última jugada** ({info['antes']})  \n{info['jugada']}")
            st.markdown(f"➡️ {info['texto']}")
            st.caption(f"La defensa jugaba **{info['cobertura']}** (mostraba {cantidad_profundos(info['mostrada'])} "
                       f"profundo/s) · EPA: **{ss.ultima['recompensa']:+.2f}** puntos esperados")
            c1, c2 = st.columns(2)
            if "simulacion" in info and c1.button("🔁 Ver de nuevo", width="stretch"):
                ss.ultima["imagen"] = None
                ss.vista = "jugada"
                st.rerun()
            if not ss.resumen and ss.vista == "jugada" and c2.button("📋 Próxima formación", width="stretch"):
                ver_previa()
                st.rerun()

        st.divider()
        c1, c2 = st.columns(2)
        with c1.popover("📋 Jugadas", width="stretch"):
            if ss.registro:
                st.dataframe(pd.DataFrame(ss.registro), hide_index=True)
            else:
                st.caption("Todavía no se jugó ninguna jugada en este drive.")
        with c2.popover("⚙️ Ajustes", width="stretch"):
            st.selectbox("Rival (coordinador defensivo)", list(C.PERFILES_RIVAL), key="rival",
                         format_func=lambda r: NOMBRES_RIVAL.get(r, r.capitalize()), on_change=nuevo_drive)
            st.slider("Yarda de inicio (desde la propia línea de gol)", 1, 80, key="yarda_inicio",
                      on_change=nuevo_drive)
            st.number_input("Semilla (misma semilla = misma secuencia de drives)", 0, 99_999, key="semilla",
                            on_change=lambda: ss.update(n_drive=0))
            if st.button("🔄 Nuevo drive", width="stretch"):
                nuevo_drive()
                st.rerun()
        st.caption(f"Rival: {NOMBRES_RIVAL.get(env.perfil, env.perfil)}  \n"
                   "🔵 ataque · 🔴 defensa · punteado: rutas planeadas · línea azul: scrimmage · "
                   "línea amarilla: primer down")
        if aviso_modelo:
            st.caption(f"ℹ️ {aviso_modelo}")

# ---------------------------------------------------------------------------
# Menús de análisis (derecha)
# ---------------------------------------------------------------------------
COLUMNAS = {"nombre": "DT", "rival": "Rival", "puntos_por_drive": "Puntos/drive", "pct_td": "% TD",
            "pct_fg": "% FG", "pct_perdidas": "% pérdidas", "yardas_por_jugada": "Yardas/jugada",
            "pct_pases_completos": "% pases completos", "pct_pases": "% jugadas de pase"}

with st.container(key="panel_analisis"):
    with st.expander("📐 Formaciones"):
        st.toggle("Mostrarlas en la cancha", key="ver_form")
        c1, c2 = st.columns(2)
        c1.selectbox("Ofensiva", list(FORMACIONES_ATAQUE), key="v_at")
        c2.selectbox("Defensiva", list(FORMACIONES_DEFENSA), key="v_def")
        st.slider("Yarda", 1, 99, key="v_yarda")
        st.radio("Hash", list(HASHES), horizontal=True, key="v_hash")
        st.caption("🔵 Ataque: QB mariscal · RB corredor · FB fullback · WR receptor · TE ala cerrada · C/G/T línea  \n"
                   "🔴 Defensa: DE ala · DT tackle · NT nose tackle · LB/ILB/OLB apoyadores · CB esquinero · "
                   "NB esquinero de slot · S safety")

    with st.expander("📈 Aprendizaje"):
        ruta = C.CARPETA_RESULTADOS / "curva_aprendizaje.csv"
        if ruta.exists():
            df = pd.read_csv(ruta)
            df["miles de jugadas"] = df["timesteps"] / 1000
            st.markdown("El DT juega cientos de miles de jugadas por prueba y error. "
                        "**Si aprende, los puntos por drive suben.**")
            st.markdown("**Puntos por drive**")
            st.line_chart(df.set_index("miles de jugadas")["puntos_por_drive"].rolling(5, min_periods=1).mean(),
                          height=200)
            st.markdown("**% durante el entrenamiento**")
            g = df.set_index("miles de jugadas")[["pct_td", "pct_perdidas", "pct_pases"]]
            g.columns = ["Drives con TD", "Pérdidas de balón", "Jugadas de pase"]
            st.line_chart(g.rolling(5, min_periods=1).mean(), height=200)
        else:
            st.info("Todavía no hay curva de aprendizaje: se genera al entrenar con `python train.py`.")

    with st.expander("📊 Comparación"):
        st.markdown("Los DTs juegan **exactamente los mismos drives** (misma semilla) contra cada rival.")
        ruta = C.CARPETA_RESULTADOS / "comparacion.csv"
        if ruta.exists():
            st.dataframe(pd.read_csv(ruta)[list(COLUMNAS)].rename(columns=COLUMNAS), hide_index=True)
            if (C.CARPETA_RESULTADOS / "comparacion.png").exists():
                st.image(str(C.CARPETA_RESULTADOS / "comparacion.png"))
        else:
            st.info("Todavía no hay comparación guardada. Generala con `python evaluate.py` o acá abajo.")
        n = st.slider("Drives por DT y por rival (comparación rápida)", 20, 300, 60, step=20)
        if st.button("Comparar ahora"):
            from evaluate import evaluar
            dts = ["aleatorio", "clasico"] + (["entrenado"] if entrenado else [])
            with st.spinner("Jugando drives..."):
                res = evaluar(dts, list(C.PERFILES_RIVAL), n, 50_000)
            st.dataframe(res[list(COLUMNAS)].rename(columns=COLUMNAS), hide_index=True)

    with st.expander("📖 Libro de jugadas"):
        if entrenado:
            from evaluate import libro_de_jugadas
            st.markdown("Qué jugada elige la IA en cada situación (pelota en la propia 35). "
                        "Así se ve la **táctica que diseñó aprendiendo**.")
            libro = libro_de_jugadas(entrenado)
            c1, c2 = st.columns(2)
            f_down = c1.multiselect("Down", [1, 2, 3, 4], default=[1, 2, 3, 4])
            f_pers = c2.multiselect("Personal rival", list(FORMACIONES_DEFENSA), default=list(FORMACIONES_DEFENSA))
            vista = libro[libro["down"].isin(f_down) & libro["personal_rival"].isin(f_pers)]
            st.dataframe(vista[["situacion", "personal_rival", "safeties_profundos", "tipo", "jugada"]]
                         .rename(columns={"situacion": "Situación", "personal_rival": "Personal rival",
                                          "safeties_profundos": "Safeties profundos", "tipo": "Tipo",
                                          "jugada": "Jugada"}),
                         hide_index=True)
            resumen = libro.groupby("tipo").size() / len(libro) * 100
            st.caption(" · ".join(f"{t}: {v:.0f} %" for t, v in resumen.items()))
        else:
            st.info("El libro de jugadas aparece cuando hay un DT entrenado (`python train.py`).")

# Al final, para no demorar lo que se ve: prepara el dibujo para que la primera jugada salga a velocidad real.
calentar_dibujo()
