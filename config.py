"""
config.py — Parámetros del proyecto en un solo lugar.

Unidades: YARDAS (como en la NFL).
Sistema de coordenadas de la cancha:
  - x: a lo largo, de 0 (fondo de la zona de anotación izquierda) a 120 (fondo de la derecha).
       La línea de gol izquierda está en x = 10 y la derecha en x = 110.
  - y: a lo ancho, de 0 (lateral de abajo) a 53.33 (lateral de arriba).
  - El ataque siempre avanza hacia la DERECHA (x creciente).
"""

# ---------------------------------------------------------------------------
# Medidas reglamentarias de la cancha NFL
# ---------------------------------------------------------------------------
LARGO = 120.0                 # 100 yardas de juego + 2 zonas de anotación de 10
ANCHO = 160 / 3               # 53 1/3 yardas (160 pies)
ZONA_ANOTACION = 10.0         # profundidad de cada zona de anotación
GOL_IZQ = ZONA_ANOTACION      # x de la línea de gol izquierda
GOL_DER = LARGO - ZONA_ANOTACION
HASH_INFERIOR = 70.75 / 3     # los hash marks están a 70' 9" de cada lateral
HASH_SUPERIOR = ANCHO - HASH_INFERIOR
MITAD_ANCHO = ANCHO / 2
NUMEROS_DESDE_LATERAL = 12.0  # la base de los números está a 12 yardas del lateral
ALTO_NUMEROS = 2.0
LINEA_CONVERSION = 2.0        # el "try" (punto extra) se patea/juega desde la yarda 2

# ---------------------------------------------------------------------------
# Jugadores
# ---------------------------------------------------------------------------
RADIO_JUGADOR = 0.85          # radio del círculo que representa a cada jugador (yardas)
JUGADORES_POR_EQUIPO = 11

# ---------------------------------------------------------------------------
# Colores
# ---------------------------------------------------------------------------
COLOR_PASTO_1 = "#2f7d32"
COLOR_PASTO_2 = "#348a38"
COLOR_ZONA_IZQ = "#1c4f8a"    # zona de anotación del ataque (propia)
COLOR_ZONA_DER = "#8a1c1c"    # zona de anotación de la defensa (a donde va el ataque)
COLOR_LINEAS = "#f4f4f0"
COLOR_ATAQUE = "#1f5fbf"
COLOR_DEFENSA = "#d32f2f"
COLOR_BORDE_JUGADOR = "#ffffff"
COLOR_PELOTA = "#7b4a1e"
COLOR_SCRIMMAGE = "#3d8bff"   # línea de scrimmage (azul, como en la TV)
COLOR_PRIMER_DOWN = "#ffd60a" # línea de primer down (amarilla, como en la TV)
COLOR_FONDO = "#1e5e22"

# ===========================================================================
# SIMULACIÓN DE LA JUGADA (lo que pasa después del snap)
# ===========================================================================
DT_JUGADA = 0.1               # cada paso de la simulación son 0,1 segundos
DURACION_MAX_JUGADA = 10.0    # segundos: la jugada se corta sí o sí

# Velocidad máxima de cada posición (yardas por segundo; 9 yd/s ≈ 30 km/h)
VELOCIDADES = {
    "QB": 7.0, "RB": 8.6, "FB": 7.6, "WR": 9.0, "TE": 8.0, "C": 5.5, "G": 5.5, "T": 5.5,
    "DE": 7.2, "DT": 6.4, "NT": 6.0, "LB": 7.9, "ILB": 7.9, "OLB": 8.0, "CB": 8.9, "NB": 8.8, "S": 8.7,
}
RUIDO_VELOCIDAD = 0.06        # cada jugada, cada jugador corre ±6 % (días buenos y malos)

# Aceleración (yd/s²): cuán rápido cambia de velocidad o de dirección. Los receptores, que
# saben a dónde van, quiebran más rápido que los defensores que tienen que reaccionar.
ACELERACIONES = {
    "WR": 25.0, "RB": 20.0, "TE": 18.0, "FB": 16.0, "QB": 14.0, "C": 8.0, "G": 8.0, "T": 8.0,
    "CB": 13.0, "NB": 13.0, "S": 12.0, "LB": 11.0, "ILB": 11.0, "OLB": 11.0, "DE": 9.0, "DT": 9.0, "NT": 9.0,
}
GANANCIA_LLEGADA = 3.0        # al acercarse a su objetivo, cada jugador frena (más alto = frena más tarde)

# Pase
VELOCIDAD_PASE = 22.0         # yd/s del balón en el aire
TIEMPOS_LANZAMIENTO = [1.3, 2.3, 3.3]   # rápido / medio / largo (segundos después del snap)
PRESION_LANZA_YA = 2.5        # si un defensor está a menos de esto del QB, puede lanzar antes (apurado)
PROB_LANZAR_BAJO_PRESION = 0.6   # por cada 0,1 s bajo presión: probabilidad de soltar la pelota ya
PRECISION_BAJO_PRESION = 0.75 # multiplica la probabilidad de pase completo si lanza apurado
DISTANCIA_SACK = 1.2          # si un defensor llega a esta distancia del QB antes del pase: sack

# Recepción: probabilidad base según la distancia del pase y la separación del defensor
COMPLETO_BASE = 0.85          # pase corto sin nadie cerca
COMPLETO_CAIDA_POR_YARDA = 0.02    # con 0.02 el DT clásico rinde ~1,7 puntos por drive, parecido a la NFL real (~2)
SEPARACION_LIBRE = 2.0        # a partir de esta separación (yardas) el receptor está "solo"
COMPLETO_PEGADO = 0.5         # fracción de la probabilidad que queda con el defensor pegado
INTERCEPCION_MAX = 0.05       # probabilidad de intercepción con el defensor encima
INTERCEPCION_POR_YARDA = 0.0005   # los pases largos se interceptan más

# Defensa
REACCION_MARCA = 0.3          # segundos que tarda un defensor en marca personal en reaccionar
REACCION_PELOTA = 0.5         # segundos que tarda la defensa en reaccionar cuando sale el pase
VEL_IR_A_LA_PELOTA = 0.9      # fracción de su velocidad con la que la defensa va a la pelota
LECTURA_CARRERA = 0.3         # segundos que tarda la defensa en darse cuenta de que es carrera
RADIO_ZONA = 7.0              # radio de la zona que cuida un defensor en zona
ANCHO_SECTOR_PROFUNDO = 11.0  # una zona profunda vigila a los receptores a menos de esto a lo ancho
VEL_RUSHER_BLOQUEADO = (0.3, 1.2)  # yd/s que avanza un defensor mientras lo bloquean
PROB_GANAR_BLOQUEO = 0.004    # por cada 0,1 s: probabilidad de que un rusher le gane al bloqueador
DURACION_BLOQUEO_CARRERA = (0.6, 2.2)  # segundos que dura un bloqueo en una carrera
VEL_DEFENSOR_BLOQUEADO = 1.2  # yd/s que avanza un defensor bloqueado en una carrera

# Con la pelota en la mano
VEL_CON_PELOTA = 0.75        # fracción de su velocidad máxima con la que corre el que tiene la pelota
ARRANQUE_CON_PELOTA = 1.0     # segundos que tarda en volver a tomar velocidad después de recibir

# Tackle
DISTANCIA_TACKLE = 1.2
PROB_TACKLE_POR_PASO = 0.55  # por defensor en contacto, cada 0,1 s

# ===========================================================================
# EL DRIVE (episodio de aprendizaje: una serie de jugadas hasta anotar o perder la pelota)
# ===========================================================================
YARDA_INICIO_DRIVE = (15, 40) # el drive arranca en una yarda al azar entre estas
MAX_JUGADAS_DRIVE = 25
DISTANCIA_PUNT_NETA = 40
RANGO_FG_MAXIMO = 57          # yardas: más lejos no se intenta field goal (se patea punt)

# ===========================================================================
# RECOMPENSA
# ===========================================================================
# Base: EPA ("expected points added"), la métrica que usan los equipos de la NFL.
# Cada situación (yarda, down, distancia) vale una cantidad de "puntos esperados";
# la recompensa de una jugada es cuánto mejoró ese valor. Al final del drive la
# suma de recompensas = puntos conseguidos - puntos esperados al empezar.
#   puntos esperados = EP_BASE + EP_POR_YARDA * yarda - EP_POR_DOWN * (down - 1) - EP_POR_YARDA_EXTRA * (distancia - 10)
EP_BASE = -1.2
EP_POR_YARDA = 0.065
EP_POR_DOWN = 0.45
EP_POR_YARDA_EXTRA = 0.04
PUNTOS = {"td": 7.0, "fg": 3.0, "safety": -2.0}

# Extras para experimentar (0 = no se usan). Se suman a la EPA.
RECOMPENSA_EXTRA = {
    "primer_down": 0.0,       # premio por cada primer down conseguido
    "td": 0.0,                # premio adicional por touchdown
    "intercepcion": 0.0,      # castigo adicional por intercepción (ponelo negativo)
    "sack": 0.0,              # castigo adicional por sack (negativo)
    "jugada": 0.0,            # costo por jugada (negativo = apurar el drive)
}

# ===========================================================================
# EL RIVAL (coordinador defensivo con reglas, NO aprende)
# ===========================================================================
COBERTURAS = ["Cover 0", "Cover 1", "Cover 2", "Cover 3", "Cover 4"]
# Probabilidad de cada cobertura según el perfil del rival
PERFILES_RIVAL = {
    "equilibrado": {"Cover 0": 0.05, "Cover 1": 0.25, "Cover 2": 0.25, "Cover 3": 0.30, "Cover 4": 0.15},
    "agresivo":    {"Cover 0": 0.30, "Cover 1": 0.45, "Cover 2": 0.10, "Cover 3": 0.10, "Cover 4": 0.05},
    "conservador": {"Cover 0": 0.00, "Cover 1": 0.10, "Cover 2": 0.30, "Cover 3": 0.30, "Cover 4": 0.30},
}
PROB_DISFRAZ = 0.25           # probabilidad de que los safeties "muestren" otra cobertura antes del snap

# ===========================================================================
# ENTRENAMIENTO (PPO)
# ===========================================================================
PPO_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 256,           # jugadas por entorno antes de cada actualización
    "batch_size": 512,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,         # exploración: probar jugadas nuevas
}
RED_NEURONAL = [128, 128]
ENTORNOS_PARALELOS = 8
TIMESTEPS_POR_DEFECTO = 300_000   # cada "timestep" es UNA jugada

# ===========================================================================
# Carpetas
# ===========================================================================
from pathlib import Path  # noqa: E402

RAIZ = Path(__file__).resolve().parent
CARPETA_MODELOS = RAIZ / "models"
CARPETA_RESULTADOS = RAIZ / "resultados"
MODELO_POR_DEFECTO = CARPETA_MODELOS / "dt_nfl.zip"
