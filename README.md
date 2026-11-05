# 🏈 DT-IA — Un coordinador ofensivo de NFL que aprende a diseñar jugadas

TP de Inteligencia Artificial (UADE). Un agente de aprendizaje por refuerzo (PPO) hace de DT del ataque:
antes de cada jugada mira la situación y la defensa, diseña la jugada (formación, rutas, a quién pasa, cuándo,
por dónde corre) y aprende por prueba y error qué funciona.

> **Estado:** el código está listo y probado, pero **el agente todavía NO está entrenado**. El entrenamiento y
> las pruebas los corrés vos. Hasta entonces, la app funciona con los otros DTs y con vos dirigiendo.

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest tests -q
```

La línea de `torch` instala la versión sin GPU, mucho más liviana. Las pruebas tardan uno o dos minutos y tienen
que terminar con `21 passed`. En Windows, poné el proyecto en una ruta corta (por ejemplo `C:\dt-nfl`): con
rutas muy largas la instalación falla.

## Uso

```bash
python train.py                  # 1) entrena al DT (300.000 jugadas)
python evaluate.py               # 2) lo compara con los otros DTs y arma su libro de jugadas
streamlit run app.py             # 3) abre la app
```

Podés abrir la app en cualquier momento, aunque no hayas entrenado. Funciona sin internet.

### Entrenar

Mientras entrena muestra una línea por actualización. **Si aprende, `puntos/drive` sube.**

```
jugadas     4,096 | drives  760 | puntos/drive  0.44 | TD   4.5% | yardas/jugada  3.96 | pases  51.0% | 0.3 min
```

Arranca bajo porque al principio patea en cualquier down. Las vallas a superar son las de la tabla de abajo.

| Opción | Qué hace | Ejemplo |
|---|---|---|
| `--timesteps` | cuántas jugadas entrenar | `--timesteps 1000000` |
| `--rival` | contra quién: `mezcla`, `equilibrado`, `agresivo`, `conservador` | `--rival agresivo` |
| `--continuar` | sigue entrenando un modelo guardado | `--continuar models/dt_nfl.zip` |
| `--salida` | dónde guardar el modelo | `--salida models/vs_agresivo.zip` |
| `--entornos` | simulaciones en paralelo | `--entornos 4` |
| `--sin-procesos` | todo en un proceso (más lento; útil si falla el paralelismo) | |
| `--semilla` | repetir un entrenamiento | `--semilla 3` |

Tiempo: 300.000 jugadas deberían llevar unos 10 a 25 minutos. Es una **estimación** a partir de una prueba de
4.096 jugadas, no una medición de un entrenamiento completo. No se pudo verificar todavía cuántas jugadas
necesita para superar al DT clásico: quizás haga falta subir `--timesteps`.

Guarda el modelo en `models/dt_nfl.zip`, copias intermedias en `models/checkpoints/` y la curva en
`resultados/curva_aprendizaje.csv`. Un entrenamiento desde cero reemplaza la curva anterior.

### Evaluar

```bash
python evaluate.py --drives 1000             # más drives = resultados más confiables
python evaluate.py --dts aleatorio clasico   # sin modelo entrenado
```

Todos los DTs juegan los mismos drives contra cada rival. Guarda `comparacion.csv`, `comparacion.png`,
`curva_aprendizaje.png` y `libro_de_jugadas.csv` en `resultados/`. La app los muestra en sus pestañas.

### La app

La pantalla es la **cancha entera**, ocupando toda la ventana. Antes de cada jugada se ve la formación de los dos
equipos con las rutas planeadas; al apretar **Jugar** (o **¡Hike!**), la jugada se dibuja en vivo, a velocidad
real, y el resultado aparece cuando termina. Encima flotan menús semitransparentes: cada uno se oculta y se vuelve
a abrir tocando su título.

- **Drive** (izquierda): elegís quién dirige (al azar, clásico, IA o **vos**, diseñando cada jugada), ves la
  jugada elegida y el resultado de la última, con la cobertura real. En **⚙️ Ajustes** están el rival, la yarda
  de inicio y la semilla; en **📋 Jugadas**, todas las del drive.
- **Formaciones** (derecha): con "Mostrarlas en la cancha", la cancha muestra la formación que elijas.
- **Aprendizaje, Comparación y Libro de jugadas** (derecha): se llenan después de entrenar y evaluar.

Si en la PC de la defensa la jugada se ve a los saltos, en `app.py` cambiá `CADA_EN_VIVO = 1` por `2`.

## Cómo funciona

| | |
|---|---|
| **Agente** | El DT del ataque. Una red neuronal (2 capas de 128) entrenada con PPO. |
| **Episodio** | Un drive: hasta anotar, perder la pelota o patear. Cada paso es una jugada. |
| **Qué ve** (19 números) | Down, distancia, yarda, hash, personal defensivo, safeties que muestra, defensores en la caja, tendencia de blitz del rival en el drive y jugadas del drive. |
| **Qué NO ve** | La cobertura real (puede estar disfrazada) ni el perfil del rival. Por eso es parcialmente observable. |
| **Qué decide** (10 elecciones) | Formación, tipo (pase, carrera, patear), ruta de cada uno de los 5 receptores, a quién pasa, cuándo lanza y por qué hueco corre. |
| **Recompensa** | EPA: cuántos puntos esperados agregó la jugada. Es la métrica que usan los equipos de la NFL. |
| **Rival** | Un coordinador defensivo con reglas fijas: no aprende. Tiene 5 coberturas y 3 perfiles. |

El agente **no recibe ninguna jugada ni regla táctica**: solo las reglas del juego y la recompensa.

## Con qué vara medirlo

Sin entrenar nada, 400 drives por DT y por rival (puntos por drive):

| DT | Equilibrado | Agresivo | Conservador |
|---|---|---|---|
| Al azar (patea en 4to) | 1,24 | 1,18 | 1,31 |
| Clásico (reglas, no mira la defensa) | 2,97 | 3,16 | 2,46 |

Para superar al clásico, el agente tiene que leer la defensa. La simulación lo permite, porque la mejor ruta
depende de la cobertura (yardas por intento, pase al receptor de afuera, Shotgun contra Nickel):

| Ruta | Cover 0 | Cover 1 | Cover 2 | Cover 3 | Cover 4 |
|---|---|---|---|---|---|
| post | 26,3 | 30,4 | 12,4 | 16,7 | 13,5 |
| slant | 11,5 | 12,6 | 4,9 | 6,3 | 6,0 |
| flat | 7,6 | 6,9 | 9,4 | 14,8 | 10,6 |
| out | 2,0 | 2,8 | 4,2 | 8,0 | 4,6 |
| carrera (Formación I contra 4-3) | 4,4 | 3,9 | 4,0 | 5,9 | 7,7 |

Esta tabla se midió antes de los últimos ajustes de dificultad: sirve para ver el patrón, no los valores exactos.

## Qué podés tocar

Todo está en `config.py`:

| Sección | Para qué |
|---|---|
| `RECOMPENSA_EXTRA` | Sumar premios o castigos a la EPA: primer down, touchdown, intercepción, sack, costo por jugada. |
| `PERFILES_RIVAL`, `PROB_DISFRAZ` | Crear rivales, cambiar cuánto usa cada cobertura o cuánto disfraza. |
| `COMPLETO_*`, `INTERCEPCION_*`, `PROB_TACKLE_POR_PASO` | Hacer el juego más favorable al ataque o a la defensa. |
| `VELOCIDADES`, `ACELERACIONES` | Física de cada posición. |
| `PPO_PARAMS`, `RED_NEURONAL` | Cómo aprende: tasa de aprendizaje, exploración (`ent_coef`), tamaño de la red. |

Después de cambiar la recompensa, las reglas o la física hay que reentrenar para que el agente lo aprenda.
Un rival nuevo que agregues en `PERFILES_RIVAL` aparece solo en `--rival` y en el menú de la app.

## Limitaciones conocidas

- **La calibración es aproximada.** La simulación sigue siendo algo favorable al ataque: el clásico anota entre
  2,5 y 3,2 puntos por drive, contra unos 2 de la NFL. El porcentaje de pases completos da 44-51 %, contra ~64 %.
- **Los pases profundos rinden mucho y las rutas curl y comeback, poco.** Puede que el agente abuse de lo
  profundo. Si pasa, se corrige con `COMPLETO_CAIDA_POR_YARDA`.
- **Simplificaciones:** no hay reloj de partido, ni lesiones, ni penales, ni motion antes del snap. Los
  bloqueos están resueltos de forma simple y los jugadores pueden superponerse durante la jugada.
- **El agente decide todo antes del snap.** El QB no cambia de receptor durante la jugada.
- **El agente no tiene memoria entre jugadas**, salvo la tendencia de blitz del drive.
