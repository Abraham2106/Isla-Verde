# Proyecto Isla Verde v2.0 — Particionamiento Resiliente de la Red Eléctrica del ICE de Costa Rica en la Era NISQ

**Quantathon 2026 — Reto 1 (Open Quantum Institute, OQI)**

Pipeline reproducible que modela la red de transmisión de alta tensión del ICE (Instituto Costarricense de Electricidad), la reduce a escala NISQ y formula el problema de *islanding* controlado como un modelo QUBO/Ising, con líneas base clásicas verificables (Fuerza Bruta, Greedy y Goemans-Williamson).

---

## 1. Introducción y Objetivos de Desarrollo Sostenible (ODS)

Las fallas en cascada constituyen el modo de colapso más severo de un sistema eléctrico de potencia: la pérdida de una línea de transmisión redistribuye los flujos sobre las líneas restantes, sobrecargándolas y provocando desconexiones sucesivas que pueden culminar en un apagón nacional. El *islanding* controlado (particionamiento intencional de la red) es la contramedida de última línea: ante una contingencia crítica, la red se divide preventivamente en islas eléctricamente autosuficientes, cortando el mínimo número de líneas de interconexión posible. Matemáticamente, encontrar la partición óptima equivale a un problema de corte en grafos (familia Max-Cut/Min-Cut con restricciones de balance), que es NP-hard y por tanto un candidato natural para la optimización cuántica en hardware NISQ mediante formulaciones QUBO/Ising.

Este proyecto aborda el caso concreto de la red de transmisión de 230 kV del ICE de Costa Rica. A partir de los datasets oficiales de subestaciones y líneas de transmisión provistos por la hackatón, el pipeline construye el grafo físico de la red, lo reduce geoespacialmente mediante mallas hexagonales H3 hasta un tamaño tratable por procesadores cuánticos actuales (decenas de qubits), y genera la matriz QUBO y los coeficientes de Ising listos para ser consumidos por solucionadores cuánticos (QAOA, quantum annealing) o clásicos. La comparación rigurosa contra el óptimo exacto y contra el mejor algoritmo clásico aproximado conocido garantiza que toda afirmación de ventaja o paridad cuántica sea empíricamente verificable.

La relevancia del proyecto se alinea directamente con tres Objetivos de Desarrollo Sostenible de la ONU. Con el **ODS 7 (Energía asequible y no contaminante)**: Costa Rica genera cerca del 98 % de su electricidad de fuentes renovables, y proteger la continuidad del suministro protege ese logro, pues cada apagón evitado evita también el despacho de generación térmica de respaldo. Con el **ODS 9 (Industria, innovación e infraestructura)**: el islanding controlado convierte una red de transmisión existente en infraestructura resiliente por diseño, aplicando computación cuántica a un activo crítico nacional. Y con el **ODS 13 (Acción por el clima)**: una red renovable confiable es condición necesaria para la electrificación del transporte y la industria; además, la resiliencia ante eventos extremos (tormentas, sismos) es una forma directa de adaptación climática de la infraestructura energética.

## 2. Estructura del Repositorio

```
isla-verde-v2/
├── data/                      # Datasets oficiales provistos por la hackatón
│   ├── Subestaciones.csv
│   └── LineasDeTransmision.csv
├── modelador_red.py           # Pipeline clásico de datos (producción)
├── requirements.txt           # Dependencias pineadas (Python 3.12)
├── .gitignore                 # Exclusiones estándar de Python
└── README.md                  # Este documento
```

## 3. Instrucciones de Instalación

Requisito previo: **Python 3.12** (CPython, 64 bits). Verifique su versión con `python3 --version`.

**Paso 1 — Clonar el repositorio:**

```bash
git clone https://github.com/<usuario>/isla-verde-v2.git
cd isla-verde-v2
```

**Paso 2 — Crear un entorno virtual limpio:**

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Paso 3 — Actualizar pip e instalar las dependencias pineadas:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Todas las versiones de `requirements.txt` publican *wheels* precompilados para CPython 3.12 en Linux, macOS y Windows, por lo que la instalación no requiere compiladores locales y es idéntica en cualquier máquina limpia.

**Paso 4 — Verificación rápida del entorno:**

```bash
python -c "import networkx, pandas, numpy, matplotlib, cvxpy, h3; print('Entorno OK')"
```

## 4. Guía de Ejecución Rápida

Con el entorno virtual activo y los datasets oficiales en `./data`, el pipeline completo se ejecuta con un solo comando:

```bash
python3 modelador_red.py --data-dir ./data --out-dir ./scratch
```

Los parámetros son: `--data-dir` para el directorio con los CSV oficiales del ICE (por defecto `./data`) y `--out-dir` para el directorio de salida de artefactos —grafos serializados, matrices QUBO, figuras y métricas— (por defecto `./scratch`, excluido del control de versiones por `.gitignore`).

## 5. Descripción del Pipeline (`modelador_red.py`)

El script ejecuta cuatro fases secuenciales y deterministas:

**Fase 1 — Ingesta y normalización de nombres.** Lee `Subestaciones.csv` y `LineasDeTransmision.csv` y resuelve las inconsistencias de nomenclatura entre ambos datasets (variantes ortográficas, acentos, abreviaturas y espacios) mediante normalización canónica de cadenas, de modo que cada extremo de línea se vincule de forma inequívoca con su subestación.

**Fase 2 — Construcción del grafo físico y filtrado de 230 kV.** Modela la red en NetworkX como un grafo no dirigido donde los nodos son subestaciones georreferenciadas y las aristas son líneas de transmisión con sus atributos eléctricos. Se filtra el nivel de tensión de 230 kV, que constituye el anillo troncal del sistema nacional y es el nivel relevante para el análisis de fallas en cascada.

**Fase 3 — Reducción geoespacial con mallas H3.** Proyecta las subestaciones sobre la malla hexagonal jerárquica H3 y agrega nodos por celda, contrayendo el grafo a una escala compatible con hardware NISQ (número de nodos ≈ número de qubits disponibles) mientras preserva la topología de interconexión y los pesos agregados de las líneas entre regiones.

**Fase 4 — Formulación matemática QUBO/Ising.** Traduce el problema de particionamiento a la función objetivo cuadrática binaria: minimizar el peso de las líneas cortadas entre islas sujeto a penalizaciones de balance de carga/generación. Genera la matriz **Q** del QUBO y, mediante el cambio de variable `s = 2x − 1`, los coeficientes equivalentes del hamiltoniano de Ising (**h**, **J**), listos para QAOA o *annealing*.


## 8. Referencias

Goemans, M. X., & Williamson, D. P. (1995). *Improved approximation algorithms for maximum cut and satisfiability problems using semidefinite programming*. Journal of the ACM, 42(6), 1115–1145. — Lucas, A. (2014). *Ising formulations of many NP problems*. Frontiers in Physics, 2, 5. — Uber Technologies. *H3: Hexagonal hierarchical geospatial indexing system*. https://h3geo.org — Hagberg, A., Schult, D., & Swart, P. (2008). *Exploring network structure, dynamics, and function using NetworkX*.

---

**Equipo Isla Verde — Quantathon 2026, Reto 1 (OQI)**
