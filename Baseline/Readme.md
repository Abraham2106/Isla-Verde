# Ejecución de los baselines:

Para ejecutar los baselines se deben ejecutar los siguientes comandos:

## Goemans Williamson
Línea base es el algoritmo de redondeo SDP de Goemans-Williamson (GW) (razón de aproximación mayor o igual a 0.878).
```bash
python goemans_williamson.py --graph ..\scratch\isla_verde_full_graph.json --iterations 500 --runs 30 --output results_GW
```

## Greedy
Línea base voraz (greedy) (razón aprox. 0.5).
```bash
python maxcut_baselines.py --graph ..\scratch\isla_verde_full_graph.json --algorithm greedy --output results_greedy
```

## Fuerza Bruta (Máximo 25 Nodos)
La línea base de fuerza bruta debe limitarce debido a su complejidad exponencial.
```bash
python maxcut_baselines.py --graph ..\scratch\isla_verde_full_graph.json --algorithm bruteforce --output results_bruteforce
```