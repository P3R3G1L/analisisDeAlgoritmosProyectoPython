import matplotlib.pyplot as plt


# Función para leer datos del archivo .txt y extraer nombres de algoritmos y tiempos de ejecución
def read_execution_times(filename):
    algorithms = []
    times = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("Tiempo de ejecucion"):
                parts = line.split(":")
                # Extraer nombre del algoritmo y tiempo
                name = parts[0].split("(")[1].split(")")[0]
                time_ns = int(parts[1].strip().replace(" ns", ""))

                algorithms.append(name)
                times.append(time_ns)

    return algorithms, times


# Función para graficar los tiempos de ejecución
def plot_execution_times(filename):
    algorithms, times = read_execution_times(filename)

    plt.figure(figsize=(12, 8))  # Tamaño de la gráfica
    plt.barh(algorithms, times, color='skyblue')  # Gráfica de barras horizontales

    plt.xlabel("Tiempo de ejecución (ns)")
    plt.ylabel("Algoritmo")
    plt.title("Tiempos de ejecución de algoritmos de multiplicación de matrices")
    plt.tight_layout()
    plt.show()


# Nombre del archivo .txt con los datos
filename = "resultados_tiempos.txt"
plot_execution_times(filename)
