import numpy as np
import time
import concurrent.futures


# Generación de matrices de tamaño grande con valores aleatorios de 6 dígitos
#def generate_large_matrix(n):
 #   return np.random.randint(100000, 999999, size=(n, n), dtype=np.int64)

# Cargar matriz desde archivo en función del tamaño
def load_matrix_from_file(prefix, size):
    filename = f"matriz_{prefix}{size}.txt"
    return np.loadtxt(filename, dtype=np.int64)

# 1. Strassen-Winograd
def strassen_winograd(A, B):
    n = len(A)
    C = np.zeros((n, n), dtype=int)
    if n == 1:
        C[0, 0] = A[0, 0] * B[0, 0]
        return C
    if n % 2 != 0:
        raise ValueError("Strassen-Winograd necesita matrices de tamaño potencia de 2")

    mid = n // 2
    A11, A12, A21, A22 = A[:mid, :mid], A[:mid, mid:], A[mid:, :mid], A[mid:, mid:]
    B11, B12, B21, B22 = B[:mid, :mid], B[:mid, mid:], B[mid:, :mid], B[mid:, mid:]

    M1 = strassen_winograd(A11 + A22, B11 + B22)
    M2 = strassen_winograd(A21 + A22, B11)
    M3 = strassen_winograd(A11, B12 - B22)
    M4 = strassen_winograd(A22, B21 - B11)
    M5 = strassen_winograd(A11 + A12, B22)
    M6 = strassen_winograd(A21 - A11, B11 + B12)
    M7 = strassen_winograd(A12 - A22, B21 + B22)

    C[:mid, :mid] = M1 + M4 - M5 + M7
    C[:mid, mid:] = M3 + M5
    C[mid:, :mid] = M2 + M4
    C[mid:, mid:] = M1 - M2 + M3 + M6
    return C


# 2. NaivLoopUnrollingFour
def naiv_loop_unroll_four(A, B):
    n = len(A)
    C = np.zeros((n, n), dtype=int)
    for i in range(0, n, 4):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
                if i + 1 < n:
                    C[i + 1, j] += A[i + 1, k] * B[k, j]
                if i + 2 < n:
                    C[i + 2, j] += A[i + 2, k] * B[k, j]
                if i + 3 < n:
                    C[i + 3, j] += A[i + 3, k] * B[k, j]
    return C


# 3. Winograd Scaled
def winograd_scaled(A, B):
    n = len(A)
    C = np.zeros((n, n), dtype=int)
    row_factor = [sum(A[i][2 * k] * A[i][2 * k + 1] for k in range(n // 2)) for i in range(n)]
    col_factor = [sum(B[2 * k][j] * B[2 * k + 1][j] for k in range(n // 2)) for j in range(n)]

    for i in range(n):
        for j in range(n):
            C[i][j] = -row_factor[i] - col_factor[j]
            for k in range(n // 2):
                C[i][j] += (A[i][2 * k] + B[2 * k + 1][j]) * (A[i][2 * k + 1] + B[2 * k][j])

    if n % 2 == 1:
        for i in range(n):
            for j in range(n):
                C[i][j] += A[i][n - 1] * B[n - 1][j]
    return C


# 4. Sequential Block Multiplication
def sequential_block(A, B, block_size):
    n = len(A)
    C = np.zeros((n, n), dtype=int)
    for ii in range(0, n, block_size):
        for jj in range(0, n, block_size):
            for kk in range(0, n, block_size):
                for i in range(ii, min(ii + block_size, n)):
                    for j in range(jj, min(jj + block_size, n)):
                        for k in range(kk, min(kk + block_size, n)):
                            C[i][j] += A[i][k] * B[k][j]
    return C


# 5. Enhanced Parallel Block Multiplication
def enhanced_parallel_block(A, B, block_size):
    n = len(A)
    C = np.zeros((n, n), dtype=int)

    def compute_block(ii, jj, kk):
        for i in range(ii, min(ii + block_size, n)):
            for j in range(jj, min(jj + block_size, n)):
                for k in range(kk, min(kk + block_size, n)):
                    C[i][j] += A[i][k] * B[k][j]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for ii in range(0, n, block_size):
            for jj in range(0, n, block_size):
                for kk in range(0, n, block_size):
                    executor.submit(compute_block, ii, jj, kk)
    return C

# 6. NaivLoopUnrollingTwo
def naiv_loop_unrolling_two(A, B):
    n = len(A)
    C = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            for k in range(0, n, 2):
                C[i, j] += A[i, k] * B[k, j]
                if k + 1 < n:
                    C[i, j] += A[i, k + 1] * B[k + 1, j]
    return C

# 7. WinogradOriginal
def winograd_original(A, B):
    n = len(A)
    C = np.zeros((n, n), dtype=int)
    row_factor = [sum(A[i][2 * j] * A[i][2 * j + 1] for j in range(n // 2)) for i in range(n)]
    col_factor = [sum(B[2 * i][j] * B[2 * i + 1][j] for i in range(n // 2)) for j in range(n)]
    for i in range(n):
        for j in range(n):
            C[i][j] = -row_factor[i] - col_factor[j]
            for k in range(n // 2):
                C[i][j] += (A[i][2 * k] + B[2 * k + 1][j]) * (A[i][2 * k + 1] + B[2 * k][j])
    if n % 2 == 1:
        for i in range(n):
            for j in range(n):
                C[i][j] += A[i][n - 1] * B[n - 1][j]
    return C

# 8. StrassenNaiv
def strassen_naiv(A, B):
    n = len(A)
    if n == 1:
        return A * B
    if n % 2 != 0:
        raise ValueError("El tamaño de la matriz debe ser una potencia de 2 para Strassen")
    mid = n // 2
    A11, A12, A21, A22 = A[:mid, :mid], A[:mid, mid:], A[mid:, :mid], A[mid:, mid:]
    B11, B12, B21, B22 = B[:mid, :mid], B[:mid, mid:], B[mid:, :mid], B[mid:, mid:]
    M1 = strassen_naiv(A11 + A22, B11 + B22)
    M2 = strassen_naiv(A21 + A22, B11)
    M3 = strassen_naiv(A11, B12 - B22)
    M4 = strassen_naiv(A22, B21 - B11)
    M5 = strassen_naiv(A11 + A12, B22)
    M6 = strassen_naiv(A21 - A11, B11 + B12)
    M7 = strassen_naiv(A12 - A22, B21 + B22)
    C = np.zeros((n, n), dtype=int)
    C[:mid, :mid] = M1 + M4 - M5 + M7
    C[:mid, mid:] = M3 + M5
    C[mid:, :mid] = M2 + M4
    C[mid:, mid:] = M1 - M2 + M3 + M6
    return C

# 9. Sequential Block III.3
def sequential_block_3(A, B, block_size):
    n = len(A)
    C = np.zeros((n, n), dtype=int)
    for ii in range(0, n, block_size):
        for jj in range(0, n, block_size):
            for kk in range(0, n, block_size):
                for i in range(ii, min(ii + block_size, n)):
                    for j in range(jj, min(jj + block_size, n)):
                        for k in range(kk, min(kk + block_size, n)):
                            C[i, j] += A[i, k] * B[k, j]
    return C

# 10. III.5 Enhanced Parallel Block
def enhanced_parallel_block_v2(A, B, block_size):
    n = len(A)
    C = np.zeros((n, n), dtype=int)

    def compute_block(ii, jj, kk):
        for i in range(ii, min(ii + block_size, n)):
            for j in range(jj, min(jj + block_size, n)):
                for k in range(kk, min(kk + block_size, n)):
                    C[i, j] += A[i, k] * B[k, j]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for ii in range(0, n, block_size):
            for jj in range(0, n, block_size):
                for kk in range(0, n, block_size):
                    futures.append(executor.submit(compute_block, ii, jj, kk))
        concurrent.futures.wait(futures)

    return C

# Función para medir tiempo de ejecución
def measure_time(func, A, B, block_size=None):
    start = time.perf_counter()  # Comienza la medición con alta precisión
    if block_size:
        result = func(A, B, block_size)
    else:
        result = func(A, B)
    end = time.perf_counter()  # Finaliza la medición
    return (end - start)  * 1000000000  # Tiempo en nanosegundos

# Función para agregar tiempos de ejecución en un archivo .txt
def save_and_display_results(filename, results, matrix_size):
    with open(filename, 'w') as f:
        for name, time_ns in results:
            line = f"Tiempo de ejecucion ({name}) con tamano {matrix_size}x{matrix_size}: {time_ns:.0f} ns\n"
            f.write(line)
            print(line.strip())

# Función para imprimir una matriz
def print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{val:10d}" for val in row))
    print()

# Prueba de los algoritmos con matrices
matrix_size = 16 # Tamaño de la matriz
block_size = matrix_size//2 # Tamaño de bloque para los algoritmos de bloque
# Cargar matrices A y B desde archivos según el tamaño especificado
A = load_matrix_from_file('A', matrix_size)
B = load_matrix_from_file('B', matrix_size)

# Medición de tiempos
results = [
    ("Strassen-Winograd", measure_time(strassen_winograd, A, B)),
    ("NaivLoopUnrollingFour", measure_time(naiv_loop_unroll_four, A, B)),
    ("Winograd Scaled", measure_time(winograd_scaled, A, B)),
    ("IV.3 Sequential Block", measure_time(sequential_block, A, B, block_size)),
    ("IV.5 Enhanced Parallel Block", measure_time(enhanced_parallel_block, A, B, block_size)),
    ("NaivLoopUnrollingTwo", measure_time(naiv_loop_unrolling_two, A, B)),
    ("Winograd Original", measure_time(winograd_original, A, B)),
    ("Strassen-Naiv", measure_time(strassen_naiv, A, B)),
    ("III.3 Sequential Block V3", measure_time(sequential_block_3, A, B, block_size)),
    ("III.5 Enhanced Parallel Block V2", measure_time(enhanced_parallel_block_v2, A, B, block_size)),
]

# Guardar los resultados en un archivo .txt con el tamaño de la matriz
save_and_display_results("resultados_tiempos.txt", results, matrix_size)