import numpy as np
import time
import concurrent.futures


# Generación de matrices de tamaño grande con valores aleatorios de 6 dígitos
def generate_large_matrix(n):
    return np.random.randint(100000, 999999, size=(n, n), dtype=np.int64)


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


# Función para medir tiempo de ejecución
def measure_time(func, A, B, block_size=None):
    start = time.time()
    if block_size:
        result = func(A, B, block_size)
    else:
        result = func(A, B)
    end = time.time()
    return (end - start) * 1000000000  # Tiempo en nanosegundos


# Prueba de los algoritmos con matrices
matrix_size = 256  # Tamaño de la matriz
block_size = matrix_size//4 # Tamaño de bloque para los algoritmos de bloque
A = generate_large_matrix(matrix_size)
B = generate_large_matrix(matrix_size)

# Medición de tiempos
print("Tiempo de ejecución para cada algoritmo:")
print("1. Strassen-Winograd:", measure_time(strassen_winograd, A, B), "ns")
print("2. NaivLoopUnrollingFour:", measure_time(naiv_loop_unroll_four, A, B), "ns")
print("3. Winograd Scaled:", measure_time(winograd_scaled, A, B), "ns")
print("4. IV.3 Sequential Block:", measure_time(sequential_block, A, B, block_size), "ns")
print("5. IV.5 Enhanced Parallel Block:", measure_time(enhanced_parallel_block, A, B, block_size), "ns")
