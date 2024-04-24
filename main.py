import math
import time
from functions import *
import matplotlib.pyplot as plt

a1 = 11
a2 = -1
a3 = -1
N = 991
A = createMatrix(N, a1, a2, a3)
b = [math.sin(i*7) for i in range(N)]
err_norm = 1e-9

print("Zadanie B")
start = time.time()
x_Jacobi, counter_Jacobi = Jacobi(A, b, err_norm)
end = time.time()
print("Czas wykonania: ", end-start)
print("Liczba iteracji: ", counter_Jacobi)
print()

start = time.time()
x_Gauss, counter_Gauss = Gauss_Seidel(A, b, err_norm)
end = time.time()
print("Czas wykonania: ", end-start)
print("Liczba iteracji: ", counter_Gauss)
print()

plt.figure()
plt.yscale('log')
plt.plot(range(len(x_Jacobi)), x_Jacobi, label='Jacobi')
plt.plot(range(len(x_Gauss)), x_Gauss, label='Gauss-Seidel')
plt.xlabel('Numer iteracji')
plt.ylabel('Norma residuum')
plt.legend()
plt.show()

a1 = 3
A = createMatrix(N, a1, a2, a3)
x_Jacobi, counter_Jacobi = Jacobi(A, b, err_norm)
x_Gauss, counter_Gauss = Gauss_Seidel(A, b, err_norm)
plt.figure()
plt.yscale('log')
plt.plot(range(len(x_Jacobi)), x_Jacobi, label='Jacobi')
plt.plot(range(len(x_Gauss)), x_Gauss, label='Gauss-Seidel')
plt.xlabel('Numer iteracji')
plt.ylabel('Norma residuum')
plt.legend()
plt.show()

print("Zadanie D")
start = time.time()
res_norm = LU(A, b)
end = time.time()
print("Czas wykonania: ", end-start)
print("Norma residuum: ", res_norm)
print()

print("Zadanie E")
N = [100, 500, 1000, 2000, 3000]
a1 = 11
time_jacobi = []
time_gauss_seidel = []
time_LU = []

for n in N:
    print("Rozmiar: ", n)
    A = createMatrix(n, a1, a2, a3)
    b = [math.sin(i*7) for i in range(n)]

    start_time = time.time()
    Jacobi(A, b, err_norm)
    time_jacobi.append(time.time() - start_time)

    start_time = time.time()
    Gauss_Seidel(A, b, err_norm)
    time_gauss_seidel.append(time.time() - start_time)

    start_time = time.time()
    LU(A, b)
    time_LU.append(time.time() - start_time)

plt.plot(N, time_jacobi, label="Jacobi", color="red")
plt.plot(N, time_gauss_seidel, label="Gauss-Seidel", color="green")
plt.plot(N, time_LU, label="LU", color="blue")
plt.legend()
plt.grid(True)
plt.xlabel('Rozmiar macierzy')
plt.ylabel('Czas wykonania [s]')
plt.show()


