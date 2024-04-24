from copy import deepcopy
import math

def createMatrix(N, a1, a2, a3):
    matrix = [[0 for i in range(N)] for j in range(N)]
    for i in range(N):
        matrix[i][i] = a1
        if i > 0:
            matrix[i][i-1] = a2
            matrix[i-1][i] = a2
        if i > 1:
            matrix[i][i-2] = a3
            matrix[i-2][i] = a3
    return matrix
        
def residual(A, b, x):
    N = len(b)
    res = [0 for i in range(N)]
    for i in range(N):
        res[i] = b[i]
        for j in range(N):
            res[i] -= A[i][j]*x[j]
    return res

def norm(v):
    return math.sqrt(sum([i*i for i in v]))

def Jacobi(A, b, err):
    N = len(b)
    counter = 0
    x = [1 for i in range(N)]
    x_new = [1 for i in range(N)]
    residuum = []
    while True:
        for i in range(N):
            x_new[i] = b[i]
            for j in range(N):
                if i != j:
                    x_new[i] -= A[i][j]*x[j]
            x_new[i] /= A[i][i]
        x = x_new.copy()
        res_vect = norm(residual(A, b, x))
        if res_vect < err or counter > 1000:
            break
        counter += 1
        residuum.append(res_vect)
    return residuum, counter

def Gauss_Seidel(A, b, err):
    N = len(b)
    counter = 0
    x = [1 for i in range(N)]
    x_new = [1 for i in range(N)]
    residuum = []
    while True:
        for i in range(N):
            x_new[i] = b[i]
            for j in range(N):
                if i != j:
                    x_new[i] -= A[i][j]*x_new[j]
            x_new[i] /= A[i][i]
        x = x_new.copy()
        res_vect = norm(residual(A, b, x))
        if res_vect < err or counter > 1000:
            break
        counter += 1
        residuum.append(res_vect)
    return residuum, counter

def eye(N):
    return [[1 if i == j else 0 for i in range(N)] for j in range(N)]

def LU(A, b):
    U = deepcopy(A)
    L = eye(len(A))
    x = [1 for i in range(len(A))]
    y = [0 for i in range(len(A))]
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            L[j][i] = U[j][i]/U[i][i]
            for k in range(i, len(A)):
                U[j][k] -= L[j][i]*U[i][k]
                
    for i in range(len(A)):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i][j]*y[j]
        y[i] /= L[i][i]
        
    for i in range(len(A)-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1, len(A)):
            x[i] -= U[i][j]*x[j]
        x[i] /= U[i][i]
        
    res_norm = norm(residual(A, b, x))
    return res_norm
