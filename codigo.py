from PIL import Image
import numpy as np
import scipy.sparse
from scipy.sparse.linalg import cg
def conjGrad(A,x,b,tol,N):

    r = b - A.dot(x)
    p = r.copy()
    for i in range(N):
        Ap = A.dot(p)
        alpha = np.dot(p,r)/np.dot(p,Ap)
        x = x + alpha*p
        r = b - A.dot(x)
        if np.sqrt(np.sum((r**2))) < tol:
            print('Itr:', i)
            break
        else:
            beta = -np.dot(r,Ap)/np.dot(p,Ap)
            p = r + beta*p
    return x 

# O tamanho do step é "inutil", porque os eixos ja estao discretizados, sao os pixels da imagem
# Transformar imagem em matriz(3 matrizes para cada R,G,B):
try:
    img = Image.open('balao.jpg')

    # Convert the image to a NumPy array
    # This creates a 3D array: (height, width, channels)
    img_array = np.array(img)

    # Print the shape to see its dimensions
    print(f"Original image array shape: {img_array.shape}")

    # Separate the channels using array slicing
    # R channel is at index 0, G at 1, B at 2
    r_matrix = img_array[:, :, 0]
    g_matrix = img_array[:, :, 1]
    b_matrix = img_array[:, :, 2]

    # Print the shape of one of the matrices to confirm it's 2D
    print(f"Red channel matrix shape: {r_matrix.shape}")
    print("\nSuccessfully created R, G, and B matrices!")

except FileNotFoundError:
    print("Error: The image file was not found. Please check the path.")
#------
# Definição de constantes
tamanho_eixo_x = int(img_array[1].size/3) # SUBSTITUIR
tamanho_eixo_y = int(img_array[0].size/3) # SUBSTITUIR
# Criar um quadrado\retangulo para remover da imagem e sabermos onde temos que resolver
tamanho_quadrado = 500
valor_x = (np.random.randint(1,tamanho_eixo_x/2))
valor_y = (np.random.randint(1,tamanho_eixo_y/2))
for i in range(valor_x, valor_x+ tamanho_quadrado):
    for j in range(valor_y, valor_y+tamanho_quadrado):
        r_matrix[i][j] = 255
        g_matrix[i][j] = 255
        b_matrix[i][j] = 255
new_image = Image.fromarray(img_array)
new_image.save('imagem_com_buraco.png')
# Resolver Laplace pras 3 cores num quadrado tamanho_quadradoxtamanho_quadrado
# VERMELHO (R)
matriz_coeficientes_diferenca_finita = scipy.sparse.lil_matrix((tamanho_quadrado* tamanho_quadrado, tamanho_quadrado * tamanho_quadrado))
vetor_b_vermelho = np.zeros(tamanho_quadrado*tamanho_quadrado) # b\
vetor_b_verde = np.zeros(tamanho_quadrado*tamanho_quadrado) # b
vetor_b_azul = np.zeros(tamanho_quadrado*tamanho_quadrado) # b
for l in range(1, (tamanho_quadrado* tamanho_quadrado)+1):
    i = int(np.ceil(l/tamanho_quadrado))
    j = int(l % tamanho_quadrado)
    if (j == 0): j = tamanho_quadrado
    if (i == 0): i = 1
    for k in range(1,6):
        # 1: u(i+1,j)
        if (k == 1 and i+1 < tamanho_quadrado+1):
            matriz_coeficientes_diferenca_finita[(i-1)*tamanho_quadrado + (j-1), i*tamanho_quadrado + (j-1)] = 1
        # 2: u(i-1,j)
        if (k == 2 and i-1 > 0):
            matriz_coeficientes_diferenca_finita[(i-1)*tamanho_quadrado + (j-1), (i-2)*tamanho_quadrado + (j-1)] = 1
        # 3: u(i,j+1)
        if (k == 3 and j+1 < tamanho_quadrado+1):
            matriz_coeficientes_diferenca_finita[(i-1)*tamanho_quadrado + (j-1), (i-1)*tamanho_quadrado + j] = 1
        # 4: u(i,j-1)
        if (k == 4 and j-1 > 0):
            matriz_coeficientes_diferenca_finita[(i-1)*tamanho_quadrado + (j-1), (i-1)*tamanho_quadrado + (j-2)] = 1
        # 5: u(i,j)
        if (k == 5):
            matriz_coeficientes_diferenca_finita[(i-1)*tamanho_quadrado + (j-1), (i-1)*tamanho_quadrado + (j-1)] = -4
    # Se i-1 == 0 entao vai pro vetor B associando o valor lá na matrix do rgb
    if (i-1 == 0): # u(i-1,j)
        vetor_b_vermelho[(i-1)*tamanho_quadrado+ (j-1)] -= r_matrix[valor_x -1,valor_y + (j-1)] # "CONDICAO INICIAL, VETOR B"
        vetor_b_verde[(i-1)*tamanho_quadrado+ (j-1)] -= g_matrix[valor_x -1,valor_y + (j-1)] # "CONDICAO INICIAL, VETOR B"
        vetor_b_azul[(i-1)*tamanho_quadrado+ (j-1)] -= b_matrix[valor_x -1,valor_y + (j-1)] # "CONDICAO INICIAL, VETOR B"
    # Se i+1 == tamanho_quadrado entao
    if (i+1 == tamanho_quadrado+1):# u(i+1,j)
        vetor_b_vermelho[(i-1) * tamanho_quadrado + (j-1)] -= r_matrix[valor_x + tamanho_quadrado, valor_y + (j-1)]
        vetor_b_verde[(i-1) * tamanho_quadrado + (j-1)] -= g_matrix[valor_x + tamanho_quadrado, valor_y + (j-1)]
        vetor_b_azul[(i-1) * tamanho_quadrado + (j-1)] -= b_matrix[valor_x + tamanho_quadrado, valor_y + (j-1)]
    # Se j-1 == 0 entao
    if (j-1 == 0):# u(i,j-1)
        vetor_b_vermelho[(i-1)*tamanho_quadrado + (j-1)] -= r_matrix[valor_x + (i-1), valor_y - 1]
        vetor_b_verde[(i-1)*tamanho_quadrado + (j-1)] -= g_matrix[valor_x + (i-1), valor_y - 1]
        vetor_b_azul[(i-1)*tamanho_quadrado + (j-1)] -= b_matrix[valor_x + (i-1), valor_y - 1]
    # se j+1 == tamanho_quadrado entaoi
    if (j+1 == tamanho_quadrado+1):
        vetor_b_vermelho[(i-1)*tamanho_quadrado + (j-1)] -= r_matrix[valor_x + (i-1), valor_y + tamanho_quadrado]
        vetor_b_verde[(i-1)*tamanho_quadrado + (j-1)] -= g_matrix[valor_x + (i-1), valor_y + tamanho_quadrado]
        vetor_b_azul[(i-1)*tamanho_quadrado + (j-1)] -= b_matrix[valor_x + (i-1), valor_y + tamanho_quadrado]
#SOLVER NAS 3, matriz A é a mesma, so muda o B
A_sparse = matriz_coeficientes_diferenca_finita.tocsr()
A_solver = -A_sparse
print("antes do solverrr!!!")
# Solve using the new matrix and the negated vectors
x_r, exit_code = cg(A_solver, -vetor_b_vermelho, rtol=1e-5)
x_g, exit_code = cg(A_solver, -vetor_b_verde, rtol=1e-5)
x_b, exit_code = cg(A_solver, -vetor_b_azul, rtol=1e-5)
#x_vermelho = np.linalg.solve(matriz_coeficientes_diferenca_finita, vetor_b_vermelho)
#x_verde = np.linalg.solve(matriz_coeficientes_diferenca_finita, vetor_b_verde)
#x_azul = np.linalg.solve(matriz_coeficientes_diferenca_finita, vetor_b_azul)
print(x_r)
print(x_g)
for i in range (0, tamanho_quadrado):
    for j in range(0, tamanho_quadrado):
        r_val = x_r[i*tamanho_quadrado + j]
        g_val = x_g[i*tamanho_quadrado + j]
        b_val = x_b[i*tamanho_quadrado + j]

        r_matrix[valor_x + i][valor_y + j] = int(np.clip(r_val, 0, 255))
        g_matrix[valor_x + i][valor_y + j] = int(np.clip(g_val, 0, 255))
        b_matrix[valor_x + i][valor_y + j] = int(np.clip(b_val, 0, 255))
imagem_reconstruida = Image.fromarray(img_array)
imagem_reconstruida.save('imagem_reconstruida.png')