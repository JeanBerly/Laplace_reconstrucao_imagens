from PIL import Image
import numpy as np

class ImageHandler:
    """Responsável por carregar, converter e salvar imagens."""
    def __init__(self, filepath):
        self.filepath = filepath

    def carregar_imagem(self):
        try:
            img = Image.open(self.filepath)
            img_array = np.array(img)
            print(f"Original image array shape: {img_array.shape}")
            
            r_matrix = img_array[:, :, 0].copy()
            g_matrix = img_array[:, :, 1].copy()
            b_matrix = img_array[:, :, 2].copy()
            
            print(f"Red channel matrix shape: {r_matrix.shape}")
            print("\nSuccessfully created R, G, and B matrices!")
            return img_array, r_matrix, g_matrix, b_matrix
        except FileNotFoundError:
            print(f"Error: The image file {self.filepath} was not found. Please check the path.")
            raise

    @staticmethod
    def salvar_imagem(img_array, filepath):
        new_image = Image.fromarray(img_array)
        new_image.save(filepath)


class DamageSimulator:
    """Responsável por criar a região faltante na imagem."""
    def __init__(self, tamanho_quadrado):
        self.tamanho_quadrado = tamanho_quadrado

    def aplicar_dano(self, img_array, r_matrix, g_matrix, b_matrix):
        tamanho_eixo_x = img_array.shape[0]
        tamanho_eixo_y = img_array.shape[1]
        
        valor_x = np.random.randint(1, tamanho_eixo_x // 2)
        valor_y = np.random.randint(1, tamanho_eixo_y // 2)
        
        for i in range(valor_x, valor_x + self.tamanho_quadrado):
            for j in range(valor_y, valor_y + self.tamanho_quadrado):
                r_matrix[i][j] = 255
                g_matrix[i][j] = 255
                b_matrix[i][j] = 255
                
        img_array[:, :, 0] = r_matrix
        img_array[:, :, 1] = g_matrix
        img_array[:, :, 2] = b_matrix
        
        return valor_x, valor_y


class LaplaceAssembler:
    """Responsável por montar a matriz de coeficientes e os vetores de fronteira."""
    def __init__(self, tamanho_quadrado):
        self.tamanho_quadrado = tamanho_quadrado

    def _montar_matriz_a(self):
        """Monta a matriz de coeficientes (Operador Laplaciano em diferenças finitas) apenas uma vez."""
        tamanho_total = self.tamanho_quadrado * self.tamanho_quadrado
        matriz_a = np.zeros((tamanho_total, tamanho_total))
        
        for l in range(1, tamanho_total + 1):
            i = int(np.ceil(l / self.tamanho_quadrado))
            j = int(l % self.tamanho_quadrado)
            if j == 0: j = self.tamanho_quadrado
            if i == 0: i = 1
            
            for k in range(1, 6):
                if k == 1 and i + 1 < self.tamanho_quadrado + 1:
                    matriz_a[(i-1)*self.tamanho_quadrado + (j-1), i*self.tamanho_quadrado + (j-1)] = 1
                if k == 2 and i - 1 > 0:
                    matriz_a[(i-1)*self.tamanho_quadrado + (j-1), (i-2)*self.tamanho_quadrado + (j-1)] = 1
                if k == 3 and j + 1 < self.tamanho_quadrado + 1:
                    matriz_a[(i-1)*self.tamanho_quadrado + (j-1), (i-1)*self.tamanho_quadrado + j] = 1
                if k == 4 and j - 1 > 0:
                    matriz_a[(i-1)*self.tamanho_quadrado + (j-1), (i-1)*self.tamanho_quadrado + (j-2)] = 1
                if k == 5:
                    matriz_a[(i-1)*self.tamanho_quadrado + (j-1), (i-1)*self.tamanho_quadrado + (j-1)] = -4
                    
        return matriz_a

    def _montar_vetor_b(self, canal_matrix, valor_x, valor_y):
        """Extrai as condições de contorno (bordas do buraco) para parametrizar o vetor independente."""
        tamanho_total = self.tamanho_quadrado * self.tamanho_quadrado
        vetor_b = np.zeros(tamanho_total)
        
        for l in range(1, tamanho_total + 1):
            i = int(np.ceil(l / self.tamanho_quadrado))
            j = int(l % self.tamanho_quadrado)
            if j == 0: j = self.tamanho_quadrado
            if i == 0: i = 1
            
            if i - 1 == 0:
                vetor_b[(i-1)*self.tamanho_quadrado + (j-1)] -= canal_matrix[valor_x - 1, valor_y + (j-1)]
            if i + 1 == self.tamanho_quadrado + 1:
                vetor_b[(i-1)*self.tamanho_quadrado + (j-1)] -= canal_matrix[valor_x + self.tamanho_quadrado, valor_y + (j-1)]
            if j - 1 == 0:
                vetor_b[(i-1)*self.tamanho_quadrado + (j-1)] -= canal_matrix[valor_x + (i-1), valor_y - 1]
            if j + 1 == self.tamanho_quadrado + 1:
                vetor_b[(i-1)*self.tamanho_quadrado + (j-1)] -= canal_matrix[valor_x + (i-1), valor_y + self.tamanho_quadrado]
                
        return vetor_b

    def montar_sistema(self, r_matrix, g_matrix, b_matrix, valor_x, valor_y):
        matriz_a = self._montar_matriz_a()
        
        vetor_b_vermelho = self._montar_vetor_b(r_matrix, valor_x, valor_y)
        vetor_b_verde = self._montar_vetor_b(g_matrix, valor_x, valor_y)
        vetor_b_azul = self._montar_vetor_b(b_matrix, valor_x, valor_y)

        return matriz_a, vetor_b_vermelho, vetor_b_verde, vetor_b_azul


class LaplaceSolver:
    """Responsável por resolver a EDP discretizada e recompor os pixels."""
    def __init__(self, tamanho_quadrado):
        self.tamanho_quadrado = tamanho_quadrado

    def resolver_sistema(self, matriz_a, b_vermelho, b_verde, b_azul):
        # Resolve o sistema iterando sobre os três vetores de cores
        x_vermelho, x_verde, x_azul = [np.linalg.solve(matriz_a, b) for b in (b_vermelho, b_verde, b_azul)]
        
        print("Soluções calculadas com sucesso.")
        return x_vermelho, x_verde, x_azul

    def reconstruir_imagem(self, img_array, r_matrix, g_matrix, b_matrix, x_vermelho, x_verde, x_azul, valor_x, valor_y):
        for i in range(0, self.tamanho_quadrado):
            for j in range(0, self.tamanho_quadrado):
                r_val = x_vermelho[i*self.tamanho_quadrado + j]
                g_val = x_verde[i*self.tamanho_quadrado + j]
                b_val = x_azul[i*self.tamanho_quadrado + j]

                r_matrix[valor_x + i][valor_y + j] = int(np.clip(r_val, 0, 255))
                g_matrix[valor_x + i][valor_y + j] = int(np.clip(g_val, 0, 255))
                b_matrix[valor_x + i][valor_y + j] = int(np.clip(b_val, 0, 255))
                
        img_array[:, :, 0] = r_matrix
        img_array[:, :, 1] = g_matrix
        img_array[:, :, 2] = b_matrix
        return img_array


def main():
    ARQUIVO_ENTRADA = 'balao.jpg'
    TAMANHO_QUADRADO = 100

    image_handler = ImageHandler(ARQUIVO_ENTRADA)
    img_array, r_matrix, g_matrix, b_matrix = image_handler.carregar_imagem()

    damage_simulator = DamageSimulator(TAMANHO_QUADRADO)
    valor_x, valor_y = damage_simulator.aplicar_dano(img_array, r_matrix, g_matrix, b_matrix)
    image_handler.salvar_imagem(img_array, 'imagem_com_buraco.png')

    assembler = LaplaceAssembler(TAMANHO_QUADRADO)
    matriz_a, b_vermelho, b_verde, b_azul = assembler.montar_sistema(r_matrix, g_matrix, b_matrix, valor_x, valor_y)

    solver = LaplaceSolver(TAMANHO_QUADRADO)
    x_vermelho, x_verde, x_azul = solver.resolver_sistema(matriz_a, b_vermelho, b_verde, b_azul)
    
    img_reconstruida = solver.reconstruir_imagem(
        img_array, r_matrix, g_matrix, b_matrix, 
        x_vermelho, x_verde, x_azul, 
        valor_x, valor_y
    )
    image_handler.salvar_imagem(img_reconstruida, 'imagem_reconstruida.png')

if __name__ == "__main__":
    main()