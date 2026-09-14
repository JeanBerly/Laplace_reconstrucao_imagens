# Image Inpainting via Equação de Laplace

**Visão Geral:** Este projeto implementa uma técnica clássica de processamento de imagens (Inpainting) utilizando conceitos de Matemática Computacional. O objetivo é reconstruir regiões faltantes ou corrompidas de uma imagem resolvendo numericamente a **Equação de Laplace** através do **Método de Diferenças Finitas** e otimizando a solução do sistema linear resultante com o **Método do Gradiente Conjugado**.

## Fundamentação Matemática

O problema de preencher uma região faltante em uma imagem pode ser modelado como um problema de valor de contorno. A ideia central é que a transição de cores na região danificada deve ser o mais suave possível, respeitando as cores das bordas (pixels saudáveis ao redor do "buraco").

### 1. A Equação de Laplace
Para garantir essa suavidade máxima, minimizamos a variação da intensidade dos pixels. Isso é equivalente a exigir que o Laplaciano da função intensidade de cor seja zero na região interna da máscara. Para uma função de intensidade de pixel $u(x,y)$, a Equação de Laplace é dada por:

$$ \Delta u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0 $$

Esta equação diferencial parcial (EDP) elíptica garante que o valor de cada pixel interno seja aproximadamente a média dos seus vizinhos, promovendo uma interpolação suave. Isso se dá pois a solução é uma função harmônica e portanto vale o teorema da média.

### 2. Discretização via Diferenças Finitas
Como uma imagem já é um domínio discreto (um grid de pixels), aproximamos as derivadas parciais usando o esquema clássico de diferenças finitas centrais (estêncil de 5 pontos). Para um pixel na posição $(i, j)$, a aproximação discreta do Laplaciano igualada a zero resulta em:

$$ u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j} = 0 $$

Esta relação nos diz que o valor de $u_{i,j}$ depende estritamente dos seus quatro vizinhos diretos (acima, abaixo, esquerda e direita).

### 3. O Sistema Linear ($Ax = b$)
Aplicando o estêncil de diferenças finitas a todos os pixels desconhecidos (a região do "buraco"), construímos um sistema linear $Ax = b$, onde:

*   **Matriz A:** É a matriz de coeficientes do operador Laplaciano. Ela é altamente esparsa, simétrica e definida negativa. Cada linha representa a equação de um pixel, contendo o valor `-4` na diagonal principal e `1` nas posições correspondentes aos seus vizinhos internos.
*   **Vetor x:** Contém as intensidades desconhecidas dos pixels que queremos calcular.
*   **Vetor b:** Armazena as Condições de Contorno de Dirichlet. Quando o estêncil "toca" um pixel conhecido (a borda do buraco), esse valor conhecido é passado para o lado direito da equação, compondo o vetor $b$.

### 4. Otimização Numérica: Gradiente Conjugado
Devido à natureza da matriz $A$ (simétrica e esparsa), resolver o sistema usando a inversão direta clássica é computacionalmente proibitivo para imagens grandes ($O(N^3)$). Portanto, o projeto utiliza a biblioteca `scipy.sparse` para construir a matriz no formato LIL (List of Lists) e depois convertê-la para CSR (Compressed Sparse Row), permitindo cálculos algébricos rápidos.

A solução do sistema é obtida iterativamente utilizando o **Método do Gradiente Conjugado** (`scipy.sparse.linalg.cg`), que encontra a solução de forma extremamente eficiente para matrizes com essas propriedades.

## Estrutura do Código (Arquitetura SOLID)

O código foi refatorado seguindo os princípios de SOLID, focado na Responsabilidade Única (SRP) e DRY (Don't Repeat Yourself), quebrando o processo em componentes especialistas:

*   `ImageHandler`: Gerencia I/O de arquivos e manipulação de canais RGB via biblioteca Pillow e NumPy.
*   `DamageSimulator`: Responsável por gerar artificialmente uma região faltante ("buraco") na imagem para testes.
*   `LaplaceAssembler`: Isola a complexidade matemática da montagem da matriz Laplaciana esparsa e a extração do contorno de Dirichlet para o vetor $b$ (parametrizado para evitar duplicação de lógica entre os canais de cores).
*   `LaplaceSolver`: Resolve o sistema iterativamente com o Gradiente Conjugado e mapeia os resultados numéricos contínuos de volta para valores discretos de cor (0-255).

## Como Utilizar

### Pré-requisitos
Certifique-se de ter as seguintes bibliotecas instaladas em seu ambiente (preferencialmente utilizando o `uv` ou seu gerenciador favorito):

```bash
pip install numpy scipy pillow
