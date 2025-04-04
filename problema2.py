import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Parte (a) - Implementação do algoritmo halftoning
# Padrões halftoning 3x3 para 10 níveis de cinza
HALFTONE_PATTERNS = [
    np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),  # Nível 0 (preto)
    np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),  # Nível 1
    np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]]),  # Nível 2
    np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]),  # Nível 3
    np.array([[1, 0, 0], [1, 0, 1], [1, 0, 0]]),  # Nível 4
    np.array([[1, 0, 1], [1, 0, 1], [1, 0, 0]]),  # Nível 5
    np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]]),  # Nível 6
    np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1]]),  # Nível 7
    np.array([[1, 1, 1], [1, 1, 1], [1, 0, 1]]),  # Nível 8
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])   # Nível 9 (branco)
]

def halftone(image):
    """Aplica halftoning a uma imagem em tons de cinza"""
    # Normalizar a imagem para 0-9
    img_normalized = ((image.astype(float) - image.min()) * 9 / 
                     (image.max() - image.min())).astype(int)
    
    # Criar imagem de saída
    h, w = img_normalized.shape
    output = np.zeros((h*3, w*3), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            level = img_normalized[i, j]
            level = max(0, min(9, level))  # Garantir que o nível está entre 0-9
            output[i*3:(i+1)*3, j*3:(j+1)*3] = HALFTONE_PATTERNS[level] * 255
    
    return output

# Parte (b) - Imagem de teste 256x256
def create_test_image():
    """Cria imagem de teste com quadrados de 16x16 e valores de 0 a 255"""
    image = np.zeros((256, 256), dtype=np.uint8)
    
    for i in range(16):
        for j in range(16):
            value = i * 16 + j
            image[i*16:(i+1)*16, j*16:(j+1)*16] = value
    
    return image


# Parte (c) - Processamento das imagens TIFF do livro
def process_tiff_images():
    """Processa as imagens TIFF do livro"""
    print("\nVerificando arquivos TIFF no diretório...")
    tiff_files = [f for f in os.listdir() if f.lower().endswith('.tif')]
    print(f"Arquivos TIFF encontrados: {tiff_files or 'Nenhum'}")

    required_files = {
        'cameraman': 'Fig0222(b)(cameraman).tif',
        'crowd': 'Fig0222(c)(crowd).tif'
    }
    
    # Verifica quais arquivos estão faltando
    missing_files = [f for f in required_files.values() if not os.path.exists(f)]
    
    if missing_files:
        print("\nAVISO: Arquivos TIFF faltando:")
        for f in missing_files:
            print(f"- {f}")
        print("\nExecute a parte (b) ou coloque os arquivos na pasta.")
        return
    
    try:
        print("\nProcessando imagens TIFF...")
        
        # Processa Cameraman
        cameraman = np.array(Image.open(required_files['cameraman']).convert('L'))
        halftoned_cam = halftone(cameraman)
        plot_comparison(cameraman, halftoned_cam, "Cameraman Original vs Halftone")
        
        # Processa Crowd
        crowd = np.array(Image.open(required_files['crowd']).convert('L'))
        halftoned_crowd = halftone(crowd)
        plot_comparison(crowd, halftoned_crowd, "Crowd Original vs Halftone")
        
    except Exception as e:
        print(f"\nErro ao processar TIFFs: {str(e)}")


# Funções auxiliares
def plot_comparison(original, halftoned, title):
    """Mostra comparação lado a lado"""
    plt.figure(figsize=(12, 6))
    plt.suptitle(title, fontsize=14)
    
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray', vmin=0, vmax=255)
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(halftoned, cmap='gray', vmin=0, vmax=255)
    plt.title('Halftoned (3x aumentada)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Execução principal
if __name__ == "__main__":
    print("=== HALFTONING PROGRAM ===")
    print("Este programa executa:")
    print("(a) Algoritmo halftoning com padrões 3x3")
    print("(b) Gera imagem de teste 256x256")
    print("(c) Processa imagens TIFF do livro (cameraman e crowd)\n")
    
    # Parte (b) - Imagem de teste
    print("\n=== EXECUTANDO PARTE (b) ===")
    test_image = create_test_image()
    halftoned_test = halftone(test_image)
    plot_comparison(test_image, halftoned_test, "Imagem de Teste 256x256")
    
    # Parte (c) - Imagens TIFF
    print("\n=== EXECUTANDO PARTE (c) ===")
    process_tiff_images()
    
    print("\n=== FINALIZADO ===")