import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter, median_filter

# Carregar a imagem
img = cv2.imread("listras.png", cv2.IMREAD_GRAYSCALE)

# Função para aplicar filtros e exibir resultados
def aplicar_filtros(img):
    filtros = {
        "Média 3x3": uniform_filter(img, size=3),
        "Média 7x7": uniform_filter(img, size=7),
        "Máximo 3x3": maximum_filter(img, size=3),
        "Máximo 7x7": maximum_filter(img, size=7),
        "Mínimo 3x3": minimum_filter(img, size=3),
        "Mínimo 7x7": minimum_filter(img, size=7),
        "Mediana 3x3": median_filter(img, size=3),
        "Mediana 7x7": median_filter(img, size=7),
    }

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.ravel()
    for i, (title, filtered) in enumerate(filtros.items()):
        axs[i].imshow(filtered, cmap="gray")
        axs[i].set_title(title)
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()
    plt.tight_layout()
    plt.savefig("resultado_listras.png")  # Salva como imagem


aplicar_filtros(img)
