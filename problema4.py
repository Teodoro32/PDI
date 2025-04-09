
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image_and_histogram(original, equalized, title_prefix):
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    axs[0, 0].imshow(original, cmap='gray')
    axs[0, 0].set_title(f'{title_prefix} Original')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(equalized, cmap='gray')
    axs[0, 1].set_title(f'{title_prefix} Equalizada')
    axs[0, 1].axis('off')

    axs[1, 0].hist(original.ravel(), bins=256, range=[0, 256], color='gray')
    axs[1, 0].set_title('Histograma Original')
    
    axs[1, 1].hist(equalized.ravel(), bins=256, range=[0, 256], color='gray')
    axs[1, 1].set_title('Histograma Equalizado')

    plt.tight_layout()
    plt.savefig(f'{title_prefix}.png')
    plt.close()

def apply_clahe(image, tile_grid_size):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def process_image(path, name):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Parte (a): Equalização global
    eq_global = cv2.equalizeHist(img)
    show_image_and_histogram(img, eq_global, f"{name}_global")

    # Parte (b): Equalização local
    eq_clahe_5x5 = apply_clahe(img, (5, 5))
    show_image_and_histogram(img, eq_clahe_5x5, f"{name}_clahe_5x5")

    eq_clahe_7x7 = apply_clahe(img, (7, 7))
    show_image_and_histogram(img, eq_clahe_7x7, f"{name}_clahe_7x7")

# Processar ambas as imagens
process_image("pratica_2_1_1.png", "imagem1")
process_image("pratica_2_1_2.png", "imagem2")
