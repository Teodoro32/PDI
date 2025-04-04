import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2

# 1. Funções auxiliares
def apply_gamma_correction(image, gamma=1.0):
    """Aplica correção gamma a uma imagem em tons de cinza"""
    image_normalized = (image - image.min()) / (image.max() - image.min())
    gamma_corrected = np.power(image_normalized, gamma)
    return gamma_corrected

def apply_gamma_rgb(image, gamma_r=1.0, gamma_g=1.0, gamma_b=1.0):
    """Aplica correção gamma separadamente a cada canal RGB"""
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    r_corrected = apply_gamma_correction(r, gamma_r)
    g_corrected = apply_gamma_correction(g, gamma_g)
    b_corrected = apply_gamma_correction(b, gamma_b)
    return np.stack([r_corrected, g_corrected, b_corrected], axis=2)

# 2. Carregar e processar imagens
print("Carregando imagens HDR...")
memorial_hdr = imageio.imread('hw1_memorial.hdr')
atrium_hdr = imageio.imread('hw1_atrium.hdr')

# Converter para tons de cinza
memorial_gray = cv2.cvtColor(memorial_hdr, cv2.COLOR_RGB2GRAY)
atrium_gray = cv2.cvtColor(atrium_hdr, cv2.COLOR_RGB2GRAY)

# 3. Visualização original (Parte b)
print("\nMostrando imagens originais em tons de cinza...")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(memorial_gray, cmap='gray')
plt.title('Memorial (cinza original)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(atrium_gray, cmap='gray')
plt.title('Atrium (cinza original)')
plt.axis('off')
plt.tight_layout()
plt.show()

# 4. Aplicação de gamma em tons de cinza (Parte c)
gamma = 0.5
print(f"\nAplicando correção gamma (γ={gamma}) em tons de cinza...")
memorial_gamma = apply_gamma_correction(memorial_gray, gamma)
atrium_gamma = apply_gamma_correction(atrium_gray, gamma)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(memorial_gamma, cmap='gray')
plt.title(f'Memorial com gamma={gamma}')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(atrium_gamma, cmap='gray')
plt.title(f'Atrium com gamma={gamma}')
plt.axis('off')
plt.tight_layout()
plt.show()

# 5. Aplicação de gamma em RGB (Parte d)
print("\nAplicando correção gamma em canais RGB...")

# Mesmo gamma para todos os canais
gamma = 0.5
memorial_rgb_gamma = apply_gamma_rgb(memorial_hdr, gamma, gamma, gamma)
atrium_rgb_gamma = apply_gamma_rgb(atrium_hdr, gamma, gamma, gamma)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(memorial_rgb_gamma)
plt.title(f'Memorial RGB com gamma={gamma} em todos canais')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(atrium_rgb_gamma)
plt.title(f'Atrium RGB com gamma={gamma} em todos canais')
plt.axis('off')
plt.tight_layout()
plt.show()

# Gammas diferentes por canal
gamma_r, gamma_g, gamma_b = 0.6, 0.5, 0.4
print(f"\nAplicando gammas diferentes por canal (R:{gamma_r}, G:{gamma_g}, B:{gamma_b})...")
memorial_rgb_diff = apply_gamma_rgb(memorial_hdr, gamma_r, gamma_g, gamma_b)
atrium_rgb_diff = apply_gamma_rgb(atrium_hdr, gamma_r, gamma_g, gamma_b)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(memorial_rgb_diff)
plt.title(f'Memorial RGB com gamma R:{gamma_r}, G:{gamma_g}, B:{gamma_b}')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(atrium_rgb_diff)
plt.title(f'Atrium RGB com gamma R:{gamma_r}, G:{gamma_g}, B:{gamma_b}')
plt.axis('off')
plt.tight_layout()
plt.show()

print("\nProcessamento concluído!")