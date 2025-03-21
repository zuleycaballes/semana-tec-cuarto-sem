# Autor: Zuleyca Soto A01741687
# Fecha: 19/3/2025
# Descripcion: Este programa convierte una imagen a escala de grises (si es a color), aplica zero padding ajustable 
# mediante el parámetro pad_factor y realiza una convolución con un kernel 3×3 (filtro de realce de bordes). Luego, 
# se muestran la imagen original, la imagen con padding y la imagen convolucionada.

import cv2
import numpy as np
import matplotlib.pyplot as plt

def convolution(image, kernel, average=False, pad_factor=1, verbose=False):
    # Si la imagen tiene 3 canales (color), se convierte a escala de grises.
    if len(image.shape) == 3:
        print("Found 3 Channels: {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size: {}".format(image.shape))
    else:
        print("Image Shape: {}".format(image.shape))
    
    print("Kernel Shape: {}".format(kernel.shape))
    
    # Visualización opcional de la imagen original
    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')
        plt.show()
    
    # Obtener las dimensiones de la imagen y el kernel
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    
    # Crear un arreglo de salida con el mismo tamaño que la imagen original
    output = np.zeros(image.shape)
    
    # Calcular el padding mínimo para mantener el tamaño original de la imagen.
    # Para un kernel 3x3, el padding mínimo es 1 píxel en cada lado.
    min_pad_height = int((kernel_row - 1) / 2)
    min_pad_width = int((kernel_col - 1) / 2)
    
    # Ajustar el padding multiplicando el valor mínimo por el factor deseado.
    # Si pad_factor es 1, se utiliza el padding mínimo; si es mayor, se agrega más padding.
    pad_height = min_pad_height * pad_factor
    pad_width = min_pad_width * pad_factor
    
    # Aplicar zero padding a la imagen, añadiendo una franja de ceros alrededor
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, 
                 pad_width:padded_image.shape[1] - pad_width] = image
    
    # Visualización opcional de la imagen con padding
    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image (pad_factor = {})".format(pad_factor))
        plt.axis('off')
        plt.show()
    
    # Realizar la convolución recorriendo la imagen original
    # Para cada píxel de la imagen original, se toma una región del padded_image del mismo tamaño que el kernel
    for row in range(image_row):
        for col in range(image_col):
            region = padded_image[row:row + kernel_row, col:col + kernel_col]
            # Se aplica el kernel multiplicando elemento a elemento y sumando el resultado
            output[row, col] = np.sum(kernel * region)
            # Si se activa la opción 'average', se promedia el resultado
            if average:
                output[row, col] /= (kernel_row * kernel_col)
    
    print("Output Image size: {}".format(output.shape))
    
    # Visualización de la imagen resultante de la convolución
    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}x{} Kernel".format(kernel_row, kernel_col))
        plt.axis('off')
        plt.show()
    
    return output

# Ejemplo de uso:
# Cargar la imagen en escala de grises 
img = cv2.imread(r"C:\Users\Zuleyca Soto\Desktop\Dia2\semana-tec-cuarto-sem\Actividades\Turquia.jpg", cv2.IMREAD_GRAYSCALE)

# Definir un kernel 3x3 para realce de bordes
kernel = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])

# Ejecutar la función de convolución.
# Cambia 'pad_factor' para aumentar o disminuir la cantidad de padding (1 = mínimo, >1 = más padding)
output_image = convolution(img, kernel, average=False, pad_factor=5, verbose=True)

# Visualización final (tres subgráficos: imagen original, imagen con padding y imagen convolucionada)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Imagen Original")
plt.axis('off')

plt.subplot(1, 3, 2)
# Se muestra la imagen con padding, usando el mismo cálculo del padding anterior
pad = ((kernel.shape[0] - 1) // 2) * 5
padded = np.pad(img, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
plt.imshow(padded, cmap='gray')
plt.title("Imagen con Padding ({} píxeles)".format(pad))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(output_image, cmap='gray')
plt.title("Imagen Convolucionada")
plt.axis('off')

plt.tight_layout()
plt.show()