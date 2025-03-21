# Autor: Zuleyca Soto A01741687
# Fecha: 19/3/2025
# Descripcion: Este código convierte la imagen a escala de grises y aplica una convolución sin padding usando un 
# kernel 3×3, procesando solo las regiones donde el kernel encaja completamente, y muestra el resultado.

import cv2
import numpy as np
import matplotlib.pyplot as plt

def convolution(image, kernel, average=False, verbose=False):
    # Convertir a escala de grises si la imagen tiene 3 canales
    if len(image.shape) == 3:
        print("Found 3 Channels: {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Converted to Gray Channel. Size: {}".format(image.shape))
    else:
        print("Image Shape: {}".format(image.shape))
    
    print("Kernel Shape: {}".format(kernel.shape))
    
    # Mostrar la imagen original si verbose es True
    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')
        plt.show()
    
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    # Calcular el tamaño de la imagen resultante sin padding:
    # Salida = (image_rows - kernel_rows + 1, image_cols - kernel_cols + 1)
    out_rows = image_row - kernel_row + 1
    out_cols = image_col - kernel_col + 1

    # Inicializar la imagen de salida con el tamaño calculado
    output = np.zeros((out_rows, out_cols))
    
    # Realizar la convolución solo en las posiciones "válidas"
    for row in range(out_rows):
        for col in range(out_cols):
            # Extraer la región de la imagen que coincide con el tamaño del kernel
            region = image[row:row+kernel_row, col:col+kernel_col]
            # Multiplicación elemento a elemento y suma de los resultados
            output[row, col] = np.sum(kernel * region)
            # Si se desea promediar el resultado, se divide entre el número de elementos
            if average:
                output[row, col] /= (kernel_row * kernel_col)
    
    print("Output Image size: {}".format(output.shape))
    
    # Mostrar la imagen resultante si verbose es True
    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image (No Padding)")
        plt.axis('off')
        plt.show()
    
    return output

# Ejemplo de uso:
# Cargar la imagen en escala de grises
img = cv2.imread(r"C:\Users\Zuleyca Soto\Desktop\Dia2\semana-tec-cuarto-sem\Actividades\Turquia.jpg", cv2.IMREAD_GRAYSCALE)

# Definir un kernel 3x3 (filtro de realce de bordes)
kernel = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])

# Ejecutar la función de convolución
output_image = convolution(img, kernel, average=False, verbose=True)