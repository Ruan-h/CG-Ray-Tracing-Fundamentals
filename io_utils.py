import numpy as np


def save_ppm(filename, image_data, width, height):
    with open(filename, 'w') as f:
        f.write(f'P3\n{width} {height}\n255\n')
        for row in image_data:
            for pixel in row:
                r = int(np.clip(pixel[0] * 255.999, 0, 255))
                g = int(np.clip(pixel[1] * 255.999, 0, 255))
                b = int(np.clip(pixel[2] * 255.999, 0, 255))
                f.write(f'{r} {g} {b} ')
            f.write('\n')
    print(f"imagem salva em '{filename}'")
