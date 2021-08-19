from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

d = 2

_ = np.full((2 * d, 2 * d), 255)
B = np.full((2 * d, 2 * d), 0)

meander = np.concatenate(
    (
        np.concatenate((_, B, B, B, B, B, B, B), axis=1),
        np.concatenate((_, B, _, _, _, _, _, B), axis=1),
        np.concatenate((_, B, _, B, B, B, _, B), axis=1),
        np.concatenate((_, B, _, _, _, B, _, B), axis=1),
        np.concatenate((_, B, B, B, _, B, _, B), axis=1),
        np.concatenate((_, _, _, _, _, B, _, B), axis=1),
        np.concatenate((B, B, B, B, B, B, _, B), axis=1),
    ),
    axis=0
)
thin_black_line = np.full((d, 2 * d * 8), 0)
thick_black_line = np.concatenate((thin_black_line, thin_black_line), axis=0)
thin_white_line = np.full((d, 2 * d * 8), 255)
thick_white_line = np.concatenate((thin_white_line, thin_white_line), axis=0)


final = np.concatenate(
    (
        thick_white_line,
        thin_black_line,
        thick_white_line,
        meander,
        thick_white_line,
        thin_black_line,
        thin_white_line,
        thin_black_line,
        thin_white_line,
    ),
    axis=0,
)
plt.imshow(np.concatenate((final, final, final, final), axis = 1))
plt.show()
