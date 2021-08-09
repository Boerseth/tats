import functools

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# The script assumes that an image is provided of an object with cylindrical symmetry, taken
# from the side.
#        _,----,_
#        )      (
#       (        )
#        \      /
#         `-,,-`
#
# The image needs to have been prepared in the following way:
#   - The area above the top line and below the bottom line are filled in white (255, 255, 255)
#   - The area to the sides of the side lines filled in red (255, 0, 0), including the pixel
#     immediately right next to the top- and bottom-most uncoloured lines
#      \   white  /
#       \_,----,_/
#        )      (
#  red  (        )  red
#        \      /
#         ;-,,-;
#        /      \
#       /  white \
# This will let the script find the line of pixels that constitute the
#   - top line
#   - bottom line
#   - left line
#   - right line
#
# The problem at hand boils down to finding the pixel value of the new image at position (i, j)_new
# in terms of the old image, using the pixel value in position (x, y)_old.
#
# The top- bottom- left- right- lines will end up being mapped onto the edges of the NEW rectangular
# edges.
#        _,----,_        __________
#        )      (       |          |
#       (        )  ->  |          |
#        \      /       |          |
#         `-,,-`        '----------'
#           old              new
#
# So, given (i, j)_new, how to find (x, y)_old?
#
# We can identify all the points on the cylindrical object using a cylindrical coordinate system.
# This associates with each point some angle  theta  , which can be used as proportional to  i .
# The pixel locations of the old image, however, are defined by the viewing angle  phi  as seen
# from the camera, where the object takes up an angle  2alpha . So, we need a map
#
#      i        ->   theta   ->   phi   ->   x
#  (x-coord                                 (x-coord
#  of new image)                            of old image)
#
# The first arrow is the easiest, the last slightly more tricky, but the very trickiest part is the
# geometry of the middle one,  theta->phi . There may be a more elegant way to write it, but I
# worked out the formula
#         ___________________________________________________
#    z = √ 1 + sin(alpha)^2 - 2 sin(alpha) sin(theta + alpha)
#
#    phi = alpha - arccos((1 + z^2 - sin(alpha)^2)/2z)
#
# The left and right lines show the shape of the cylindrical object, and vertical lines in our new
# image should follow paths similar to these. The start- and end- points of these horizontal lines
# lie on the top- and bottom-lines that we have found, and are completely decided by the angle  phi,
# as a fraction of the angle  alpha .
# 
# We then need only to know where  j  maps onto one of the side-lines (I use the left side-line for
# the left half of the picture, and vv.), and then how to move and stretch the side-curves when we
# move them to start and stop at the start- and end-points we found. This is done in the simplest
# possible way I could think of, completely proportionally, and assumes that the start- and end are
# not in the same  x- position. My method could be improved by instead using the angle  theta  to
# decide how things morph.
#
# This, in the end, will result in a vector with non-integer value, and I just round them down.
# Another improvement would be to instead take a weighted average of the four pixel values around
# the final point, but that to me seems like overkill.


#-------------------------------------------------------------------------------
# FIND LINES

def is_white(pixel):
    return all(val == 255 for val in pixel)


def is_red(pixel):
    return pixel[0] == pixel[3] == 255 and pixel[1] == pixel[2] == 0


with Image.open("./pegasus.png") as im:
    a = np.asarray(im)

height, width, _ = a.shape
top_line = []
already_found_start = False
for x in range(width):
    y = 0
    while is_white(a[y, x]):
        y += 1

    if not is_red(a[y, x]):
        already_found_start = True
        top_line.append((x, y))
    else:
        if already_found_start:
            break

bottom_line = []
already_found_start = False
for x in range(width):
    y = height - 1
    while is_white(a[y, x]):
        y -= 1

    if not is_red(a[y, x]):
        already_found_start = True
        bottom_line.append((x, y))
    else:
        if already_found_start:
            break


left_line = []
for y in range(top_line[0][1], bottom_line[0][1] + 1):
    x = 0
    while is_red(a[y, x]):
        x += 1
    left_line.append((x, y))


right_line = []
for y in range(top_line[-1][1], bottom_line[-1][1] + 1):
    x = width - 1
    while is_red(a[y, x]):
        x -= 1
    right_line.append((x, y))


#-------------------------------------------------------------------------------
# Get rectangular image


n_r = len(right_line)
n_l = len(left_line)
n_t = len(top_line)
n_b = len(bottom_line)

l_r = sum(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        for (x1, y1), (x2, y2) in zip(right_line[:-1], right_line[1:]))
l_l = sum(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        for (x1, y1), (x2, y2) in zip(left_line[:-1], left_line[1:]))


@functools.lru_cache(maxsize=None)
def find_left_point(rho):
    length = 0
    finder = 1
    while length < l_l * rho and finder < len(left_line) - 1:
        x1, y1 = left_line[finder - 1]
        x2, y2 = left_line[finder]
        length += np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        finder += 1
    return left_line[finder]


@functools.lru_cache(maxsize=None)
def find_right_point(rho):
    length = 0
    finder = 1
    while length < l_r * rho and finder < len(right_line) - 1:
        x1, y1 = right_line[finder - 1]
        x2, y2 = right_line[finder]
        length += np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        finder += 1
    return right_line[finder]

alpha = np.pi / 18


def get_phi(theta):
    z = np.sqrt(1 + np.sin(alpha) ** 2 - 2 * np.sin(alpha) * np.sin(theta + alpha))
    acos_arg = min((z ** 2 + np.cos(alpha) ** 2) / (2 * z), 1)
    phi = alpha - np.arccos(acos_arg)
    if theta > np.pi / 2 - alpha:
        return 2 * alpha - phi
    return phi

N_i = 1600
N_j = 16 * 27


def get_x_y(i, j):
    theta = (np.pi - 2 * alpha) * i / (N_i)
    phi = get_phi(theta)
    
    scale = phi / (2 * alpha)
    start = top_line[int(n_t * scale)]
    end = bottom_line[int(n_b * scale)]
    if i < N_i / 2:
        start_0 = left_line[0]
        end_0 = left_line[-1]
        p_0 = find_left_point(j / N_j)
    else:
        start_0 = right_line[0]
        end_0 = right_line[-1]
        p_0 = find_right_point(j / N_j)
    return (
        int(start[0] + (p_0[0] - start_0[0]) * (end[0] - start[0]) / (end_0[0] - start_0[0])),
        int(start[1] + (p_0[1] - start_0[1]) * (end[1] - start[1]) / (end_0[1] - start_0[1])),
    )


for divisor in [72, 36, 18, 9]:
    alpha = np.pi / divisor
    a_new = np.empty_like(a, shape=(N_j, N_i, 4))
    for j in range(N_j):
        for i in range(N_i):
            x, y = get_x_y(i, j)
            a_new[j, i] = a[y, x]
    im = Image.fromarray(a_new)
    im.save(f"./final{divisor:02}_{N_j}.png")

