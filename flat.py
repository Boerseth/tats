import sys
from typing import Any, Callable, List
from functools import lru_cache

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def pixel_is_white(pixel):
    return list(pixel) == [255, 255, 255]


def pixel_is_red(pixel):
    return list(pixel) == [255, 0, 0]


def pixel_is_in_pic(pixel):
    return not (pixel_is_white(pixel) or pixel_is_red(pixel))


def get_pixel_state(pixel):
    if pixel_is_white(pixel):
        return "WHITE"
    if pixel_is_red(pixel):
        return "RED"
    return "PIC"


def find_first_nonwhite_pixel_in_column(image, x, y_0, dy, y_lim, y_prev):
    # Two positions next to one another, to move in direction  dy  searching for the white/pic border
    y1 = y_prev or y_0
    y2 = y1 + dy

    pixel_state_1 = get_pixel_state(image[y1, x])
    pixel_state_2 = get_pixel_state(image[y2, x])
    while pixel_state_1 == pixel_state_2:
        if pixel_state_1 == "RED":  # If both are red, we are outside of the interesting columns and should just return
            return y1
        if pixel_state_1 == "PIC":  # If both y1 and y2 are inside the picture, we must move backward to find the border
            if y1 == y_0:
                return y1
            y1, y2 = y1 - dy, y1
        else:  # If both are white, we must continue forward
            if y2 == y_lim - dy:
                return y2
            y1, y2 = y2, y2 + dy
        pixel_state_1 = get_pixel_state(image[y1, x])
        pixel_state_2 = get_pixel_state(image[y2, x])

    if pixel_state_1 == "WHITE":
        return y2  # Could be either PIC or RED
    return y1  # I don't think this ever happens...


def find_first_nonred_pixel_in_row(image, y, x_0, dx, x_lim, x_prev):
    # Two positions next to one another, to move in direction  dx  searching for the red/pic border
    x1 = x_prev or x_0
    x2 = x1 + dx

    pixel_state_1 = get_pixel_state(image[y, x1])
    pixel_state_2 = get_pixel_state(image[y, x2])
    while pixel_state_1 == pixel_state_2:
        if pixel_state_1 == "WHITE":  # If both are white, we are outside of the interesting rows and should just return
            return x1
        if pixel_state_1 == "PIC":  # If both x1 and x2 are inside the picture, we must move backward to find the border
            if x1 == x_0:
                return x1
            x1, x2 = x1 - dx, x1
        else:  # If both are red, we must continue forward
            if x2 == x_lim - dx:
                return x2
            x1, x2 = x2, x2 + dx
        pixel_state_1 = get_pixel_state(image[y, x1])
        pixel_state_2 = get_pixel_state(image[y, x2])

    if pixel_state_1 == "RED":
        return x2  # Could be either PIC or WHITE
    return x1  # I don't think this ever happens...



def find_horizontal_line(image, is_top_line=True):
    height, width, _ = image.shape
    y_0, dy, y_lim = (0, 1, height) if is_top_line else (height - 1, -1, -1)

    line = []
    has_found_start = False
    for x in range(width):
        y_prev = line[-1][1] if line else None
        y = find_first_nonwhite_pixel_in_column(image, x, y_0, dy, y_lim, y_prev)
        if y == -1:
            continue
        if pixel_is_red(image[y, x]):
            if has_found_start:
                break
            continue
        has_found_start = True
        line.append((x, y))

    return line


def find_vertical_line(image, is_left_line=True, start=0):
    height, width, _ = image.shape
    x_0, dx, x_lim = (0, 1, width) if is_left_line else (width - 1, -1, -1)

    line = []
    has_found_start = False
    for y in range(start, height):
        x_prev = line[-1][0] if line else None
        x = find_first_nonred_pixel_in_row(image, y, x_0, dx, x_lim, x_prev)
        if x == -1:
            continue
        if pixel_is_white(image[y, x]):
            if has_found_start:
                break
            continue
        has_found_start = True
        line.append((x, y))

    return line


def get_spline(points: List[np.ndarray]) -> Callable:
    N = len(points) - 2
    @lru_cache
    def spline(x: float) -> np.ndarray:
        n = int(N * x)
        w = (N * x) % 1
        return (1 - w) * points[n] + w * points[n + 1]
    return spline


def weighted_spline(w: float, sp1: Callable, sp2: Callable) -> Callable:
    def spline(x: float) -> Any:
        return (1 - w) * sp1(x) + w * sp2(x)
    return spline


def weighted(w, a, b):
    return (1 - w) * a + w * b


def pixel_weight(a, p):
    x_, y_ = p[0], p[1]
    x, xw = int(x_), x_ % 1
    y, yw = int(y_), y_ % 1
    return weighted(
        xw,
        weighted(yw, a[x, y], a[x, y+1]),
        weighted(yw, a[x+1, y], a[x+1, y+1]),
    )


if len(sys.argv) == 1:
    print("Script requires an image file to be distorted as argument.")
    quit()
filename = sys.argv[1]
with Image.open(f"./{filename}") as im:
    a = np.asarray(im)

if a.shape[2] == 4:
    a = np.delete(a, 3, 2)
assert a.shape[2] == 3

t_list = [np.array(p) for p in find_horizontal_line(a, True)]
b_list = [np.array(p) for p in find_horizontal_line(a, False)]
t = get_spline(t_list)
b = get_spline(b_list)

N = 2001
ds = 1 / N
L_t = sum(np.linalg.norm(t(s + ds) - t(s)) / np.linalg.norm(t(s) - b(s)) for s in np.linspace(0, 1, N-1))
L_b = sum(np.linalg.norm(b(s + ds) - b(s)) / np.linalg.norm(t(s) - b(s)) for s in np.linspace(0, 1, N-1))

def t_scaled(x):
    traversed_length = 0
    s_suggested = 0
    while x * L_t > traversed_length:
        s_suggested += ds
        try:
            traversed_length += (
                np.linalg.norm(t(s_suggested) - t(s_suggested - ds)) / np.linalg.norm(t(s_suggested) - b(s_suggested))
            )
        except:
            print(x)
            print(s_suggested)
            print(traversed_length, flush=True)
            raise
    return t(s_suggested)


def b_scaled(x):
    traversed_length = 0
    s_suggested = 0
    while x * L_t > traversed_length:
        s_suggested += ds
        traversed_length += (
            np.linalg.norm(b(s_suggested) - b(s_suggested - ds)) / np.linalg.norm(t(s_suggested) - b(s_suggested))
        )
    return b(s_suggested)



l_list = [np.array(p) for p in find_vertical_line(a, True, int(t(0)[1]))]
r_list = [np.array(p) for p in find_vertical_line(a, False, int(t(1)[1]))]
l = get_spline([p - l_list[0] for p in l_list])
r = get_spline([p - r_list[0] for p in r_list])
t_len = sum(np.linalg.norm(t1 - t2) for t1, t2 in zip(t_list[:-1], t_list[1:]))
b_len = sum(np.linalg.norm(b1 - b2) for b1, b2 in zip(b_list[:-1], b_list[1:]))
l_len = sum(np.linalg.norm(l1 - l2) for l1, l2 in zip(l_list[:-1], l_list[1:]))
r_len = sum(np.linalg.norm(r1 - r2) for r1, r2 in zip(r_list[:-1], r_list[1:]))
print("t_len", t_len)
print("b_len", b_len)
print("l_len", l_len)
print("r_len", r_len)

assert l(0)[0] == l(0)[1] == 0
assert r(0)[0] == r(0)[1] == 0

N_i = int((t_len + b_len) / 2)
N_j = int((l_len + r_len) / 2)
#N_i = 2000
#N_j = 270

image_new = np.zeros((N_j, N_i))

for i in range(N_i):
    w = i / (N_i - 1)
    middle_line = weighted_spline(w, l, r)

    t_w = t(w)
    b_w = b(w)
    m_1 = middle_line(1)

    y_scale = (b_w[0] - t_w[0]) / m_1[0]
    x_scale = (b_w[1] - t_w[1]) / m_1[1]
    A = np.array([[y_scale, 0], [0, x_scale]])

    for j in range(N_j):
        v = j / (N_j - 1)
        m = middle_line(v)
        p = A.dot(m) + t_w
        y, x = int(p[0]), int(p[1])
        image_new[j, i] = a[x, y, 0]

im = Image.fromarray(image_new)
im = im.convert('RGB')
new_filename = filename.replace(".png", "") + f"_{N_i}_{N_j}.png"
# new_filename = f"{filename.split('.')[0]}_distorted_{N_i}_{N_j}.jpeg"
im.save(f"./{new_filename}")
