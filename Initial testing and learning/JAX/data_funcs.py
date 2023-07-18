import numpy as np


def generate_data():
    size = (128, 2)
    N = size[0]
    first_quadrant = np.random.normal(loc=(1, 1), scale=0.1, size=size)
    second_quadrant = np.random.normal(loc=(-1, 1), scale=0.1, size=size)
    third_quadrant = np.random.normal(loc=(-1, -1), scale=0.1, size=size)
    fourth_quadrant = np.random.normal(loc=(1, -1), scale=0.1, size=size)
    data = np.concatenate(
        [first_quadrant, second_quadrant, third_quadrant, fourth_quadrant],
        dtype="float32",
    )
    labels = np.concatenate(
        [np.zeros(N), np.ones(N), np.ones(N) * 2, np.ones(N) * 3], dtype="float32"
    )
    return data, labels


def generate_data_double_cluster():
    size = (128, 2)
    N = size[0]
    first_quadrant = np.random.normal(loc=(1, 1), scale=0.1, size=size)
    second_quadrant = np.random.normal(loc=(-1, 1), scale=0.1, size=size)
    third_quadrant = np.random.normal(loc=(-1, -1), scale=0.1, size=size)
    fourth_quadrant = np.random.normal(loc=(2, 2), scale=0.1, size=size)
    data = np.concatenate(
        [first_quadrant, second_quadrant, third_quadrant, fourth_quadrant],
        dtype="float32",
    )
    labels = np.concatenate(
        [np.zeros(N), np.ones(N), np.ones(N) * 2, np.ones(N) * 3], dtype="float32"
    )
    return data, labels


def generate_random_data(n):
    size = (128, 2)
    N = size[0]
    quadrants = []
    for i in range(n):
        draw_theta = np.random.uniform(low=0, high=2 * np.pi)
        x, y = np.cos(draw_theta), np.sin(draw_theta)
        quadrants.append(np.random.normal(loc=(x, y), scale=0.1, size=size))
    data = np.concatenate(quadrants, dtype="float32")
    labels = np.concatenate([np.ones(N) * i for i in range(n)], dtype="float32")
    return data, labels
