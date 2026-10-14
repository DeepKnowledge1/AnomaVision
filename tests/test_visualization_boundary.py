import numpy as np

from anomavision.visualization.boundary import boundary_image


def test_boundary_image_draws_thin_localization_boundary():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1

    result = boundary_image(image, mask, boundary_color=(255, 0, 0))

    # The resized localization must produce visible boundary pixels.
    red_pixels = np.all(result == np.array([255, 0, 0], dtype=np.uint8), axis=-1)
    assert red_pixels.any()
