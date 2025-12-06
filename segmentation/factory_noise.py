import cv2
import numpy as np


def add_motion_blur(image, ksize=15):
    kernel = np.zeros((ksize, ksize))
    kernel[int((ksize - 1) / 2), :] = 1
    kernel = kernel / ksize
    return cv2.filter2D(image, -1, kernel)


def add_gaussian_noise(image, sigma=20):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_random_occlusion(image, box_size=50):
    h, w, _ = image.shape
    x = np.random.randint(0, w - box_size)
    y = np.random.randint(0, h - box_size)
    image[y:y+box_size, x:x+box_size] = 0
    return image


def factory_augment(image):
    img = image.copy()
    img = add_motion_blur(img)
    img = add_gaussian_noise(img)
    img = add_random_occlusion(img)
    return img
