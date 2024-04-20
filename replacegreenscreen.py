#!/usr/bin/python3
import cv2
import os
import numpy as np


def replace_green_screen_with_bg(inp, bg):
    # Create a mask of the green screen
    lower_green = (0, 100, 0)
    upper_green = (100, 255, 100)

    mask = cv2.inRange(inp, lower_green, upper_green)

    # replace the green screen with white
    inp[mask != 0] = bg[mask != 0]

    return inp


def stronger_green_replacement(inp, bg):
    # for bottom 20% of the image, use bigger range of green, cuz shadow
    lower_green = (0, 80, 0)
    upper_green = (80, 255, 80)

    strong_mask = cv2.inRange(inp, lower_green, upper_green)

    mask_bottom_20 = np.zeros_like(strong_mask)
    mask_bottom_20[int(strong_mask.shape[0] * 0.8) :, :] = 1

    mask = np.bitwise_and(strong_mask, mask_bottom_20)

    inp[mask != 0] = bg[mask != 0]
    return inp


def remove_green_artifacts(inp):
    # at this point, the green screen is replaced with white
    # but there are still some green pixels left, which need to be removed

    # here, we use a mask that selects anything that even thinks about being green
    lower_green = np.array([0, 30, 0])
    upper_green = np.array([150, 255, 150])

    mask = cv2.inRange(inp, lower_green, upper_green)

    # inp[mask != 0] = [0, 0, 255]
    # average the two channels, which effectively "cancels" the green without
    # changing the overall brightness of the pixel
    avg = (inp[mask != 0, 0] + inp[mask != 0, 2]) / 2
    inp[mask != 0, 1] = avg  # green = avg red and blue

    # ensure that pixel values are within valid range [0, 255]
    inp = np.clip(inp, 0, 255)

    return inp


def resize_image(inp, width):
    ratio = width / inp.shape[1]
    dim = (width, int(inp.shape[0] * ratio))
    return cv2.resize(inp, dim, interpolation=cv2.INTER_AREA)


def replaceGreenScreen(inputimg, backgroundimg, outputimg):
    inp = cv2.imread(inputimg, cv2.IMREAD_UNCHANGED)
    bg = cv2.imread(backgroundimg)

    inp = resize_image(inp, 1080)

    # rotate image, cuz it's actually stored rotated, but an exif tag displays it not-rotated
    inp = cv2.rotate(inp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    bg = cv2.rotate(bg, cv2.ROTATE_90_COUNTERCLOCKWISE)

    white = np.full_like(inp, ((255, 255, 255)))

    inp = replace_green_screen_with_bg(inp, white)
    inp = stronger_green_replacement(inp, white)
    # inp = remove_green_artifacts(inp)

    # replace white with bg
    mask = np.all(inp == [255, 255, 255], axis=-1)
    inp[mask] = bg[mask]

    cv2.imwrite(outputimg, inp)


if __name__ == "__main__":
    lsphotos = os.listdir("photos")
    for i, file in enumerate(lsphotos):
        print("Processing", file, f"[{i+1}/{len(lsphotos)}]")
        filepath = os.path.join("photos", file)
        outfile = os.path.join("output", file)
        replaceGreenScreen(filepath, "./bg.jpg", outfile)
    print("Done")
