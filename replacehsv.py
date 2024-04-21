#!/usr/bin/python3
import multiprocessing
import cv2
import os
import numpy as np

def get_bg_mask_greenscreen(inp, lower=45, upper=100):
    # Create a mask of the green screen
    lower_green = np.array([lower, 100, 70])
    upper_green = np.array([upper, 255, 255])

    # Create a binary mask where the green pixels are white and the non-green pixels are black
    mask = cv2.inRange(inp, lower_green, upper_green)

    return mask


def get_shadows_mask(inp, lower=45, upper=100):
    lower_green = np.array([lower, 229, 35])
    upper_green = np.array([upper, 255, 150])

    # Create a binary mask where the green pixels are white and the non-green pixels are black
    mask = cv2.inRange(inp, lower_green, upper_green)

    return mask


def get_shrek_mask(inp, lower=35, upper=75):
    # Create a mask of the green screen
    lower_green = np.array([lower, 10, 10])
    upper_green = np.array([upper, 255, 120])

    # Create a binary mask where the green pixels are white and the non-green pixels are black
    mask = cv2.inRange(inp, lower_green, upper_green)

    return mask


def unshrekify(mask, bgr):
    # average the two channels, which effectively "cancels" the green without
    # changing the overall brightness of the pixel
    avg = (bgr[mask != 0, 0] + bgr[mask != 0, 2]) / 2

    bgr[mask != 0, 1] = avg  # green = avg red and blue


def resize_image(inp, width):
    ratio = width / inp.shape[1]
    dim = (width, int(inp.shape[0] * ratio))
    return cv2.resize(inp, dim, interpolation=cv2.INTER_AREA)


def replace_green_screen(inputimg, bg, outputimg):
    bgr = cv2.imread(inputimg, cv2.IMREAD_UNCHANGED)
    # rotate image, cuz it's actually stored rotated, but an exif tag displays it not-rotated
    bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Create a mask for the green screen areas
    mask = get_bg_mask_greenscreen(hsv, 45, 100)
    shadow_mask = get_shadows_mask(hsv, 45, 100)

    # bgr[mask != 0] = [0, 0, 255]
    # bgr[shadow_mask != 0] = [0, 0, 255]
    
    mask = (mask | shadow_mask)
    bgr[mask != 0] = bg[mask != 0]

    shrek_mask = get_shrek_mask(hsv) & ~mask
    # bgr[shrek_mask != 0] = [0, 0, 255]
    unshrekify(shrek_mask, bgr)

    cv2.imwrite(outputimg, bgr)

    return inputimg


lsphotos = os.listdir("photos")
bg = cv2.imread("bg_big.jpg")
bg = cv2.rotate(bg, cv2.ROTATE_90_COUNTERCLOCKWISE)
i = multiprocessing.Value("i", 0)


def thread_job(file):
    global i
    filepath = os.path.join("photos", file)
    outfile = os.path.join("output", file)
    replace_green_screen(filepath, bg, outfile)

    with i.get_lock():
        i.value += 1
        count = i.value
    print("Processed", file, f"[{count}/{len(lsphotos)}]")


def main():
    with multiprocessing.Pool() as pool:
        pool.map(thread_job, lsphotos)

    print("Done")


if __name__ == "__main__":
    # thread_job("DSC09552.JPG")
    main()
