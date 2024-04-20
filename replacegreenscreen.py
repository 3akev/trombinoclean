#!/usr/bin/python3
import cv2
import os
import numpy as np


def smooth_mask(mask, kernel_size):
    # Define the structuring element (kernel) for the morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform opening (erosion followed by dilation)
    smoothed_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return smoothed_mask


def get_border_mask(mask, thickness=1):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    border_mask = np.zeros_like(mask)
    cv2.drawContours(border_mask, contours, -1, (255), thickness=thickness)

    return border_mask


def get_bg_mask_greenscreen(inp, thres=100, thres2=100):
    # Create a mask of the green screen
    lower_green = (0, thres, 0)
    upper_green = (thres2, 255, thres2)

    mask = cv2.inRange(inp, lower_green, upper_green)
    return mask


def get_shadows_mask(inp):
    return get_bg_mask_greenscreen(inp, 40, 15)


def unshrekify(inp):
    mask = get_bg_mask_greenscreen(inp, 30, 100)

    # average the two channels, which effectively "cancels" the green without
    # changing the overall brightness of the pixel
    avg = (inp[mask != 0, 0] + inp[mask != 0, 2]) / 2

    inp[mask != 0, 1] = avg  # green = avg red and blue
    # inp[mask != 0] = [0, 0, 255]

    return inp


def resize_image(inp, width):
    ratio = width / inp.shape[1]
    dim = (width, int(inp.shape[0] * ratio))
    return cv2.resize(inp, dim, interpolation=cv2.INTER_AREA)


def apply_mask_smoother(inp, bg, bg_mask):
    # Blur the mask to create a smooth transition
    bg_mask = cv2.GaussianBlur(bg_mask, (3, 3), 0)

    # Normalize the mask to keep its values between 0 and 1
    bg_mask = bg_mask / 255.0

    # Expand the dimensions of bg_mask to match the shape of inp
    bg_mask = np.stack([bg_mask] * 3, axis=-1)

    inp = (1.0 - bg_mask) * inp + bg_mask * bg
    return inp


def replace_green_screen(inputimg, bg, outputimg):
    inp = cv2.imread(inputimg, cv2.IMREAD_UNCHANGED)
    inp = resize_image(inp, 1080)

    # rotate image, cuz it's actually stored rotated, but an exif tag displays it not-rotated
    inp = cv2.rotate(inp, cv2.ROTATE_90_COUNTERCLOCKWISE)

    mask1 = smooth_mask(get_bg_mask_greenscreen(inp), 2)
    # apply mask 1, it's pretty safe
    # but it leaves lots of artifacts
    inp[mask1 != 0] = bg[mask1 != 0]

    mask2 = smooth_mask(get_shadows_mask(inp), 1)

    inp[mask2 != 0] = bg[mask2 != 0]

    inp = unshrekify(inp)

    bg_mask = np.bitwise_or(mask1, mask2)

    # border_mask = get_border_mask(bg_mask, 5)

    # apply the background to the image
    # inp[mask2 != 0] = [0, 0, 255]
    # inp[bg_mask != 0] = bg[bg_mask != 0]

    # apply the background again, but this time with smoothing
    # this alone doesn't suffice, cuz it leaves green artifacts
    # so we do two passes
    inp = apply_mask_smoother(inp, bg, get_border_mask(bg_mask))

    # ensure that pixel values are within valid range [0, 255] (just in case)
    inp = np.clip(inp, 0, 255)

    cv2.imwrite(outputimg, inp)


def main():
    lsphotos = os.listdir("photos")
    bg = cv2.imread("bg.jpg")
    bg = cv2.rotate(bg, cv2.ROTATE_90_COUNTERCLOCKWISE)
    for i, file in enumerate(lsphotos):
        print("Processing", file, f"[{i+1}/{len(lsphotos)}]")
        filepath = os.path.join("photos", file)
        outfile = os.path.join("output", file)
        replace_green_screen(filepath, bg, outfile)
    print("Done")


if __name__ == "__main__":
    # bg = cv2.imread("bg.jpg")
    # bg = cv2.rotate(bg, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # replace_green_screen("./photos/DSC09572.JPG", bg, "output.jpg")
    main()
