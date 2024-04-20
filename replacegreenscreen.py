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


def get_bg_mask_greenscreen(inp, thres=100, thres2=100):
    # Create a mask of the green screen
    lower_green = (0, thres, 0)
    upper_green = (thres2, 255, thres2)

    mask = cv2.inRange(inp, lower_green, upper_green)
    return mask


def get_bg_mask_stronger(inp):
    # for bottom 20% of the image, use bigger range of green, cuz shadow
    strong_mask = get_bg_mask_greenscreen(inp, 80)

    mask_bottom_20 = np.zeros_like(strong_mask)
    mask_bottom_20[int(strong_mask.shape[0] * 0.8) :, :] = 1

    mask = np.bitwise_and(strong_mask, mask_bottom_20)
    return mask


def remove_green_artifacts(inp, bg_mask):
    # here, we use a mask that selects anything that even thinks about being green
    mask = get_bg_mask_greenscreen(inp, 30, 100)

    # filter pixels that are too close to black
    mask = np.bitwise_and(mask, np.all(inp > 25, axis=-1))

    # mask = np.bitwise_and(mask, get_border_mask(bg_mask, 10))

    # average the two channels, which effectively "cancels" the green without
    # changing the overall brightness of the pixel
    avg = (inp[mask != 0, 0] + inp[mask != 0, 2]) / 2

    # Create a mask for when red and blue are too different
    # diff_mask = abs(inp[mask != 0, 0] - inp[mask != 0, 2]) > 50

    # Only update 'avg' where red and blue are not too different
    # avg[~diff_mask] = inp[mask != 0, 1][~diff_mask]

    inp[mask != 0, 1] = avg  # green = avg red and blue
    # inp[mask != 0] = np.full_like(inp[mask != 0], (0, 0, 255))

    return inp


def resize_image(inp, width):
    ratio = width / inp.shape[1]
    dim = (width, int(inp.shape[0] * ratio))
    return cv2.resize(inp, dim, interpolation=cv2.INTER_AREA)


def get_border_mask(mask, border_size):
    # Define the structuring element (kernel) for dilation
    # Here we use a square kernel. You can also use a circular kernel or any other shape.
    kernel = np.ones((border_size, border_size), np.uint8)

    # Dilate the mask to create a border around it
    border_mask = cv2.dilate(mask, kernel, iterations=1)

    # Subtract the original mask from the dilated mask
    # This leaves only the border
    border_mask = border_mask - mask

    return border_mask


def replace_green_screen(inputimg, bg, outputimg):
    inp = cv2.imread(inputimg, cv2.IMREAD_UNCHANGED)
    inp = resize_image(inp, 1080)

    # rotate image, cuz it's actually stored rotated, but an exif tag displays it not-rotated
    inp = cv2.rotate(inp, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # smooth it a bit so stray pixels are removed
    mask1 = smooth_mask(get_bg_mask_greenscreen(inp), 2)

    # stronger mask for the bottom 20% of the image, also smooth it
    mask2 = smooth_mask(get_bg_mask_stronger(inp), 2)

    bg_mask = np.bitwise_or(mask1, mask2)
    inp = remove_green_artifacts(inp, bg_mask)

    border_mask = get_border_mask(bg_mask, 10)

    # replace white with bg
    # mask = np.all(inp == [255, 255, 255], axis=-1)
    # inp[bg_mask != 0] = [0, 0, 255]
    inp[bg_mask != 0] = bg[bg_mask != 0]

    # ensure that pixel values are within valid range [0, 255]
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
