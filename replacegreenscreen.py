#!/usr/bin/python3
import multiprocessing
import cv2
import os
import numpy as np


def smooth_mask(mask, kernel_size):
    # Define the structuring element (kernel) for the morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform opening (erosion followed by dilation)
    smoothed_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)

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

    # make sure green > red and blue
    mask = np.logical_and(mask, inp[:, :, 1] > inp[:, :, 0])
    mask = np.logical_and(mask, inp[:, :, 1] > inp[:, :, 2])
    return mask.astype("uint8")


def get_shadows_mask(inp):
    # shadows can be blue-ish than red-ish
    lower_green = (0, 50, 0)
    upper_green = (100, 120, 40)

    mask = cv2.inRange(inp, lower_green, upper_green)

    # Find connected components
    # num_labels, labels = cv2.connectedComponents(stronk)
    #
    # # Count pixels in each component
    # counts = np.bincount(labels.flatten())
    #
    # return [
    #     np.where(labels == lbl, 1, 0).astype("uint8") for lbl in counts[counts > 100]
    # ]
    return mask


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
    bg_mask = cv2.GaussianBlur(bg_mask, (1, 1), 0)

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
    dark_green = (1, 91, 27)

    # calc mean green value of the image
    # green = inp[:2, :, 1].mean()
    # sd = inp[:2, :, 1].std()
    # print("Mean green value:", green)

    mask1 = smooth_mask(get_bg_mask_greenscreen(inp), 2)
    # apply mask 1, it's pretty safe
    # but it leaves lots of artifacts
    inp[mask1 != 0] = bg[mask1 != 0]

    # blocks = [x for x in get_shadows_mask(inp) if np.any(x != 0)]
    # shadow_mask = np.zeros_like(mask1)
    # for block in blocks:
    #     avg_color = inp[block != 0].mean(axis=0)
    #     if avg_color[1] > (avg_color[0] + avg_color[2]):  # VERY GREEN
    #         mask2 = smooth_mask(block, 1)
    #         inp[mask2 != 0] = bg[mask2 != 0]
    #         shadow_mask += mask2

    shadow_mask = smooth_mask(get_shadows_mask(inp), 3)

    # inp = unshrekify(inp)

    bg_mask = np.bitwise_or(mask1, shadow_mask)

    # apply the background to the image
    # inp[mask2 != 0] = [0, 0, 255]
    inp[shadow_mask != 0] = [0, 0, 255]
    # inp[bg_mask != 0] = bg[bg_mask != 0]

    # apply the background again, but this time with smoothing
    # this alone doesn't suffice, cuz it leaves green artifacts
    # so we do two passes
    inp = apply_mask_smoother(inp, bg, get_border_mask(bg_mask))

    # ensure that pixel values are within valid range [0, 255] (just in case)
    inp = np.clip(inp, 0, 255)

    cv2.imwrite(outputimg, inp)

    return inputimg


lsphotos = os.listdir("photos")
bg = cv2.imread("bg.jpg")
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
    # thread_job("DSC09551.JPG")
    main()
