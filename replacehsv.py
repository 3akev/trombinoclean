#!/usr/bin/python3
import multiprocessing
import cv2
import os

CROP_W_PERCENT = 0
CROP_H_PERCENT = 20
X_CENTER = 0
Y_CENTER = -500

def unshrekify(mask, bgr):
    # average the two channels, which effectively "cancels" the green without
    # changing the overall brightness of the pixel
    bgr[mask != 0, 1] = (bgr[mask != 0, 0] + bgr[mask != 0, 2]) / 2


def resize_image(inp, width):
    ratio = width / inp.shape[1]
    dim = (width, int(inp.shape[0] * ratio))
    return cv2.resize(inp, dim, interpolation=cv2.INTER_AREA)


def crop_image(inp, w_percent=0, h_percent=10, x_center = 0, y_center = 0):
    width, height = inp.shape[1], inp.shape[0]
    left = int(x_center + width * w_percent / 2) // 100
    top = int(y_center + height * h_percent / 2) // 100
    right = int(x_center + width * (100 - w_percent / 2)) // 100
    bottom = int(y_center + height * (100 - h_percent / 2)) // 100
    return inp[top:bottom, left:right]


def replace_green_screen(inputimg, bg, outputimg):
    bgr = cv2.imread(inputimg, cv2.IMREAD_UNCHANGED)
    # rotate image, cuz it's actually stored rotated, but an exif tag displays it not-rotated
    bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if CROP_W_PERCENT != 0 or CROP_H_PERCENT != 0:
        bgr = crop_image(bgr, CROP_W_PERCENT, CROP_H_PERCENT)

    # convert to hsv for easier color selection
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Define the range of green screen color
    hue_low = 45
    hue_high = 100

    # Create a mask of the green screen
    mask = cv2.inRange(hsv, (hue_low, 100, 70), (hue_high, 255, 255))

    # Create a mask of the shadow-y green screen areas
    shadow_mask = cv2.inRange(hsv, (hue_low, 229, 35), (hue_high, 255, 150))

    # bgr[mask != 0] = [0, 0, 255]
    # bgr[shadow_mask != 0] = [0, 0, 255]

    mask = mask | shadow_mask
    bgr[mask != 0] = bg[mask != 0]

    # Create a mask of ogre skin
    shrek_mask = cv2.inRange(hsv, (35, 10, 10), (75, 255, 120))
    # exclude the green screen from the mask
    shrek_mask = shrek_mask & ~mask

    # bgr[shrek_mask != 0] = [0, 0, 255]
    unshrekify(shrek_mask, bgr)

    bgr = resize_image(bgr, 1080)

    os.makedirs(os.path.dirname(outputimg), exist_ok=True)
    cv2.imwrite(outputimg, bgr)

    return inputimg


lsphotos = []
for dirpath, dirnames, filenames in os.walk("../raw/"):
    dirpath = os.path.relpath(dirpath, "../raw/")
    for filename in filenames:
        if filename.endswith(".JPG"):
            lsphotos.append(os.path.join(dirpath, filename))

bg = cv2.imread("bg_big.jpg")
#bg = cv2.rotate(bg, cv2.ROTATE_90_COUNTERCLOCKWISE)
if CROP_W_PERCENT != 0 or CROP_H_PERCENT != 0:
    bg = crop_image(bg, CROP_W_PERCENT, CROP_H_PERCENT, X_CENTER, Y_CENTER)
i = multiprocessing.Value("i", 0)


def thread_job(file):
    global i
    filepath = os.path.join("../raw/", file)
    outfile = os.path.join("../data/", file)
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
