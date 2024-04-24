#!/usr/bin/python3
import multiprocessing
import cv2
import os
import traceback

CROP_MARGIN_W_PERCENT = 0.4
CROP_MARGIN_H_PERCENT = 0.3
FACE_DETECTION_IMAGE_WIDTH = 100
RESIZE_WIDTH = 270


def unshrekify2(mask, bgr):
    # weighted average that still leaves some green
    bgr[mask != 0, 1] = (
        bgr[mask != 0, 0] * 0.3 + bgr[mask != 0, 1] * 0.4 + bgr[mask != 0, 2] * 0.3
    )


def resize_image(inp, width):
    ratio = width / inp.shape[1]
    dim = (width, int(inp.shape[0] * ratio))
    return cv2.resize(inp, dim, interpolation=cv2.INTER_AREA)


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def detect_face(bgr):
    mult = bgr.shape[1] // FACE_DETECTION_IMAGE_WIDTH
    tmp = resize_image(bgr, FACE_DETECTION_IMAGE_WIDTH)
    gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

    face_rect = [x * mult for x in face_cascade.detectMultiScale(gray, 1.1, 4)[0]]

    x, y, w, h = face_rect
    # bgr[y : y + h, x : x + w] = [0, 0, 255]

    mid_x = x + w // 2
    mid_y = y + h // 2
    return mid_x, mid_y


def crop_center_on(inp, x, y):
    width, height = inp.shape[1], inp.shape[0]
    left = max(0, x - int(width * CROP_MARGIN_W_PERCENT))
    top = max(0, y - int(height * CROP_MARGIN_H_PERCENT))
    right = min(width, x + int(width * CROP_MARGIN_W_PERCENT))
    bottom = min(height, y + int(height * CROP_MARGIN_H_PERCENT))
    return inp[top:bottom, left:right]


def replace_green_screen(inputimg, bg, outputimg):
    bgr = cv2.imread(inputimg)

    # convert to hsv for easier color selection
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Define the range of green screen color
    hue_low = 45
    hue_high = 100

    # Create a mask of the green screen
    mask = cv2.inRange(hsv, (hue_low, 100, 70), (hue_high, 255, 255))

    # Create a mask of the shadow-y green screen areas
    shadow_mask = cv2.inRange(hsv, (hue_low, 229, 35), (hue_high, 255, 150))

    # overexposed green screen
    overexposed_mask = cv2.inRange(hsv, (hue_low, 75, 230), (hue_high, 110, 255))

    # bgr[mask != 0] = [0, 0, 255]
    # bgr[shadow_mask != 0] = [0, 0, 255]
    # bgr[overexposed_mask != 0] = [0, 0, 255]

    mask = mask | shadow_mask | overexposed_mask
    bgr[mask != 0] = bg[mask != 0]

    # Create a mask of ogre skin
    shrek_mask = cv2.inRange(hsv, (15, 10, 10), (85, 255, 255))
    # exclude the green screen from the mask
    shrek_mask = shrek_mask & ~mask

    # bgr[shrek_mask != 0] = [0, 0, 255]

    # shrek 2: when shrek turns human
    unshrekify2(shrek_mask, bgr)

    mid_x, mid_y = detect_face(bgr)
    bgr = crop_center_on(bgr, mid_x, mid_y)
    bgr = resize_image(bgr, RESIZE_WIDTH)

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
bg = cv2.rotate(bg, cv2.ROTATE_90_COUNTERCLOCKWISE)
i = multiprocessing.Value("i", 0)


def thread_job(file):
    global i
    try:
        filepath = os.path.join("../raw/", file)
        outfile = os.path.join("../data/", file)

        if not os.path.exists(filepath):
            replace_green_screen(filepath, bg, outfile)

        with i.get_lock():
            i.value += 1
            count = i.value
        print("Processed", file, f"[{count}/{len(lsphotos)}]")
    except:
        print("Exception in file ", file)
        traceback.print_exc()


def main():
    with multiprocessing.Pool() as pool:
        pool.map(thread_job, lsphotos)

    print("Done")


if __name__ == "__main__":
    # thread_job("DSC09552.JPG")
    main()
