#!/usr/bin/python3
import multiprocessing
import cv2
import os


def get_bg_mask_greenscreen(inp, lower=45, upper=100):
    # Create a mask of the green screen
    return cv2.inRange(inp, (lower, 100, 70), (upper, 255, 255))


def get_shadows_mask(inp, lower=45, upper=100):
    # Create a mask of the shadow-y green screen areas
    return cv2.inRange(inp, (lower, 229, 35), (upper, 255, 150))


def get_shrek_mask(inp, lower=35, upper=75):
    # Create a mask of ogre skin
    return cv2.inRange(inp, (lower, 10, 10), (upper, 255, 120))


def unshrekify(mask, bgr):
    # average the two channels, which effectively "cancels" the green without
    # changing the overall brightness of the pixel
    bgr[mask != 0, 1] = (bgr[mask != 0, 0] + bgr[mask != 0, 2]) / 2


def resize_image(inp, width):
    ratio = width / inp.shape[1]
    dim = (width, int(inp.shape[0] * ratio))
    return cv2.resize(inp, dim, interpolation=cv2.INTER_AREA)


def crop_image_center(inp, percent=20):
    width, height = inp.shape[1], inp.shape[0]
    left = width * percent // 100
    top = height * percent // 100
    right = width * (100 - percent) // 100
    bottom = height * (100 - percent) // 100
    return inp[top:bottom, left:right]


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

    mask = mask | shadow_mask
    bgr[mask != 0] = bg[mask != 0]

    shrek_mask = get_shrek_mask(hsv) & ~mask
    # bgr[shrek_mask != 0] = [0, 0, 255]
    unshrekify(shrek_mask, bgr)

    bgr = crop_image_center(bgr, 12)
    bgr = resize_image(bgr, 1080)

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
