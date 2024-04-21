import cv2
import numpy as np


def resize_image(inp, width):
    ratio = width / inp.shape[1]
    dim = (width, int(inp.shape[0] * ratio))
    return cv2.resize(inp, dim, interpolation=cv2.INTER_AREA)


def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image[y, x]

        # Convert the BGR pixel to HSV
        hsv_pixel = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_BGR2HSV)

        # Print the BGR and HSV values
        print("BGR Color: ", pixel)
        print("HSV Color: ", hsv_pixel[0][0])


# Load an image
image = cv2.imread("./output/DSC09551.JPG")

# Resize the image to 1080p
image = resize_image(image, 1080)

# Create a window
cv2.namedWindow("image", cv2.WINDOW_NORMAL)

cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Move the window to the top left corner
cv2.moveWindow("image", 0, 0)

# Set the callback function for mouse events
cv2.setMouseCallback("image", pick_color)

# Display the image
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
