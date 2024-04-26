# trombinoclean

A python script to replace a green screen with a background image, and crop the
image with face detection.

Runs on Python 3.11.8, with the package versions in `requirements.txt`. Other
versions may work, but are untested.

## Usage

The main entrypoint is `replacehsv.py`. There are a few parameters to adjust in
the beginning of the file.

| Parameter                   | Values                 | Notes                                                                       |
| --------------------------- | ---------------------- | --------------------------------------------------------------------------- |
| USE_FACE_DETECTION          | True, False            | If false, falls back to using center point of image as crop point           |
| CROP_MARGIN_W_PERCENT       | float between 0 and 1  | 0.2 means we keep a 20% horizontal margin around the crop point             |
| CROP_MARGIN_H_PERCENT       | float between 0 and 1  | 0.2 means we keep a 20% vertical margin around the crop point               |
| RESIZE_WIDTH                | int (in pixels)        | Output image width. Height is deduced from aspect ratio                     |
| FACE_DETECTION_IMAGE_WIDTH  | int (in pixels)        | Width of image used in face detection. Height is deduced from aspect ratio  |
| INPUT_FORMATS               | List of strings        | File extensions to be recovered when reading images                         |
| SOURCE_DIR                  | string (path)          | Path to directory containing image files                                    |
| OUTPUT_DIR                  | string (path)          | Path to directory to output image files                                     |
| BACKGROUND_IMAGE            | string (path)          | Path to image used to replace green screen                                  |
| NUM_PROCESSES               | int, None              | Number of images to process in parallel. None uses number of CPUs           |


## Notes
- Face detection eats ram like a hungry hungry hippo. If the OOM killer pays
  you a visit, try reducing `FACE_DETECTION_IMAGE_WIDTH` or `NUM_PROCESSES` in `replacehsv.py`, or
  settle for fallback cropping.
- The source images and the background MUST have the same dimensions. 
  The script does not handle resizing the background image.
