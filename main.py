import os
import cv2
from Granulify import Granulify
import numpy as np


def load_images():
    all_files = os.listdir("./Images")
    filtered_files = [file for file in all_files if
                      file.lower().endswith('.png') and '_segmented.' not in file.lower()]
    return [(cv2.imread(os.path.join("./Images", file)), file) for file in filtered_files]


if __name__ == '__main__':
    images = load_images()
    granulify = Granulify(images, display=True)
    granulify.run()