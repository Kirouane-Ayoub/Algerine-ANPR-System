import os
import re

import cv2
import easyocr

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class OCR:
    def __init__(self) -> None:
        pass

    def easyocr_fun(self, img):
        # Convert image to grayscale for better OCR accuracy
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Initialize EasyOCR reader for English language
        reader = easyocr.Reader(["en"])

        # Perform OCR on the grayscale image
        result = reader.readtext(gray)

        # Extract numbers from the OCR result using regex
        numbers = re.findall(r"\d+", result[0][-2])
        numbers = "".join(numbers)
        return numbers
