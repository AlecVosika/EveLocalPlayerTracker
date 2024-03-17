import json
import re
from time import sleep
import logging
from typing import Optional, Tuple, List

import cv2
import mss
import numpy as np
import pytesseract
import pygetwindow as gw
import pygame
import pyttsx3

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ScreenTextExtractor:
    def __init__(self, config_file: str = 'config.json') -> None:
        self.config_file: str = config_file
        self.data = self.load_config()
        self.tesseract_executable: str = self.data.get('tesseract_cmd', '')
        self.app_name: str = self.data.get('app_name', '')
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_executable
        self.coords: Optional[Tuple[int, int, int, int]] = self.get_window_coordinates()
        pygame.init()
        self.tts_engine = pyttsx3.init()

    def load_config(self) -> dict:
        try:
            with open(self.config_file) as f:
                return json.load(f)
        except FileNotFoundError as e:
            logger.error(f"Config file not found: {e}")
            return {}

    def get_window_coordinates(self) -> Optional[Tuple[int, int, int, int]]:
        try:
            window = gw.getWindowsWithTitle(self.app_name)[0]  # Adjust the index if needed
            return (window.left, window.top, window.width, window.height)
        except IndexError:
            logger.error("Window not found.")
            return None
        except Exception as e:
            logger.error(f"Error finding window: {e}")
            return None

    def capture_screenshot(self) -> Optional[np.ndarray]:
        if self.coords:
            monitor = {"top": self.coords[1], "left": self.coords[0], "width": self.coords[2], "height": self.coords[3]}
            with mss.mss() as sct:
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            logger.error("Coordinates not set.")
            return None

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        # Resize the image to increase text size
        img_resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        
        # Denoise colored image
        denoised_img = cv2.fastNlMeansDenoisingColored(img_resized, None, 10, 10, 7, 21)

        # Convert to grayscale for further processing
        gray = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to get a binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # cv2.imshow("Captured Region", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return thresh

    def extract_text(self, img: np.ndarray) -> List[str]:
        preprocessed_img = self.preprocess_image(img)
        extracted_text = pytesseract.image_to_string(preprocessed_img)
        pattern = r'^[^\w\s]+ '  # Matches leading special chars followed by a space
        formatted_text = re.sub(pattern, '', extracted_text, flags=re.MULTILINE)
        return [line for line in formatted_text.split('\n') if line.strip()]

    def speak_text(self, text: str) -> None:
        """Read aloud the given text."""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
def calculate_and_speak_difference(self, previous_names, extracted_names) -> None:
    """Calculate the numerical difference between the two sets and speak it, including handling None values."""
    try:
        # Function to clean strings and extract only digits
        def clean_and_extract_digits(s):
            return re.sub(r'[^\d]', '', s)  # Remove all non-digit characters

        # Extracting and cleaning numbers from the sets
        previous_numbers = [int(clean_and_extract_digits(name)) for name in previous_names if clean_and_extract_digits(name).isdigit()]
        current_numbers = [int(clean_and_extract_digits(name)) for name in extracted_names if clean_and_extract_digits(name).isdigit()]

        # Handling the case where we go from no numbers to numbers or vice versa
        if not previous_numbers and current_numbers:
            # From None to a number
            current_sum = sum(current_numbers)
            self.speak_text(f"from no value to {current_sum}")
        elif previous_numbers and not current_numbers:
            # From a number to None
            previous_sum = sum(previous_numbers)
            self.speak_text(f"from {previous_sum} to no value")
        elif previous_numbers and current_numbers:
            # Normal case: both previous and current captures contain numbers
            previous_sum = sum(previous_numbers)
            current_sum = sum(current_numbers)
            difference = current_sum - previous_sum
            if difference > 0:
                self.speak_text(f"plus {difference}")
            elif difference < 0:
                self.speak_text(f"minus {abs(difference)}")
            # If there is no change, you might want to say something or not.
            else:
                self.speak_text(f"no change")
    except Exception as e:
        logger.error(f"Error calculating or speaking difference: {e}")

    def run(self) -> None:
        previous_names = set()
        while True:
            img = self.capture_screenshot()
            if img is not None:
                extracted_names = set(self.extract_text(img))
                if previous_names and extracted_names != previous_names:
                    logger.info("\n Ping! Change detected.\n")
                    self.calculate_and_speak_difference(previous_names, extracted_names)
                previous_names = extracted_names
                logger.info(f"\n{', '.join(extracted_names)}\n")
            else:
                logger.error("Unable to capture screenshot.")
            sleep(5)

if __name__ == "__main__":
    screen_text_extractor = ScreenTextExtractor()
    screen_text_extractor.run()
