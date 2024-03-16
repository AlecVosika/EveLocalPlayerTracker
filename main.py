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

    def play_sound(self, sound_file) -> None:
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # Wait for audio to finish
            pygame.time.Clock().tick(10)
        return None
    
    def run(self) -> None:
        previous_names = set()  # Use a set for easier comparison of differences
        while True:
            img = self.capture_screenshot()
            if img is not None:
                extracted_names = set(self.extract_text(img))  # Convert list of names to a set
                if previous_names and extracted_names != previous_names:
                    new_names = extracted_names - previous_names
                    removed_names = previous_names - extracted_names
                    changes = []
                    if new_names:
                        changes.append(f"New: {', '.join(new_names)}")
                    if removed_names:
                        changes.append(f"Removed: {', '.join(removed_names)}")
                    changes_str = '; '.join(changes)
                    logger.info("\n Ping! Change detected.\n")
                    logger.info(f"Changes: {changes_str}")  # Print what has changed
                    self.play_sound('ok.mp3')
                previous_names = extracted_names
                logger.info(f"\n{', '.join(extracted_names)}\n")
            else:
                logger.error("Unable to capture screenshot.")
            sleep(5)

if __name__ == "__main__":
    screen_text_extractor = ScreenTextExtractor()
    screen_text_extractor.run()
