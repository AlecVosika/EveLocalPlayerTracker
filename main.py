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

    def extract_text(self, img: np.ndarray) -> List[str]:
        extracted_text = pytesseract.image_to_string(img)
        pattern = r'^[^\w\s]+ '  # Matches leading special chars followed by a space
        formatted_text = re.sub(pattern, '', extracted_text, flags=re.MULTILINE)
        return [line for line in formatted_text.split('\n') if line.strip()]

    def play_sound(self, sound_file) -> None:
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()
        return None

    def run(self) -> None:
        previous_names = None
        while True:
            img = self.capture_screenshot()
            if img is not None:
                extracted_names = self.extract_text(img)
                if previous_names is not None and extracted_names != previous_names:
                    logger.info("\n Ping! Change detected.\n")
                    self.play_sound('ok.mp3')
                previous_names = extracted_names
                logger.info(f"\n{extracted_names}\n")
            else:
                logger.error("Unable to capture screenshot.")
            sleep(5)


if __name__ == "__main__":
    screen_text_extractor = ScreenTextExtractor()
    screen_text_extractor.run()
