import json
import re
import logging
import numpy as np
import cv2
import mss
import pytesseract
import pygetwindow as gw
import pyttsx3
from time import sleep
from typing import Optional, Tuple, List

# Configure logging to assist in debugging and tracking the flow of the application.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScreenNumberExtractor:
    """
    Extracts numbers from a specified application window on the screen.

    Attributes:
        config_file (str): Path to the JSON configuration file.
        tesseract_cmd (str): Command or path for the Tesseract OCR engine.
        app_name (str): Name of the application window to capture.
        window_coords (Optional[Tuple[int, int, int, int]]): Coordinates and dimensions of the application window.
        tts_engine: Text-to-Speech engine instance.
    """
    def __init__(self, config_file: str = 'config.json') -> None:
        """
        Initializes the ScreenNumberExtractor with a configuration file.

        Args:
            config_file (str): Path to the JSON configuration file.
        """
        self.config_file: str = config_file
        self.tesseract_cmd, self.app_name = self.load_config()

        # Set the path for the Tesseract engine.
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        # Attempt to get the coordinates of the specified application window.
        self.window_coords: Optional[Tuple[int, int, int, int]] = self.get_window_coords()

        # Initialize the Text-to-Speech engine.
        self.tts_engine = pyttsx3.init()

    def load_config(self) -> Tuple[str, str]:
        """
        Loads the configuration from a JSON file.

        Returns:
            Tuple[str, str]: The Tesseract command and application name.
        """
        try:
            with open(self.config_file) as f:
                data = json.load(f)
            return data.get('tesseract_cmd', ''), data.get('app_name', '')
        except FileNotFoundError as e:
            logger.error(f"Config file not found: {e}")
            return '', ''

    def get_window_coords(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Retrieves the coordinates and dimensions of the specified application window.

        Returns:
            Optional[Tuple[int, int, int, int]]: The window coordinates and dimensions, or None if not found.
        """
        try:
            window = gw.getWindowsWithTitle(self.app_name)[0]  # Adjust index if multiple windows have the same title.
            return (window.left, window.top, window.width, window.height)
        except (IndexError, Exception) as e:
            logger.error(f"Error finding window: {e}")
            return None

    def capture_image(self) -> Optional[np.ndarray]:
        """
        Captures an image of the specified application window.

        Returns:
            Optional[np.ndarray]: The captured image, or None if the coordinates are not set.
        """
        if self.window_coords:
            monitor = {"top": self.window_coords[1], "left": self.window_coords[0], 
                       "width": self.window_coords[2], "height": self.window_coords[3]}
            with mss.mss() as sct:
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        logger.error("Window coordinates not set.")
        return None

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocesses the captured image for better OCR recognition.

        Args:
            img (np.ndarray): The captured image.

        Returns:
            np.ndarray: The preprocessed image.
        """
        # Resize, denoise, convert to grayscale, and threshold the image to prepare it for OCR.
        img_resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        denoised_img = cv2.fastNlMeansDenoisingColored(img_resized, None, 10, 10, 7, 21)
        gray = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def extract_numbers(self, img: np.ndarray) -> List[int]:
        """
        Extracts numbers from the preprocessed image using OCR.

        Args:
            img (np.ndarray): The preprocessed image.

        Returns:
            List[int]: A list of extracted numbers.
        """
        # Preprocess the image and use OCR to extract text, then find all numbers in the text.
        preprocessed_img = self.preprocess_image(img)
        text = pytesseract.image_to_string(preprocessed_img)
        numbers = re.findall(r'\d+', text)
        return list(map(int, numbers))

    def speak(self, text: str) -> None:
        """
        Uses the TTS engine to announce the given text.

        Args:
            text (str): The text to be announced.
        """
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def calculate_and_announce_difference(self, prev_numbers: List[int], current_numbers: List[int]) -> None:
        """
        Calculates and announces the difference between two sets of numbers.

        Args:
            prev_numbers (List[int]): The previous set of numbers.
            current_numbers (List[int]): The current set of numbers.
        """
        try:
            if prev_numbers or current_numbers:
                difference = sum(current_numbers) - sum(prev_numbers)
                action_word = "plus" if difference > 0 else "minus"
                self.speak(f"{action_word} {abs(difference)}")
        except Exception as e:
            logger.error(f"Error calculating or speaking difference: {e}")

    def run(self) -> None:
        """
        The main loop of the application. Continuously captures and processes the application window,
        extracts numbers, and announces changes in the numbers extracted over time.
        """
        previous_numbers = []
        while True:
            img = self.capture_image()
            if img is not None:
                extracted_numbers = self.extract_numbers(img)
                # Default to [1] if no numbers are extracted, ensuring there's always a value to work with.
                if not extracted_numbers:
                    extracted_numbers = [1]
                if extracted_numbers != previous_numbers:
                    logger.info("\nChange detected.\n")
                    self.calculate_and_announce_difference(previous_numbers, extracted_numbers)
                previous_numbers = extracted_numbers
                logger.info(f"\nExtracted numbers: {', '.join(map(str, extracted_numbers))}\n")
            else:
                logger.error("Unable to capture image.")
            sleep(5)  # Wait for 5 seconds before the next capture cycle.

if __name__ == "__main__":
    extractor = ScreenNumberExtractor()
    extractor.run()
