# Screen Number Extractor

The Screen Number Extractor is a Python utility designed for extracting numbers from a specified application window on your screen. It captures the window's content, processes the image to identify numbers, and uses text-to-speech to announce changes in the detected numbers. This tool integrates several powerful libraries, including OpenCV for image processing, pytesseract for OCR (Optical Character Recognition), and pyttsx3 for text-to-speech functionality.

## Features

- Automatic detection of specified application window coordinates.
- Real-time capture and processing of the window's content.
- Extraction of numbers from the processed image using OCR.
- Announcement of changes in detected numbers via text-to-speech.

## Dependencies

To run this utility, you need to have the following Python libraries installed:

- numpy
- opencv-python (cv2)
- mss for screen capture
- pytesseract for OCR
- pygetwindow for window management
- pyttsx3 for text-to-speech
- Additionally, Tesseract-OCR software must be installed and its path correctly configured in `config.json`.

## Configuration

Before running the utility, ensure you have a `config.json` file in the same directory with the following structure:

```json
{
  "tesseract_cmd": "<path_to_tesseract_executable>",
  "app_name": "<name_of_application_window>"
}

You will also need to have Tesseract OCR installed on your system. Refer to the [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) for installation instructions.

## Installation

1. Ensure you have Python installed on your system.
2. Install the required Python libraries using pip install requirements.txt:

```bash
pip install requirements.txt
