# Screen Text Extractor

Screen Text Extractor is a Python utility designed for extracting text from a specified window on your desktop. It uses image processing to capture the content of a window and employs OCR (Optical Character Recognition) to convert the captured image into text. This tool is particularly useful for monitoring changes in window content, such as chat windows, system logs, or any application interface where text content updates regularly.

## Features

- **Window Selection**: Targets a specific window by its title for capturing screenshots.
- **Text Extraction**: Utilizes Tesseract OCR for extracting text from images.
- **Change Detection**: Monitors the targeted window for changes in text content and logs the differences.
- **Sound Notification**: Plays a sound notification upon detecting changes in the text content.
- **Flexible Configuration**: Allows customization through a configuration file, including the path to the Tesseract executable and the name of the application window to monitor.

## Dependencies

To use Screen Text Extractor, you will need to have the following libraries installed:

- `numpy`
- `opencv-python` (cv2)
- `pytesseract`
- `mss` for screen capture
- `pygetwindow` for targeting specific windows
- `pygame` for playing sound notifications

You will also need to have Tesseract OCR installed on your system. Refer to the [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) for installation instructions.

## Installation

1. Ensure you have Python installed on your system.
2. Install the required Python libraries using pip install requirements.txt:

```bash
pip install requirements.txt
