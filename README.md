# Algerine ANPR System

## Overview

The **Algerine Real-Time Automatic Number Plate Recognition (ANPR) System** is an advanced solution tailored for the unique requirements of Algeria. It combines state-of-the-art computer vision and deep learning technologies to detect, track, and interpret vehicle license plates with exceptional speed and accuracy. This system is built for applications in traffic management, law enforcement, and parking management, providing robust performance in diverse environments.

- **Real-Time Detection and Tracking**: Utilizes the YOLOv8 object detection algorithm and the SORT tracking algorithm to accurately detect and track number plates from live video feeds or recorded footage.
- **Comprehensive Data Handling**: Capable of saving results in CSV logs, capturing and storing images of number plates and vehicles, and saving detection videos for later review.
- **Versatile Applications**: Ideal for traffic monitoring, law enforcement operations, and parking management.

## Number Plate Dataset

The dataset includes a diverse collection of images and metadata related to Algerian vehicle license plates. It encompasses various types of plates such as standard civilian, government, and specialized plates, ensuring a comprehensive foundation for training and evaluating the ANPR system.

## YOLOv8

**YOLOv8** (You Only Look Once version 8) is an advanced object detection and image segmentation model developed by Ultralytics. Known for its speed and accuracy, YOLOv8 represents the latest evolution in the YOLO model family, optimized for real-time object detection with exceptional efficiency.

## EasyOCR

**EasyOCR** is a Python library specializing in Optical Character Recognition (OCR). It simplifies the extraction of text from images and scanned documents, supporting over 80 languages and various writing scripts, including Latin, Chinese, Arabic, Devanagari, and Cyrillic. EasyOCR can handle both natural scene text and dense document text, making it a versatile tool for text recognition tasks.

## Tracking with SORT Algorithm

The system employs the **Simple Online and Realtime Tracking (SORT)** algorithm to maintain accurate tracking of detected number plates. This enhances the system's ability to follow vehicles across multiple frames, ensuring continuous and reliable tracking in real-time.

## User Interface

The user interface is developed with **Streamlit**, offering an intuitive and interactive experience. Users can easily operate the system, monitor real-time detection and tracking, and manage saved data through a user-friendly dashboard.

## Data Saving Features

The application provides comprehensive data saving capabilities:
- **CSV Logs**: Records detailed logs of detected number plates.
- **Image Storage**: Captures and saves images of detected number plates and vehicles.
- **Video Storage**: Saves detection videos for later analysis and review.

## Usage

To get started with the Algerine ANPR System, follow these steps:

1. **Install Dependencies**:
   Ensure you have all the required dependencies by running:
   ```sh
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   Launch the app using Streamlit by executing the following command:
   ```sh
   streamlit run src/app.py
   ```