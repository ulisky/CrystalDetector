# Crystal Detector and Real-Time Tracker

This project is a complete computer vision pipeline for detecting, tracking, and analyzing the behavior of crystals in video streams. It features a user-friendly graphical interface (GUI) built with Tkinter for easy operation.

The core of the system uses a YOLOv8 model for initial object detection, a DeepSort tracker for maintaining object identities across frames, and a custom-trained ResNet50 embedding network to robustly re-identify crystals even after rotation or temporary occlusion.

---

## Features

-   **Real-Time Detection & Tracking**: Identifies and tracks multiple crystals simultaneously in a video file.
-   **Robust Re-Identification**: Utilizes a custom PyTorch embedding model trained with cosine similarity loss to prevent ID switching when crystals flip or overlap.
-   **Dynamic Statistics**: Calculates key metrics like unique crystal count and average velocity.
-   **Interactive GUI**: A user-friendly interface built with Tkinter allows for easy video selection, real-time visualization, and control over the analysis.
-   **Visualization Tools**: Renders bounding boxes and IDs on each frame for immediate visual feedback.
-   **Data Export**: Optionally saves all processed frames and can compile them into a final output video.

---

## Core Technologies

-   **Python 3.x**
-   **PyTorch**: For the custom embedding/re-identification model.
-   **YOLOv8 (Ultracyclics)**: For high-performance object detection.
-   **DeepSORT**: For real-time object tracking.
-   **OpenCV**: For video and image processing.
-   **Tkinter**: For the graphical user interface.
-   **Git LFS**: For managing large model files.

---

## Setup and Installation

This project contains large model files. Please follow these steps carefully to ensure the repository is set up correctly.

### 1. Pre-requisite: Install Git LFS

Before cloning, you must have Git LFS (Large File Storage) installed on your system.
-   **On macOS:** `brew install git-lfs`
-   **On Windows/Linux:** [Download and install from the official website](https://git-lfs.com).

After installing, set it up by running this command once in your terminal:
```bash
git lfs install
-

### 2. Clone the Repository
