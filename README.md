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

-   **Python 3.10+**
-   **PyTorch**: For the custom embedding/re-identification model.
-   **YOLOv8 (Ultralytics)**: For high-performance object detection.
-   **DeepSORT**: For real-time object tracking.
-   **OpenCV**: For video and image processing.
-   **Tkinter**: For the graphical user interface.
-   **UV**: Fast Python package and project manager.
-   **Git LFS**: For managing large model files.

---

## Setup and Installation

This project contains large model files and uses UV for fast, reliable dependency management. Please follow these steps carefully to ensure the repository is set up correctly.

### 1. Pre-requisite: Install Git LFS

Before cloning, you must have Git LFS (Large File Storage) installed on your system.

-   **On macOS:** `brew install git-lfs`
-   **On Windows/Linux:** [Download and install from the official website](https://git-lfs.com).

After installing, set it up by running this command once in your terminal:
```bash
git lfs install
```

### 2. Pre-requisite: Install UV

UV is a fast Python package manager that replaces pip and virtualenv.

**On macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (using pip):**
```bash
pip install uv
```

After installation, restart your terminal or run:
```bash
source ~/.bashrc  # or source ~/.zshrc on macOS
```

### 3. Clone the Repository
```bash
git clone <your-repository-url>
cd CrystalDector
```

Git LFS will automatically download the large model files (`.pt` and `.pth` files) during cloning.

### 4. Install Python 3.10+

This project requires Python 3.10 or higher. Check your Python version:
```bash
python3 --version
```

If you need to install Python 3.10+:

**On macOS (using Homebrew):**
```bash
brew install python@3.11
```

**Using UV (recommended):**
```bash
uv python install 3.11
uv python pin 3.11
```

**On Windows/Linux:**
Download from [python.org](https://www.python.org/downloads/)

### 5. Install Dependencies

UV will automatically create a virtual environment and install all dependencies:
```bash
uv sync
```

This command will:
- Create a `.venv` directory in your project
- Install all required packages from `pyproject.toml`
- Generate a `uv.lock` file for reproducible builds

---

## Required Model Files

Ensure these files are in your project root directory:

1. **`YOLO_best.pt`** - YOLOv8 detection model (managed by Git LFS)
2. **`cosine_epoch30.pth`** - Custom ResNet50 embedding model (managed by Git LFS)

If these files are missing after cloning, pull them manually:
```bash
git lfs pull
```

---

## Usage

### Running the GUI Application

The easiest way to use the tracker is through the graphical interface:
```bash
uv run python tracker_gui.py
```

**GUI Features:**
1. Click **"Select Video File"** to choose your input video
2. Configure options:
   - **Save Annotated Frames**: Save processed frames to `modified_images/` folder
   - **Draw Velocity Vectors**: Visualize crystal movement
3. Click **"Start Analysis"** to begin processing
4. Monitor progress in real-time
5. View statistics after completion
6. Optionally click **"Create Video"** to compile frames into an output video

### Running Command-Line Scripts

**For basic tracking without GUI:**
```bash
uv run python tracker_deepsort.py
```

**Note:** Edit the `working_directory` variable in the script to match your project location.

**To create a video from saved frames:**
```bash
uv run python create_video.py
```

### Configuration

Before running `tracker_deepsort.py`, update these variables in the script:
```python
working_directory = "/path/to/your/CrystalDector"  # Update this path
video_source = r"video_4.mp4"                       # Your input video
draw_bounding_boxes = True                          # Draw boxes around crystals
draw_velocity_vector = True                         # Draw velocity arrows
save_images = True                                  # Save annotated frames
```

---

## Project Structure
```
CrystalDector/
├── tracker_gui.py              # GUI application (recommended)
├── tracker_deepsort.py         # Command-line tracking script
├── train_resnet50_cosine.py    # Embedding model training script
├── create_video.py             # Convert frames to video
├── pyproject.toml              # Project dependencies (UV config)
├── uv.lock                     # Locked dependency versions
├── requirements.txt            # Legacy requirements (for reference)
├── YOLO_best.pt               # YOLOv8 detection model
├── cosine_epoch30.pth         # ResNet50 embedding model
├── modified_images/           # Output folder for annotated frames
└── README.md                  # This file
```

---

## Managing Dependencies

### Adding New Packages
```bash
uv add package-name
```

### Removing Packages
```bash
uv remove package-name
```

### Updating All Dependencies
```bash
uv lock --upgrade
```

### Updating Specific Package
```bash
uv lock --upgrade-package numpy
```

### Viewing Installed Packages
```bash
uv pip list
```

---

## Output

The system generates:

1. **Console Statistics:**
   - Number of unique crystals tracked
   - Average velocity
   - Standard deviation of velocity

2. **Annotated Frames** (if enabled):
   - Saved to `modified_images/` directory
   - Each frame shows bounding boxes and crystal IDs
   - Optional velocity vectors

3. **Output Video** (optional):
   - Compiled from annotated frames
   - Named `{original_filename}_tracked.mp4`

---

## Troubleshooting

### "Module not found" errors

Make sure you're running scripts with `uv run`:
```bash
uv run python tracker_gui.py
```

Or activate the virtual environment:
```bash
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
python tracker_gui.py
```

### Model files not found

Pull large files with Git LFS:
```bash
git lfs pull
```

### Python version incompatibility

Ensure you're using Python 3.10+:
```bash
python3 --version
uv python install 3.11
uv python pin 3.11
uv sync
```

### Slow performance on Mac

The system automatically uses Apple Metal (MPS) GPU acceleration when available. Check the console output to confirm:
```
Using device: Apple Metal (MPS GPU)
```

### Dependency conflicts

Remove the lock file and resync:
```bash
rm uv.lock
uv sync
```

---

## Development

### Training Custom Embedding Model

To train your own embedding model on custom crystal images:
```bash
uv run python train_resnet50_cosine.py --data-root /path/to/crystal/images --epochs 30
```

The script expects a folder of individual crystal images and will train a ResNet50 model with cosine similarity loss.

---

## System Requirements

- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional but recommended (CUDA for NVIDIA, MPS for Apple Silicon)
- **Storage**: ~2GB for models and dependencies

---

## Performance Tips

1. **GPU Acceleration**: The system automatically uses available GPU (CUDA/MPS)
2. **Frame Skipping**: Adjust `FRAME_SKIP` in scripts for faster processing
3. **Video Resolution**: Resize frames for faster processing (see commented code in `tracker_deepsort.py`)
4. **Batch Size**: Modify detection batch size for memory optimization

---

## License

[MIT, same as YOLO usage]

---

## Acknowledgments

- YOLOv8 by Ultralytics
- DeepSORT tracking algorithm
- PyTorch and torchvision
- UV by Astral

---