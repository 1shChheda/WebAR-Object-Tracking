# Real-Time AR Tracking & 3D Model Overlay

This project demonstrates real-time 3D cup detection and tracking using `mediaPipe Objectron`, `openCV` and overlays the detected cup with a 3D model using augmented reality (AR). The 3D rendering is done using `trimesh` and `pyrender`.


## Setup Instructions

Follow these steps to set up and run the project:

### 1. Clone the Repository

```bash
git clone https://github.com/1shChheda/WebAR-Object-Tracking.git
cd real-ar-tracking
```

### 2. Create and Activate a Conda Environment

Make sure you have [Conda](https://docs.conda.io/en/latest/miniconda.html) installed. Then create a new environment and activate it:

```bash
conda create -p ar-venv python=3.9 -y
conda activate ./venv
```

### 3. Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Run the Main Script

Ensure your webcam is connected, then run the script to start real-time AR tracking:

```bash
python main.py
```

## Note:

- The `cup_green_obj.obj` file is used as the 3D model for rendering. Make sure it is present in the `/cup_obj` directory.
- To exit the application, press the 'q' key.

## Author

This project is created and maintained by **Vansh Chheda(https://github.com/1shChheda)**.