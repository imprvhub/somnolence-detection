# Somnolence Detection System

A real-time computer vision solution for driver drowsiness detection using OpenCV and MediaPipe face mesh detection.

> 🚧 Initial Release: Core drowsiness detection system implemented with plans for enhancement.

## Overview

This project implements a drowsiness detection system that monitors a user's eyes in real-time to detect signs of fatigue and alertness. Using advanced computer vision techniques, it calculates the Eye Aspect Ratio (EAR) to determine if a person's eyes are closing for extended periods, indicating potential drowsiness.

## Key Features

- Real-time eye tracking using MediaPipe Face Mesh
- Eye Aspect Ratio (EAR) calculation
- Visual drowsiness alerts
- Live EAR value display
- Face mesh visualization
- Mirror display for user comfort

## Technical Components

**Eye Detection**
- Precise 6-point eye landmark detection
- Individual left and right eye tracking
- Real-time EAR calculation

**Alert System**
- Dynamic threshold-based detection
- Visual alert system with on-screen warnings
- Configurable sensitivity settings

## Requirements

- Python 3.8+
- Webcam
- Dependencies:
  - OpenCV (cv2)
  - MediaPipe
  - NumPy
  - SciPy

## Quick Start

```bash
# Clone the repository
git clone https://github.com/imprvhub/somnolence-detection.git
cd somnolence-detection

# Install dependencies
pip install -r requirements.txt

# Run the application
python somnolence_detection.py
```

## Usage

The application will launch with webcam activation. Use the following controls:
- `q` - Quit the application
- Visual indicators will show:
  - Green eye contours for tracking visualization
  - EAR value display
  - Red warning text for drowsiness alerts

## Configuration

Key parameters can be adjusted in the code:
- `EAR_THRESH`: Eye Aspect Ratio threshold (default: 0.25)
- `CLOSED_EYES_FRAME`: Consecutive frames for alert (default: 20)

## Roadmap

- [ ] Configurable settings interface
- [ ] Audio alerts
- [ ] Data logging and analytics
- [ ] Multiple face tracking
- [ ] Mobile device support
- [ ] Performance optimization for low-power devices

### Key Notes
This project showcases computer vision and gesture recognition techniques. The gestures were chosen for their detection reliability and technical suitability, without intent to define or standardize their meanings, acknowledging cultural variations.

#### Intended Use
- Research and academic purposes
- Technical demonstrations
- Computer vision development

### Testing
The project includes automated tests using pytest. Tests cover core functionality, EAR calculations, and system robustness.

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest -v
```

For detailed test coverage: `pytest --cov=somnolence_detection`

#### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file in the root directory of this repository for detailed terms and conditions.

---
*Built with OpenCV and MediaPipe*
