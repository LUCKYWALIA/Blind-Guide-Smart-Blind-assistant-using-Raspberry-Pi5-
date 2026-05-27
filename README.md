# Blind Guide 
### AI-Based Smart Assistive System for Visually Impaired Individuals

Blind Guide is an AI-powered assistive system developed using **Raspberry Pi 5**, **Computer Vision**, and **Artificial Intelligence** to help visually impaired individuals navigate safely and interact with their surroundings more independently.

The system combines:
- Real-time obstacle detection
- AI-based object detection
- OCR text reading
- Voice assistance
- Live camera feed processing

to provide an intelligent and portable accessibility solution.

---

#  Features

## 🔹 Real-Time Obstacle Detection
- Detects nearby obstacles using OpenCV
- Provides intelligent navigation alerts:
  - STOP
  - SLOW DOWN
  - CLEAR PATH

---

## 🔹 AI-Based Object Detection
- Detects surrounding objects in real-time
- Improves environmental awareness for users

---

## 🔹 OCR Text Detection & Reading
- Detects and reads printed text using Tesseract OCR
- Useful for:
  - books
  - sign boards
  - labels
  - documents

---

## 🔹 Voice Assistance
- Real-time voice guidance and alerts
- Audio feedback through:
  - earbuds
  - headphones
  - speakers

---

## 🔹 Embedded Real-Time Processing
- Runs directly on Raspberry Pi 5
- Optimized for real-time image processing

---

## 🔹 Portable Standalone System
- Lightweight and portable assistive solution
- Designed for real-world deployment

---

## 🔹 Remote Access & Monitoring
- PuTTY for SSH-based access
- Remote Desktop for GUI-based monitoring

---

#  Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main Programming Language |
| OpenCV | Computer Vision & Obstacle Detection |
| Tesseract OCR | Text Detection & Reading |
| NumPy | Image Processing |
| Raspberry Pi 5 | Embedded Processing Unit |
| gTTS | Voice Assistance |
| DroidCam / Camera Module | Live Video Input |
| PuTTY | Remote SSH Access |
| Remote Desktop | Remote GUI Access |

---

#  System Architecture

```text
Camera Input
      ↓
Frame Capture
      ↓
Image Preprocessing
      ↓
+---------------------------+
|  Obstacle Detection       |
|  AI Object Detection      |
|  OCR Text Reading         |
+---------------------------+
      ↓
Decision Logic
      ↓
Voice Assistance + Alerts
      ↓
User Guidance
```

---

#  Working Modes

#  Walking Mode
This mode helps users navigate safely by detecting obstacles in real-time.

### Techniques Used
- Edge Detection
- Contour Detection
- Motion Analysis
- Zone-Based Obstacle Detection

### Smart Alerts
- STOP
- SLOW DOWN
- CLEAR PATH

---

#  OCR Reading Mode
This mode captures and reads printed text from surroundings.

### OCR Pipeline
- Grayscale Conversion
- Noise Reduction
- Image Sharpening
- Adaptive Thresholding
- Tesseract OCR Processing
- Voice Output

---




#  Hardware Requirements

- Raspberry Pi 5
- Camera Module / DroidCam
- Earbuds / Speaker
- Power Supply
- Monitor (optional)

---

#  Installation

##  Install Python Dependencies

```bash
pip install opencv-python
pip install pytesseract
pip install numpy
pip install gtts
```

---

##  Install System Packages (Raspberry Pi)

```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install mpg123
```

---

# Running the Project

```bash
python3 BlindGuide.py
```

---

#  Controls

| Key | Function |
|---|---|
| W | Toggle Walking Mode |
| R | OCR Text Reading |
| Q | Quit Program |

---

#  Concepts Implemented

- Artificial Intelligence
- Computer Vision
- OCR (Optical Character Recognition)
- Embedded Systems
- Real-Time Image Processing
- Human-Computer Interaction
- Assistive Technology

---

# Future Scope

- Smart glasses integration
- GPS-based navigation
- Ultrasonic sensor integration
- Mobile application connectivity
- Advanced AI navigation
- Cloud-based monitoring
- Offline AI object detection models

---

#  Social Impact

Blind Guide aims to:
- Improve accessibility
- Enhance independent navigation
- Reduce dependency on traditional blind sticks
- Provide a low-cost assistive solution for visually impaired individuals

---

