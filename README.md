# ASL Gesture Recognition System

A real-time American Sign Language (ASL) gesture recognition system built with MediaPipe and machine learning, capable of recognizing hand gestures and triggering system actions.

## 🔗 MediaPipe Solutions

This project leverages Google's MediaPipe framework, a powerful cross-platform framework for building perception pipelines. MediaPipe provides robust hand tracking and gesture recognition capabilities that enable real-time computer vision applications.

### Useful MediaPipe Resources:
- [MediaPipe Documentation](https://developers.google.com/mediapipe)
- [MediaPipe Hand Solutions](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [MediaPipe Model Maker](https://developers.google.com/mediapipe/solutions/model_maker)
- [MediaPipe GitHub Repository](https://github.com/google/mediapipe)
-[MediaPipe-samples GitHub Repository](https://github.com/google-ai-edge/mediapipe-samples.git)
## 🎯 Project Overview

This system consists of two main components:
1. **Training Pipeline** (`training.py`) - Custom model training using MediaPipe Model Maker
2. **Inference Application** (`gesture_app.py`) - Real-time gesture recognition and system control

## 🚀 Training Phase

### Prerequisites for Training
- Google Colab (Recommended for smooth training without dependency issues)
- Dataset with ASL gesture images
- MediaPipe Model Maker

### Training Process

The training is implemented using **MediaPipe Model Maker**, which allows you to customize existing pre-trained models according to your specific dataset. This approach provides several advantages:

- **Transfer Learning**: Leverages pre-trained models for faster convergence
- **Custom Dataset Support**: Easily train on your own gesture dataset
- **Simplified Pipeline**: Handles preprocessing and model optimization automatically
- **Optimized Performance**: Generates models optimized for real-time inference

### 🔧 Training Configuration

```python
# Hyperparameters optimized for 9000+ image dataset
hparams = gesture_recognizer.HParams(
    learning_rate=0.001,     # Increased learning rate
    epochs=500,              # Extended training epochs
    batch_size=16,           # Optimal batch size
    shuffle=True,
    lr_decay=0.99,          # Learning rate decay
    gamma=0,                # Dataset balance parameter
    export_dir="exported_model"
)

# Model architecture with increased capacity
model_options = gesture_recognizer.ModelOptions(
    dropout_rate=0.1,
    layer_widths=[1024, 512, 256]  # Increased layer widths
)
```

### 📊 Dataset

This project uses a comprehensive ASL gesture dataset from Roboflow:

**Dataset Link**: [ASL Gesture Dataset](https://universe.roboflow.com/ds/gE2TzpqR3J?key=PteZiresb6)

The dataset contains images for all 26 letters of the American Sign Language alphabet (A-Z), providing a solid foundation for gesture recognition training.

#### Dataset Structure:
```
dataset/
├── A/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── B/
│   ├── image1.jpg
│   └── ...
├── C/
├── D/
└── ... (continues for all 26 letters A-Z)
```

### 🌐 Google Colab Training

**We strongly recommend using Google Colab for training** to avoid dependency conflicts and leverage free GPU resources.

**Training Notebook**: [Google Colab Training Link](https://colab.research.google.com/drive/1avTWwCZ7pgBGqrfrMm3wRYwZ5lIrMO6H?usp=sharing)

#### Why Google Colab?
- **Pre-installed Dependencies**: MediaPipe Model Maker comes pre-configured
- **No Environment Issues**: Eliminates local setup complications
- **Easy Data Upload**: Simple file upload interface
- **Automatic Model Export**: Direct download of trained models

### Training Steps:
1. **Data Preparation**: Upload your gesture dataset to Colab
2. **Data Splitting**: Automatically splits data into train/validation/test sets (80/10/10)
3. **Model Training**: Trains custom gesture recognizer using transfer learning
4. **Model Evaluation**: Tests model performance on unseen data
5. **Model Export**: Generates `.task` model file for inference

## 🎮 Inference Phase

### System Requirements
- Python 3.7+
- OpenCV (`cv2`)
- MediaPipe
- PyAutoGUI
- Webcam or IP camera

### Installation
```bash
pip install mediapipe mediapipe-model-maker opencv-python pyautogui
```

### 🎯 Gesture Recognition & System Control

The inference application (`gesture_app.py`) provides real-time gesture recognition with the following features:

#### Supported Gestures & Actions:

**Current Implementation:**
| Gesture | Action | Description | Mnemonic |
|---------|--------|-------------|----------|
| **C** | Open Edge Browser | Launches Microsoft Edge browser | **C**hrome/Browser |
| **L** | Maximize/Restore Window | Toggles current window size | **L**arge/maximize |
| **O** | Open File Explorer | Opens system file manager | **O**pen files |
| **W** | Open YouTube | Opens YouTube in Chrome guest mode | **W**atch videos |
| **V** | Open CMD + Virtual Env | Activates Python virtual environment | **V**irtual environment |

**Available Gestures for Customization:**
The system recognizes all 26 ASL letters (A-Z), but currently only implements 5 actions. You can easily customize and add actions for the remaining 21 letters based on your specific needs.

**Suggested Customizations:**
- **A** → **A**pplications menu
- **B** → **B**rowser (different browser)
- **D** → **D**esktop/Show desktop
- **E** → **E**ditor (text editor)
- **F** → **F**iles (different file manager)
- **G** → **G**ames folder
- **H** → **H**ome directory
- **I** → **I**nformation/System info
- **M** → **M**usic player
- **N** → **N**otepad
- **P** → **P**owerShell
- **S** → **S**creenshot
- **T** → **T**erminal
- And so on...

#### 🔧 Key Features:

**High Confidence Threshold**: 70% confidence required to prevent false triggers
```python
CONFIDENCE_THRESHOLD = 0.70
```

**Gesture Cooldown System**: Prevents repeated actions with 1-second cooldown
```python
if (current_time - last_gesture_timestamp) > 1.0:  # 1 second cooldown
```

**Multi-Platform Support**: Works on Windows, macOS, and Linux
```python
if platform.system() == "Windows":
    # Windows-specific commands
elif platform.system() == "Darwin":  # macOS
    # macOS-specific commands
elif platform.system() == "Linux":
    # Linux-specific commands
```

**Real-time Visual Feedback**: 
- Green text for high-confidence gestures (≥70%)
- Red text for low-confidence gestures (<70%)
- Hand landmark visualization
- Confidence score display

### 🎥 Video Input Options

The system supports multiple video input sources:

1. **Webcam**: Default camera (index 0)
```python
cap = cv2.VideoCapture(0)
```

2. **IP Camera/DroidCam**: Uncomment and configure the URL
```python
# droidcam_url = 'http://10.xx.xx.xx:4747/video'
# cap = cv2.VideoCapture(droidcam_url)
```

### 🚀 Running the Application

1. **Prepare Model**: Ensure `new.task` model file is in the same directory
2. **Run Application**:
```bash
python gesture_app.py
```
3. **Perform Gestures**: Make ASL gestures in front of the camera
4. **System Actions**: Watch as gestures trigger corresponding system actions
5. **Exit**: Press 'q' to quit the application

### 📊 Performance Optimization

- **Asynchronous Processing**: Uses MediaPipe's live stream mode for optimal performance
- **Error Handling**: Comprehensive exception handling for robust operation
- **Resource Management**: Proper cleanup of camera and model resources
- **Cross-platform Compatibility**: Handles different operating system requirements


## 🔧 Customization

### Adding Actions for Existing Gestures:

The system already recognizes all 26 ASL letters but only implements actions for 5 of them. You can easily add actions for any of the remaining letters by modifying the `print_and_draw_results()` function in `gesture_app.py`.

**Example - Adding action for letter 'A':**
```python
elif current_gesture.upper() == "A" or current_gesture.lower() == "a":
    print(f"Action: Open Applications Menu (Confidence: {current_confidence:.2f})")
    try:
        if platform.system() == "Windows":
            pyautogui.hotkey('win')  # Open Start menu
        elif platform.system() == "Darwin":  # macOS
            pyautogui.hotkey('cmd', 'space')  # Open Spotlight
        elif platform.system() == "Linux":
            pyautogui.hotkey('alt', 'f1')  # Open applications menu
    except Exception as e:
        print(f"Error opening applications menu: {e}")
    last_triggered_gesture = current_gesture
    last_gesture_timestamp = current_time
```

**Tips for Gesture-Action Mapping:**
- **Use mnemonics**: Choose gestures that relate to the action (C for Camera, O for Open, etc.)
- **Consider frequency**: Map commonly used actions to easier gestures
- **Platform compatibility**: Ensure actions work across different operating systems
- **User workflow**: Design mappings that match your specific workflow needs

### Adding New Gesture Classes:
1. **Expand Dataset**: Add new gesture images to your training dataset
2. **Retrain Model**: Use the Colab notebook to retrain with expanded dataset
3. **Update Code**: Add new gesture conditions in `print_and_draw_results()` function
4. **Define Actions**: Implement corresponding system actions

### Adjusting Sensitivity:
```python
# Modify confidence threshold
CONFIDENCE_THRESHOLD = 0.80  # Increase for higher accuracy

# Adjust cooldown timing
if (current_time - last_gesture_timestamp) > 2.0:  # Longer cooldown
```

## 🐛 Troubleshooting

**Common Issues:**

1. **Camera Not Found**: Check camera index or IP camera URL
2. **Model Loading Error**: Ensure `new.task` file is in correct location
3. **Low Recognition Accuracy**: Check lighting conditions and gesture positioning
4. **System Actions Not Working**: Verify platform-specific command paths


## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📞 Support

For questions and support, please open an issue in the repository or refer to the MediaPipe documentation for technical details.

---

