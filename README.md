# AquaDex - Marine Organism Detection System

AquaDex is an AI-powered web application for detecting and classifying marine organisms in microscopic images using Flask, PyTorch, and YOLOv8.

## Features

- 🔬 **Microscopic Image Analysis**: Upload and analyze marine organism images
- 🤖 **AI-Powered Detection**: Uses YOLOv8 for object detection
- 🎯 **Species Classification**: MobileNetV2 with ArcFace for organism classification  
- 🔍 **Image Enhancement**: FSRCNN super-resolution for low-quality images
- 🌐 **Web Interface**: Clean, responsive Flask web application
- 📊 **Real-time Results**: Instant analysis with confidence scores

## Quick Start

### 1. Install Dependencies
```bash
# If 'pip' is not recognized, use:
python -m pip install -r requirements.txt
```

### 2. Ensure Model Files
Place these files in the `models/` directory:
- `best.pt` - YOLOv8 detection model
- `fsrcnn_superres.pth` - Super-resolution model
- `mobilenetv2_arcface.pth` - Classification model

### 3. Run Application
```bash
python run.py
```

### 4. Open Browser
Navigate to: http://localhost:5000

## Project Structure

```
AquaDex/
├── app.py                 # Main Flask application
├── run.py                 # Application runner with checks
├── requirements.txt       # Python dependencies
├── models/               # AI model files
│   ├── best.pt
│   ├── fsrcnn_superres.pth
│   └── mobilenetv2_arcface.pth
├── static/               # Static web assets
│   ├── css/             # Stylesheets
│   ├── js/              # JavaScript files
│   └── images/          # Image assets
├── templates/            # HTML templates
│   ├── index.html       # Home page
│   ├── analysis.html    # Analysis selection
│   ├── upload_analysis.html  # Upload interface
│   └── ...
└── uploads/             # Uploaded images (auto-created)
```

## API Endpoints

- `GET /` - Home page
- `GET /analysis` - Analysis page
- `GET /upload-analysis` - Upload interface
- `POST /upload` - Image upload and analysis
- `GET /database` - Database page (placeholder)
- `GET /history` - History page (placeholder)

## Supported Organisms

Currently detects and classifies:
- Ceratium
- Coscinodiscus  
- Dinophysis
- Euglena

## Technical Details

### Models Used
1. **YOLOv8**: Object detection to locate organisms in images
2. **FSRCNN**: Super-resolution enhancement for low-quality crops
3. **MobileNetV2 + ArcFace**: Classification of detected organisms

### Image Processing Pipeline
1. Upload image via web interface
2. YOLOv8 detects organism bounding boxes
3. Extract crops from detected regions
4. Classify crops using MobileNetV2
5. If confidence < 70%, enhance with FSRCNN and re-classify
6. Return results with bounding boxes and classifications

## Development

### Adding New Organism Classes
1. Update `CLASS_NAMES` in `app.py`
2. Retrain classification model with new classes
3. Update model file and adjust `num_classes` parameter

### Customizing Detection
- Adjust confidence threshold in `/upload` route
- Modify image preprocessing in `preprocess_crop()`
- Update enhancement logic in `enhance_crop_with_fsrcnn()`

## Troubleshooting

### Common Issues
1. **"pip is not recognized"**: Use `python -m pip install -r requirements.txt` instead
2. **Models not loading**: The app will run in demo mode if models fail to load
3. **YOLOv8 compatibility**: Some model versions may have compatibility issues - demo mode will activate
4. **Import errors**: Install all requirements with `python -m pip install -r requirements.txt`
5. **Memory issues**: Reduce image size or batch processing
6. **Port conflicts**: Change port in `app.run()` or `run.py`

### Demo Mode
If AI models fail to load, the application automatically switches to demo mode:
- Shows sample detection results
- Allows testing of the web interface
- Displays "Demo Mode" message in results

### Performance Tips
- Use GPU if available (change `DEVICE = torch.device('cuda')`)
- Optimize image sizes before upload
- Consider model quantization for faster inference

## License

This project is for educational and research purposes.