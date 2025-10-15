from flask import Flask, request, jsonify, send_from_directory, render_template, make_response
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- MODEL PATHS ---
MODEL_DIR = "models"
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
FSRCNN_MODEL_PATH = os.path.join(MODEL_DIR, "fsrcnn_superres.pth")
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_DIR, "mobilenetv2_arcface.pth")

DEVICE = torch.device('cpu')

yolo_model = None
fsrcnn_model = None
classifier_model = None

# --- FSRCNN MODEL ---
class FSRCNN(nn.Module):
    def __init__(self, scale_factor=4, num_channels=3, d=56, s=12, m=4):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, d, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(d, s, kernel_size=1)
        self.conv3 = nn.Conv2d(s, d, kernel_size=1)
        self.deconv = nn.ConvTranspose2d(
            d, num_channels,
            kernel_size=9,
            stride=scale_factor,
            padding=9 // 2,
            output_padding=scale_factor - 1
        )
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.deconv(x)
        return x

# --- CLASSIFIER MODEL ---
class MobileNetV2ArcFace(nn.Module):
    def __init__(self, num_classes=4, embedding_dim=512):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.last_channel, embedding_dim)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        emb = self.backbone(x)
        logits = self.classifier(emb)
        return logits

# --- LOAD MODELS ---
def load_models():
    global yolo_model, fsrcnn_model, classifier_model
    
    # Load YOLOv8 model with compatibility fix
    try:
        print("⏳ Loading YOLOv8 model...")
        # First try the base YOLOv8 model which always works
        yolo_model = YOLO('yolov8n.pt')
        yolo_model.to(DEVICE)
        print("✅ YOLOv8 loaded (using base model).")
    except Exception as e:
        print(f"❌ YOLOv8 loading failed: {e}")
        yolo_model = None
    
    # Load FSRCNN model
    try:
        if os.path.exists(FSRCNN_MODEL_PATH):
            print("⏳ Loading FSRCNN model...")
            fsrcnn_model = FSRCNN(scale_factor=4)
            state_dict = torch.load(FSRCNN_MODEL_PATH, map_location=DEVICE, weights_only=True)
            fsrcnn_model.load_state_dict(state_dict)
            fsrcnn_model.eval().to(DEVICE)
            print("✅ FSRCNN loaded.")
        else:
            print(f"❌ FSRCNN model not found at {FSRCNN_MODEL_PATH}")
            fsrcnn_model = None
    except Exception as e:
        print(f"❌ FSRCNN loading failed: {e}")
        fsrcnn_model = None
    
    # Load Classifier model
    try:
        if os.path.exists(CLASSIFIER_MODEL_PATH):
            print("⏳ Loading Classifier model...")
            classifier_model = MobileNetV2ArcFace(num_classes=4, embedding_dim=512)
            state_dict = torch.load(CLASSIFIER_MODEL_PATH, map_location=DEVICE, weights_only=True)
            classifier_model.load_state_dict(state_dict)
            classifier_model.eval().to(DEVICE)
            print("✅ Classifier loaded.")
        else:
            print(f"❌ Classifier model not found at {CLASSIFIER_MODEL_PATH}")
            classifier_model = None
    except Exception as e:
        print(f"❌ Classifier loading failed: {e}")
        classifier_model = None
    
    # Summary
    loaded_models = sum([yolo_model is not None, fsrcnn_model is not None, classifier_model is not None])
    print(f"📊 Model loading summary: {loaded_models}/3 models loaded successfully")

# Load models on startup
print("\n" + "="*50)
print("🚀 Initializing AquaDex Models...")
print("="*50)
load_models()
print("="*50 + "\n")

# --- CONSTANTS ---
CLASS_NAMES = ["Ceratium", "Coscinodiscus", "Dinophysis", "Euglena"]
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

# --- DEMO MODE ---
def demo_analysis():
    """Demo analysis when models are not available"""
    import random
    
    # Create a simple demo image with text
    demo_img = np.ones((400, 600, 3), dtype=np.uint8) * 50
    cv2.putText(demo_img, "DEMO MODE", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 188, 212), 3)
    cv2.putText(demo_img, "Models not loaded", (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(demo_img, "Showing sample detection", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add demo bounding boxes
    cv2.rectangle(demo_img, (100, 300), (200, 380), (0, 255, 0), 2)
    cv2.rectangle(demo_img, (300, 320), (400, 380), (255, 0, 0), 2)
    cv2.putText(demo_img, "Ceratium 0.85", (100, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(demo_img, "Euglena 0.78", (300, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Encode demo image
    _, buffer = cv2.imencode('.jpg', demo_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    demo_image_b64 = base64.b64encode(buffer).decode('utf-8')
    
    # Create demo crop images
    crop1 = np.ones((60, 60, 3), dtype=np.uint8) * 100
    cv2.putText(crop1, "C", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    _, buffer1 = cv2.imencode('.jpg', crop1, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    crop1_b64 = base64.b64encode(buffer1).decode('utf-8')
    
    crop2 = np.ones((60, 60, 3), dtype=np.uint8) * 120
    cv2.putText(crop2, "E", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    _, buffer2 = cv2.imencode('.jpg', crop2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    crop2_b64 = base64.b64encode(buffer2).decode('utf-8')
    
    demo_detections = [
        {
            'x1': 100, 'y1': 300, 'x2': 200, 'y2': 380,
            'cls_id': 0,
            'label': 'Ceratium (Demo)',
            'confidence': 0.85,
            'crop_base64': crop1_b64
        },
        {
            'x1': 300, 'y1': 320, 'x2': 400, 'y2': 380,
            'cls_id': 3,
            'label': 'Euglena (Demo)',
            'confidence': 0.78,
            'crop_base64': crop2_b64
        }
    ]
    
    species_count = {'Ceratium (Demo)': 1, 'Euglena (Demo)': 1}
    
    return jsonify({
        'processed_image': demo_image_b64,
        'detections': demo_detections,
        'species_count': species_count,
        'total_organisms': 2,
        'demo_mode': True,
        'message': 'Demo mode active - AI models not loaded. Install model files to enable real analysis.'
    })

# --- PREPROCESSING ---
def preprocess_crop(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(DEVICE)

def enhance_crop_with_fsrcnn(crop):
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        sr_tensor = fsrcnn_model(tensor)
        sr_img = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        sr_img = np.clip(sr_img * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)

def simplified_analysis(file):
    """Simplified analysis using only classifier (no object detection)"""
    try:
        file.seek(0)
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        h, w = image.shape[:2]
        input_tensor = preprocess_crop(image)
        
        with torch.no_grad():
            logits = classifier_model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, cls_id = torch.max(probs, dim=1)
            confidence = confidence.item()
            cls_id = cls_id.item()
        
        if confidence < 0.7 and fsrcnn_model is not None:
            enhanced_image = enhance_crop_with_fsrcnn(image)
            input_tensor_enhanced = preprocess_crop(enhanced_image)
            with torch.no_grad():
                logits_enhanced = classifier_model(input_tensor_enhanced)
                probs_enhanced = torch.softmax(logits_enhanced, dim=1)
                confidence_enhanced, cls_id_enhanced = torch.max(probs_enhanced, dim=1)
                if confidence_enhanced > confidence:
                    confidence = confidence_enhanced.item()
                    cls_id = cls_id_enhanced.item()
        
        margin = 20
        x1, y1 = margin, margin
        x2, y2 = w - margin, h - margin
        
        result_image = image.copy()
        color = COLORS[cls_id]
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
        label = f"{CLASS_NAMES[cls_id]} {confidence:.2f}"
        cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        _, buffer = cv2.imencode('.jpg', result_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        result_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        _, crop_buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        crop_b64 = base64.b64encode(crop_buffer).decode('utf-8')
        
        detection = {
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'cls_id': cls_id,
            'label': CLASS_NAMES[cls_id],
            'confidence': confidence,
            'crop_base64': crop_b64
        }
        
        species_count = {detection['label']: 1}
        
        return jsonify({
            'processed_image': result_image_b64,
            'detections': [detection],
            'species_count': species_count,
            'total_organisms': 1,
            'simplified_mode': True,
            'message': 'Simplified analysis mode - YOLOv8 not available, analyzing whole image.'
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

# --- ROUTES ---

@app.route('/')
def index():
    """Serve the main home page"""
    return render_template('index.html')

@app.route('/analysis')
def analysis_page():
    """Serve the analysis page"""
    return render_template('analysis.html')

@app.route('/upload-analysis')
def upload_analysis():
    """Serve the upload analysis page"""
    return render_template('upload_analysis.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Check if models are loaded, if not try to reload them
    global yolo_model, fsrcnn_model, classifier_model
    if yolo_model is None or fsrcnn_model is None or classifier_model is None:
        print("⚠️ Models not loaded, attempting to reload...")
        load_models()
    
    # If YOLOv8 fails but other models work, use simplified analysis
    if yolo_model is None and (fsrcnn_model is not None and classifier_model is not None):
        return simplified_analysis(file)
    # If all models fail, use demo mode
    elif yolo_model is None or fsrcnn_model is None or classifier_model is None:
        return demo_analysis()
    
    try:
        image_stream = BytesIO()
        file.save(image_stream)
        image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        results = yolo_model(image, conf=0.1, imgsz=640, device=DEVICE)
        boxes = results[0].boxes
        detections = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            input_tensor = preprocess_crop(crop)
            with torch.no_grad():
                logits = classifier_model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                confidence, cls_id = torch.max(probs, dim=1)
                confidence = confidence.item()
                cls_id = cls_id.item()

            if confidence < 0.7:
                enhanced_crop = enhance_crop_with_fsrcnn(crop)
                input_tensor_enhanced = preprocess_crop(enhanced_crop)
                with torch.no_grad():
                    logits_enhanced = classifier_model(input_tensor_enhanced)
                    probs_enhanced = torch.softmax(logits_enhanced, dim=1)
                    confidence_enhanced, cls_id_enhanced = torch.max(probs_enhanced, dim=1)
                    if confidence_enhanced > confidence:
                        confidence = confidence_enhanced.item()
                        cls_id = cls_id_enhanced.item()

            # ✅ Encode crop to base64 for frontend
            _, buffer = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            crop_base64 = base64.b64encode(buffer).decode('utf-8')

            detections.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'cls_id': cls_id,
                'label': CLASS_NAMES[cls_id],
                'confidence': confidence,
                'crop_base64': crop_base64  # ← Added for frontend
            })

        if not detections:
            # Return the original image with a message instead of an error
            _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            original_image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'processed_image': original_image_b64,
                'detections': [],
                'species_count': {},
                'total_organisms': 0,
                'message': 'No marine organisms detected in this image. Try uploading a microscopic image with visible plankton or marine organisms.',
                'no_detection': True
            })

        result_image = image.copy()
        for obj in detections:
            color = COLORS[obj['cls_id']]
            cv2.rectangle(result_image, (obj['x1'], obj['y1']), (obj['x2'], obj['y2']), color, 2)
            label = f"{obj['label']} {obj['confidence']:.2f}"
            cv2.putText(result_image, label, (obj['x1'], obj['y1'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        _, buffer = cv2.imencode('.jpg', result_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        # Calculate species count
        species_count = {}
        for detection in detections:
            species = detection['label']
            species_count[species] = species_count.get(species, 0) + 1
        
        return jsonify({
            'processed_image': jpg_as_text,
            'detections': detections,
            'species_count': species_count,
            'total_organisms': len(detections)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/logo.png')
def logo():
    logo_path = os.path.join('static', 'images', 'logo.png')
    if os.path.exists(logo_path):
        return send_from_directory('static/images', 'logo.png')
    else:
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="42" height="42" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="10" fill="#00bcd4"/>
            <text x="12" y="16" text-anchor="middle" fill="white" font-size="10">A</text>
        </svg>'''
        response = make_response(svg)
        response.headers['Content-Type'] = 'image/svg+xml'
        return response

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/database')
def database():
    return render_template('database.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/model-status')
def model_status():
    """Check which models are loaded"""
    status = {
        'yolo_loaded': yolo_model is not None,
        'fsrcnn_loaded': fsrcnn_model is not None,
        'classifier_loaded': classifier_model is not None,
        'all_loaded': all([yolo_model is not None, fsrcnn_model is not None, classifier_model is not None])
    }
    return jsonify(status)

if __name__ == '__main__':
    print("🚀 Starting AquaDex server...")
    print(f"Models dir: {os.path.abspath(MODEL_DIR)}")
    app.run(host='0.0.0.0', port=5000, debug=True)