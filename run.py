#!/usr/bin/env python3
"""
AquaDex Flask Application Runner
"""
import os
import sys

def check_models():
    """Check if required model files exist"""
    model_files = [
        "models/best.pt",
        "models/fsrcnn_superres.pth", 
        "models/mobilenetv2_arcface.pth"
    ]
    
    missing_files = []
    for model_file in model_files:
        if not os.path.exists(model_file):
            missing_files.append(model_file)
    
    if missing_files:
        print("⚠️  WARNING: Missing model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nThe application will start but AI features may not work properly.")
        print("Please ensure all model files are in the 'models/' directory.")
        return False
    
    print("✅ All model files found!")
    return True

def main():
    print("🚀 Starting AquaDex Application...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    # Check models
    check_models()
    
    # Check if uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    
    print("\n📁 Project structure:")
    print(f"   - Working directory: {os.getcwd()}")
    print(f"   - Models directory: {os.path.abspath('models')}")
    print(f"   - Static files: {os.path.abspath('static')}")
    print(f"   - Templates: {os.path.abspath('templates')}")
    
    print("\n🌐 Starting Flask server...")
    print("   - URL: http://localhost:5000")
    print("   - Press Ctrl+C to stop")
    print("=" * 50)
    
    # Import and run the Flask app
    from app import app
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    main()