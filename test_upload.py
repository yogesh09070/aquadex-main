#!/usr/bin/env python3
"""
Simple test script to verify the upload functionality
"""
import requests
import os

def test_upload():
    # Create a simple test image
    import cv2
    import numpy as np
    
    # Create a test image
    test_img = np.ones((300, 400, 3), dtype=np.uint8) * 128
    cv2.putText(test_img, "Test Image", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save test image
    cv2.imwrite('test_image.jpg', test_img)
    
    # Test upload
    url = 'http://localhost:5000/upload'
    
    try:
        with open('test_image.jpg', 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Upload successful!")
            print(f"   - Detections: {len(data.get('detections', []))}")
            if data.get('demo_mode'):
                print("   - Running in demo mode")
            else:
                print("   - AI models working")
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure Flask app is running on http://localhost:5000")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        # Clean up
        if os.path.exists('test_image.jpg'):
            os.remove('test_image.jpg')

if __name__ == "__main__":
    test_upload()