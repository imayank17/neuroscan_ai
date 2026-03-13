import cv2
import numpy as np
from scipy.signal import resample
from logger import app_logger
import io

def extract_signal_from_image(image_bytes: bytes) -> list:
    """
    Main orchestrator to convert an EEG image into a digital signal.
    1. Load image
    2. Preprocess (Gray -> Thresh -> Grid Removal)
    3. Digitize (Column Scan)
    4. Resample (To 178 points)
    """
    app_logger.info("Starting image-to-signal extraction logic...")
    
    try:
        # 1. Load Image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image.")
        
        # 2. Preprocess
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binary thresholding (invert so wave is white on black background)
        # Using adaptive thresholding for varied lighting/scans
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to remove thin grid lines and noise
        # We use a small kernel to keep the wave detail
        kernel = np.ones((2, 2), np.uint8)
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 3. Digitize (Column-wise Y-coordinate Scan)
        h, w = clean.shape
        raw_signal = []
        
        for x in range(w):
            column = clean[:, x]
            # Find indices of white pixels (the wave) in this column
            indices = np.where(column > 0)[0]
            
            if len(indices) > 0:
                # Invert Y (images have 0 at top) and take the average pixel position
                y_value = h - np.mean(indices)
                raw_signal.append(float(y_value))
            elif len(raw_signal) > 0:
                # If no pixel found and we have some data, carry over previous value
                raw_signal.append(raw_signal[-1])
            else:
                # If no pixel and no data yet, start with 0
                raw_signal.append(0.0)
        
        app_logger.info(f"Extracted {len(raw_signal)} raw data points from image width {w}")
        
        # 4. Resample to exactly 178 points (as required by model)
        if len(raw_signal) < 10:
            raise ValueError("Extracted signal too short - image may not contain a clear wave.")
            
        resampled_signal = resample(raw_signal, 178).tolist()
        
        # Ensure all values are floats (the model needs floats)
        resampled_signal = [float(val) for val in resampled_signal]
        
        app_logger.info("Successfully resampled signal to 178 points.")
        return resampled_signal
        
    except Exception as e:
        app_logger.error(f"Image extraction failed: {e}")
        # Fallback to empty list or let the caller handle it
        raise e
