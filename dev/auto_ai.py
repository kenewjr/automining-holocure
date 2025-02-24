import cv2
import numpy as np
import pydirectinput
import win32gui, win32ui, win32con
import time
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib  # For saving and loading the model and scaler

def press_space():
    """Simulate pressing the space key using pydirectinput."""
    print("Pressing space...")  # Debug print to confirm the function is triggered
    pydirectinput.press('space')

import pyautogui
import numpy as np
import cv2

def get_window_screenshot(window_name):
    """Capture a screenshot of a specific window by name."""
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd == 0:
        print(f"Window '{window_name}' not found.")
        return None

    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (left, top))
    right, bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
    width = right - left
    height = bottom - top

    print(f"Window dimensions: width={width}, height={height}")  # Debugging dimensions
    
    if width <= 0 or height <= 0:
        print(f"Invalid window dimensions: width={width}, height={height}")
        return None

    hwindc = win32gui.GetWindowDC(hwnd)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bitmap)
    memdc.BitBlt((0, 0), (width, height), srcdc, (0, 0), win32con.SRCCOPY)

    signed_ints_array = bitmap.GetBitmapBits(True)
    img = np.frombuffer(signed_ints_array, dtype='uint8')
    img.shape = (height, width, 4)

    win32gui.DeleteObject(bitmap.GetHandle())
    memdc.DeleteDC()
    srcdc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwindc)

    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def detect_red_bar(frame):
    """Detect the red bar in the frame."""
    roi_x, roi_y, roi_w, roi_h = 400, 585, 450, 65
    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    
    # Convert to HSV color space to improve red color detection
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define the red color range in HSV
    lower_red_1 = np.array([0, 120, 70])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 120, 70])
    upper_red_2 = np.array([180, 255, 255])
    
    # Create a mask for detecting red
    mask1 = cv2.inRange(hsv_roi, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv_roi, lower_red_2, upper_red_2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply a blur to the mask to reduce noise
    blurred_mask = cv2.GaussianBlur(red_mask, (5, 5), 0)
    
    # Find contours in the blurred mask
    contours, _ = cv2.findContours(blurred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the leftmost bounding box of the red bar
    leftmost_rect = None
    leftmost_x = float('inf')

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x < leftmost_x:
            leftmost_x = x
            leftmost_rect = (x, y, w, h)

    if leftmost_rect:
        x, y, w, h = leftmost_rect
        bar_start = roi_x + x
        bar_end = roi_x + x + w
        print(f"Red bar detected: start={bar_start}, end={bar_end}")  # Debugging red bar
        return bar_start, bar_end, red_mask, roi
    print("No red bar detected.")
    return None, None, red_mask, roi


def detect_arrow(frame, arrow_template):
    """Detect the arrow in the frame using template matching."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Create a pyramid of the frame to avoid resizing the template image
    frame_pyramid = [gray_frame]
    for scale in [0.8, 0.6, 0.4]:  # Apply down-scaling of the frame
        resized_frame = cv2.pyrDown(frame_pyramid[-1])  # Efficiently downscale the image
        frame_pyramid.append(resized_frame)
    
    best_match = None
    max_val_found = 0
    threshold = 0.7  # Threshold for detecting a good match
    
    # Perform template matching on each pyramid level (multi-scale)
    for scaled_frame in frame_pyramid:
        result = cv2.matchTemplate(scaled_frame, arrow_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val > max_val_found and max_val > threshold:
            best_match = max_loc
            max_val_found = max_val
    
    if best_match:
        # Calculate the center of the detected arrow position
        arrow_x = best_match[0] + arrow_template.shape[1] // 2
        print(f"Detected Arrow at X: {arrow_x}, Max Value: {max_val_found}")  # Debugging arrow detection
        return arrow_x
    else:
        print("No arrow detected.")
        return None


# AI model training function
def train_model(data, labels):
    """Train a simple machine learning model to predict space press."""
    scaler = StandardScaler()  # Initialize the scaler
    data_scaled = scaler.fit_transform(data)  # Normalize the data

    X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Check model performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Save the model and scaler to disk
    joblib.dump(model, 'arrow_predictor_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved to disk.")
    
    return model, scaler

def predict_arrow_position(model, scaler, arrow_x, bar_start, bar_end):
    """Predict if the arrow is within the bar using the trained model."""
    if arrow_x is not None and bar_start is not None and bar_end is not None:
        feature = np.array([[arrow_x, bar_start, bar_end]])  # Features: [arrow_x, bar_start, bar_end]
        feature_scaled = scaler.transform(feature)  # Normalize the feature
        prediction = model.predict(feature_scaled)
        print(f"Model prediction: {prediction[0]}")  # Debug print
        return prediction[0] == 1  # Return True if model predicts to press space
    return False


# Function to load the trained model and scaler
def load_model_and_scaler():
    """Load the trained model and scaler from disk."""
    model = joblib.load('arrow_predictor_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded from disk.")
    return model, scaler

def main():
    window_name = "HoloCure"
    arrow_template_path = "arrow_template.png"
    arrow_template = cv2.imread(arrow_template_path, cv2.IMREAD_GRAYSCALE)

    if arrow_template is None:
        print("Failed to load arrow template. Check the file path!")
        return

    try:
        # Try to load the pre-trained model and scaler
        model, scaler = load_model_and_scaler()
    except:
        # If loading fails, train a new model
        print("No pre-trained model found, training a new model.")
        # Collect training data (arrow_x, bar_start, bar_end, label) where label=1 means space press, 0 means no space
        training_data = []
        training_labels = []
        print("Collecting training data...")

        for _ in range(100):  # Collect 100 training examples
            frame = get_window_screenshot(window_name)
            if frame is None:
                continue
            
            bar_start, bar_end, red_mask, roi = detect_red_bar(frame)
            if bar_start is None:
                continue

            arrow_x = detect_arrow(frame, arrow_template)
            if arrow_x is None:
                continue

            # Simulate label (1 = press space, 0 = don't press space)
            label = 1 if bar_start <= arrow_x <= bar_end else 0
            training_data.append([arrow_x, bar_start, bar_end])
            training_labels.append(label)
            
            # Debugging: Show the ROI with detection
            debug_frame = frame.copy()
            cv2.rectangle(debug_frame, (bar_start, 0), (bar_end, frame.shape[0]), (0, 0, 255), 2)
            cv2.drawMarker(debug_frame, (arrow_x, 300), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=50, thickness=2)
            
            # Highlight the ROI area
            cv2.rectangle(debug_frame,  (400, 585), (850, 650), (255, 0, 0), 2)  # Draw blue ROI rectangle
            
            cv2.imshow("Debug Frame", debug_frame)
            cv2.waitKey(1)  # Allow a brief view of the debug frame

            time.sleep(0.1)  # Simulate some time between data collection

        # Train a model with the collected data
        model, scaler = train_model(np.array(training_data), np.array(training_labels))
    
    while True:
        # Main loop to monitor and react based on detected arrow and red bar
        frame = get_window_screenshot(window_name)
        if frame is None:
            continue
        
        bar_start, bar_end, red_mask, roi = detect_red_bar(frame)
        if bar_start is None:
            continue

        arrow_x = detect_arrow(frame, arrow_template)
        if arrow_x is None:
            continue

        # Debugging: Show the frame with detection
        debug_frame = frame.copy()
        cv2.rectangle(debug_frame, (bar_start, 0), (bar_end, frame.shape[0]), (0, 0, 255), 2)
        cv2.drawMarker(debug_frame, (arrow_x, 300), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=50, thickness=2)
        
        # Highlight the ROI area
        cv2.rectangle(debug_frame,  (400, 585), (850, 650), (255, 0, 0), 2)  # Draw blue ROI rectangle
        
        cv2.imshow("Debug Frame", debug_frame)
        cv2.waitKey(1)  # Allow a brief view of the debug frame

        # Predict if the arrow is within the bar area and decide whether to press space
        if predict_arrow_position(model, scaler, arrow_x, bar_start, bar_end):
            press_space()

        time.sleep(0.1)  # Adjust time interval between checks

if __name__ == "__main__":
    main()
