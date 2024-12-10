import cv2
import numpy as np
import pydirectinput
import win32gui, win32ui, win32con
import time


def press_space():
    """Simulate pressing the space key using pydirectinput."""
    pydirectinput.press('space')
    time.sleep(1)  # Add a 1-second delay after pressing space



def get_window_screenshot(window_name: str) -> np.ndarray | None:
    """Capture a screenshot of a specific window by name."""
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd == 0:
        print(f"Window '{window_name}' not found.")
        return None

    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (left, top))
    right, bottom = win32gui.ClientToScreen(hwnd, (right, bottom))
    width, height = right - left, bottom - top

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

    screenshot = np.frombuffer(bitmap.GetBitmapBits(True), dtype=np.uint8)
    screenshot.shape = (height, width, 4)

    win32gui.DeleteObject(bitmap.GetHandle())
    memdc.DeleteDC()
    srcdc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwindc)

    return cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)


def detect_red_bar(frame: np.ndarray) -> tuple[int | None, int | None]:
    """Detect the red bar in the frame."""
    roi_x, roi_y, roi_w, roi_h = 400, 600, 450, 45
    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Adjusted thresholds for better red detection
    lower_red_1 = np.array([0, 100, 50])  # Adjusted lower range
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 100, 50])  # Adjusted upper range
    upper_red_2 = np.array([180, 255, 255])

    red_mask = cv2.inRange(hsv_roi, lower_red_1, upper_red_1) + cv2.inRange(hsv_roi, lower_red_2, upper_red_2)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    leftmost_x = float('inf')
    leftmost_rect = None

    # Process contours to find the smallest valid red bar
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Skip tiny bars that may be noise
        if w > 5 and h > 5:  # Minimum size to consider as a red bar
            if x < leftmost_x:
                leftmost_x = x
                leftmost_rect = (x, y, w, h)

    if leftmost_rect:
        x, y, w, h = leftmost_rect
        bar_start = roi_x + x
        bar_end = roi_x + x + w
        return bar_start, bar_end

    # Return None if no valid red bar is found
    return None, None


def detect_arrow(frame: np.ndarray, arrow_template: np.ndarray) -> int | None:
    """Detect the arrow in the frame using template matching."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray_frame, arrow_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    threshold = 0.66

    if max_val > threshold:
        return max_loc[0] + arrow_template.shape[1] // 2
    return None


def predict_arrow_position(prev_x: int | None, current_x: int | None, delta_time: float) -> int | None:
    """Predict the future position of the arrow."""
    if prev_x is None or current_x is None or delta_time <= 0:
        return current_x
    velocity = (current_x - prev_x) / delta_time
    return int(current_x + velocity * 0.05)  # Predict slightly ahead


def main():
    window_name = "HoloCure"
    arrow_template_path = "arrow_template.png"
    arrow_template = cv2.imread(arrow_template_path, cv2.IMREAD_GRAYSCALE)

    if arrow_template is None:
        print("Failed to load template. Check the file path!")
        return

    prev_arrow_x = None
    prev_time = time.time()

    while True:
        frame = get_window_screenshot(window_name)
        if frame is None:
            continue

        current_time = time.time()
        delta_time = current_time - prev_time

        arrow_x = detect_arrow(frame, arrow_template)
        bar_start, bar_end = detect_red_bar(frame)

        predicted_arrow_x = predict_arrow_position(prev_arrow_x, arrow_x, delta_time)

        if predicted_arrow_x is not None and bar_start is not None and bar_end is not None:
            if bar_start <= predicted_arrow_x <= bar_end:
                press_space()

        prev_arrow_x = arrow_x
        prev_time = current_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
