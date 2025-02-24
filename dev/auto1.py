import cv2
import numpy as np
import pydirectinput
import win32gui, win32ui, win32con, win32api
import time
import ctypes


def press_space():
    """Simulate pressing the space key using pydirectinput."""
    print("Space pressed!")
    pydirectinput.press('space')
    time.sleep(1.0)  # Short delay to avoid repeated triggering


def get_dpi_scaling(hwnd=None):
    """Get the DPI scaling factor for the given window handle (hwnd)."""
    try:
        if hwnd:
            dpi = ctypes.windll.user32.GetDpiForWindow(hwnd)
        else:
            hdc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # 88 is LOGPIXELSX
            ctypes.windll.user32.ReleaseDC(0, hdc)
        return dpi / 96.0  # Convert DPI to scaling factor
    except AttributeError:
        return 1.0  # Fallback for older versions of Windows


def capture_full_window(window_name: str) -> np.ndarray | None:
    """
    Capture a screenshot of the full window, adjusted for DPI scaling.

    Args:
        window_name (str): Name of the window to capture.

    Returns:
        np.ndarray: Captured frame or None if capture fails.
    """
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd == 0:
        print(f"Window '{window_name}' not found.")
        return None

    # Get the window rectangle
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    window_width = right - left
    window_height = bottom - top

    # Get the DPI scaling factor
    scaling_factor = get_dpi_scaling(hwnd)

    # Adjust dimensions for scaling
    adjusted_width = int(window_width * scaling_factor)
    adjusted_height = int(window_height * scaling_factor)

    hwindc = win32gui.GetWindowDC(hwnd)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(srcdc, adjusted_width, adjusted_height)
    memdc.SelectObject(bitmap)
    memdc.BitBlt((0, 0), (adjusted_width, adjusted_height), srcdc, (0, 0), win32con.SRCCOPY)

    frame = np.frombuffer(bitmap.GetBitmapBits(True), dtype=np.uint8)
    frame.shape = (adjusted_height, adjusted_width, 4)  # BGRA format

    win32gui.DeleteObject(bitmap.GetHandle())
    memdc.DeleteDC()
    srcdc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwindc)

    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def detect_red_bar(frame: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[int | None, int | None]:
    """Detect the red bar in the frame within the specified ROI."""
    x, y, w, h = roi
    cropped_frame = frame[y:y + h, x:x + w]
    hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)

    # Red color range in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Find contours of the red bar
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, _, w, _ = cv2.boundingRect(cnt)
        return x, x + w  # Return the start and end of the bar

    return None, None


def detect_arrow(frame: np.ndarray, template: np.ndarray) -> int | None:
    """Detect the arrow using template matching."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= 0.7:  # Match threshold
        return max_loc[0] + template.shape[1] // 2  # Center x of the arrow

    return None


def main():
    window_name = "HoloCure"
    arrow_template_path = "arrow_template.png"
    arrow_template = cv2.imread(arrow_template_path, cv2.IMREAD_GRAYSCALE)

    if arrow_template is None:
        print("Failed to load the arrow template.")
        return

    while True:
        frame = capture_full_window(window_name)
        if frame is None:
            time.sleep(1)
            continue

        # Define ROIs
        green_roi = (300, 400, 400, 200)  # (x, y, w, h)
        red_bar_start, red_bar_end = detect_red_bar(frame, green_roi)
        arrow_pos = detect_arrow(frame, arrow_template)

        # Debug visualization
        debug_frame = frame.copy()
        x, y, w, h = green_roi
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green ROI

        if red_bar_start and red_bar_end:
            cv2.line(debug_frame, (red_bar_start, y), (red_bar_end, y + h), (0, 0, 255), 2)  # Red bar

        if arrow_pos:
            cv2.line(debug_frame, (arrow_pos, y), (arrow_pos, y + h), (255, 0, 0), 2)  # Blue arrow

        cv2.imshow("Debug", debug_frame)

        # Press space if conditions are met
        if (
            red_bar_start is not None
            and red_bar_end is not None
            and arrow_pos is not None
            and x <= red_bar_start <= x + w
            and x <= red_bar_end <= x + w
            and x <= arrow_pos <= x + w
        ):
            press_space()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
