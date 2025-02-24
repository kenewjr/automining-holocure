import cv2
import numpy as np
import pydirectinput
import win32gui, win32ui, win32con
import time
import ctypes

def press_space():
    """Simulate pressing the space key using pydirectinput."""
    print("Space pressed!")
    pydirectinput.press('space')
    time.sleep(0.2)  # Avoid repeated presses too quickly

def get_dpi_scaling(hwnd=None):
    """Get the DPI scaling factor for the given window handle (hwnd)."""
    try:
        if hwnd:
            dpi = ctypes.windll.user32.GetDpiForWindow(hwnd)
        else:
            hdc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
            ctypes.windll.user32.ReleaseDC(0, hdc)
        return dpi / 96.0  # Convert DPI to scaling factor (96 DPI = 100%)
    except AttributeError:
        return 1.0  # Fallback for older Windows versions

def capture_full_window(window_name: str) -> tuple[np.ndarray | None, tuple[int, int]]:
    """
    Capture a screenshot of the specified window by name.
    Returns (frame, (width, height)) or (None, (0,0)) if not found/fails.
    """
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd == 0:
        print(f"Window '{window_name}' not found.")
        return None, (0, 0)

    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width, height = right - left, bottom - top

    # Capture the window
    hwindc = win32gui.GetWindowDC(hwnd)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bitmap)
    memdc.BitBlt((0, 0), (width, height), srcdc, (0, 0), win32con.SRCCOPY)

    # Convert raw data to numpy array
    frame_data = bitmap.GetBitmapBits(True)
    frame = np.frombuffer(frame_data, dtype=np.uint8)
    frame.shape = (height, width, 4)  # BGRA format

    # Cleanup
    win32gui.DeleteObject(bitmap.GetHandle())
    memdc.DeleteDC()
    srcdc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwindc)

    # Convert BGRA to BGR
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return bgr_frame, (width, height)

def detect_red_bar(frame: np.ndarray) -> tuple[int | None, int | None]:
    """
    Dummy logic for detecting a 'red bar' in the frame.
    In a real scenario, you'd do HSV thresholding, etc.
    Returns (start_x, end_x) or (None, None).
    """
    # For demonstration, let's say we didn't detect anything
    return None, None

def detect_arrow(frame: np.ndarray) -> int | None:
    """
    Dummy logic for detecting an arrow in the frame.
    Returns the x-position or None if not found.
    """
    # For demonstration, let's say we didn't detect anything
    return None

def main():
    window_name = "HoloCure"

    while True:
        frame, (width, height) = capture_full_window(window_name)
        if frame is None:
            # If we fail to capture, wait and retry
            time.sleep(1)
            continue

        # 1. Do your detection logic
        bar_start, bar_end = detect_red_bar(frame)
        arrow_x = detect_arrow(frame)

        # 2. Create a debug copy
        debug_frame = frame.copy()

        # 3. Draw lines if bar or arrow is found (dummy example)
        if bar_start is not None and bar_end is not None:
            cv2.line(debug_frame, (bar_start, 100), (bar_start, 200), (0, 0, 255), 2)
            cv2.line(debug_frame, (bar_end, 100), (bar_end, 200), (0, 0, 255), 2)
        if arrow_x is not None:
            cv2.line(debug_frame, (arrow_x, 100), (arrow_x, 200), (255, 0, 0), 2)

        # 4. Show debug info
        cv2.putText(debug_frame, f"RedBar: {bar_start}-{bar_end}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(debug_frame, f"ArrowX: {arrow_x}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 5. Display the debug frame
        cv2.imshow("Debug Frame", debug_frame)

        # 6. If arrow is within the red bar, press space (dummy check)
        if bar_start is not None and bar_end is not None and arrow_x is not None:
            if bar_start <= arrow_x <= bar_end:
                press_space()

        # 7. Add a small sleep to prevent 100% CPU usage (avoid freezing)
        time.sleep(0.05)

        # 8. Check for 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
