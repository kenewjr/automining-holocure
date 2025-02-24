import cv2
import pandas as pd
import numpy as np

# Initialize variables
data = []
frame_count = 0
success_count = 0
miss_count = 0
prev_position = None  # Track the previous position of the template
arrow_stopped = False  # To check if the arrow has stopped moving

# Load the arrow template
template = cv2.imread("D:/project/automining-holocure/arrow_template.png", cv2.IMREAD_UNCHANGED)
if template is None:
    print("Error: Template image could not be loaded.")
    exit()
template_w, template_h = template.shape[1], template.shape[0]

# Open video file
cap = cv2.VideoCapture("D:/project/automining-holocure/test.mp4")  # Replace with the correct video file path
if not cap.isOpened():
    print("Error: Could not open video stream or file.")
    exit()

# Red bar detection function
def is_red_bar_present(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = (0, 120, 70)
    upper_red = (10, 255, 255)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    return cv2.countNonZero(mask) > 500  # Adjust threshold for red bar detection

# Check if the template has stopped moving
def has_stopped_moving(current_pos, prev_pos, threshold=5):
    if prev_pos is None or current_pos is None:
        return False
    distance = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
    return distance < threshold  # Check if movement is below the threshold

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame")
        break

    frame_count += 1
    result = "Miss"
    details = ""

    # Template matching
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray_frame, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    # Check if template is detected
    if max_val > 0.8:  # Adjust similarity threshold
        current_position = (max_loc[0] + template_w // 2, max_loc[1] + template_h // 2)

        # Check if the arrow has stopped moving
        if has_stopped_moving(current_position, prev_position):
            if not arrow_stopped:  # If arrow just stopped, log it
                arrow_stopped = True

                # Check if the red bar is present
                if is_red_bar_present(frame):
                    success_count += 1
                    result = "Success"
                    details = f"Arrow stopped at {current_position} with red bar in frame {frame_count}"
                else:
                    miss_count += 1
                    result = "Miss"
                    details = f"Arrow stopped at {current_position} without red bar in frame {frame_count}"

                # Log data only when the arrow has stopped
                data.append({"Frame": frame_count, "Result": result, "Details": details})
        else:
            arrow_stopped = False  # Reset if the arrow starts moving again

        prev_position = current_position
    else:
        current_position = None

cap.release()

# Calculate accuracy
if success_count + miss_count > 0:
    accuracy = (success_count / (success_count + miss_count)) * 100
else:
    accuracy = 0

# Save to Excel
df = pd.DataFrame(data)
df.loc[len(df.index)] = {"Frame": "Summary", "Result": f"Accuracy: {accuracy:.2f}%", "Details": f"Success: {success_count}, Miss: {miss_count}"}
df.to_excel("output.xlsx", index=False)

print("Processing complete. Results saved to output.xlsx.")
