import cv2
import numpy as np
import pandas as pd

# Function to convert RGB to HSV
def rgb_to_hsv(r, g, b):
    color = np.uint8([[[b, g, r]]])  # OpenCV uses BGR format
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    return hsv_color[0][0]

# Convert provided RGB values to HSV
def rgb_to_hsv(r, g, b):
    color = np.uint8([[[r, g, b]]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    return hsv_color[0][0]

# Define HSV ranges for ball colors
ball_colors = {
    "yellow": (rgb_to_hsv(211, 187, 74) - np.array([10, 50, 50]), rgb_to_hsv(211, 187, 74) + np.array([10, 50, 50])),
    "white": (rgb_to_hsv(255, 248, 231) - np.array([10, 10, 40]), rgb_to_hsv(255, 248, 231) + np.array([0, 7, 24])),
    "green": (rgb_to_hsv(70, 97, 90) - np.array([10, 50, 50]), rgb_to_hsv(70, 97, 90) + np.array([10, 50, 50])),
    "orange": (rgb_to_hsv(250, 118, 85) - np.array([10, 50, 50]), rgb_to_hsv(250, 118, 85) + np.array([10, 50, 50]))
}

# Function to determine which quadrant the ball is in
def get_quadrant(x, y, vertical_line, horizontal_line):
    if x < vertical_line:
        if y < horizontal_line:
            return 3
        else:
            return 2
    else:
        if y < horizontal_line:
            return 4
        else:
            return 1

# Function to detect balls based on color
def detect_balls(frame, lower_color, upper_color):
    mask = cv2.inRange(frame, lower_color, upper_color)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Function to draw quadrants on frame
def draw_quadrants(frame, vertical_line, horizontal_line):
    cv2.line(frame, (vertical_line, 0), (vertical_line, frame.shape[0]), (0, 0, 255), 2)
    cv2.line(frame, (0, horizontal_line), (frame.shape[1], horizontal_line), (0, 0, 255), 2)
    return frame

# Set up video capture
video_path = r"C:\Users\hp\Documents\Assigments\AI Assignment video.mp4"  # Provide the path to your video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create a VideoWriter object to save the output video
output_video_path = r"C:\Users\hp\Documents\Assigments\Processed_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Dictionary to keep track of balls' previous quadrants
ball_prev_quadrants = {"yellow": None, "white": None, "green": None, "orange": None}

# Dictionary to keep track of balls' previous presence
ball_prev_presence = {"yellow": False, "white": False, "green": False, "orange": False}

# List to store events
events = []

# Assuming the red lines are vertical and horizontal at specific positions (adjust based on your image)
vertical_line = int(0.65 * width)  # Move the vertical line to 30% of the frame's width
horizontal_line = height // 2  # Horizontal line in the middle of the frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # Draw quadrants on frame
    frame = draw_quadrants(frame, vertical_line, horizontal_line)

    for color, (lower, upper) in ball_colors.items():
        contours = detect_balls(frame_hsv, lower, upper)
        ball_present = False

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                (x, y, w, h) = cv2.boundingRect(contour)
                cx, cy = x + w // 2, y + h // 2

                current_quadrant = get_quadrant(cx, cy, vertical_line, horizontal_line)
                previous_quadrant = ball_prev_quadrants[color]

                if previous_quadrant is None:
                    ball_prev_quadrants[color] = current_quadrant
                elif current_quadrant != previous_quadrant:
                    events.append((timestamp, previous_quadrant, color, "Exit"))
                    events.append((timestamp, current_quadrant, color, "Entry"))
                    cv2.putText(frame, f"{color} {current_quadrant} Entry", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    ball_prev_quadrants[color] = current_quadrant

                ball_present = True

        if ball_present and not ball_prev_presence[color]:
            events.append((timestamp, ball_prev_quadrants[color], color, "Entry"))
            ball_prev_presence[color] = True
        elif not ball_present and ball_prev_presence[color]:
            events.append((timestamp, ball_prev_quadrants[color], color, "Exit"))
            ball_prev_presence[color] = False

    out.write(frame)
    cv2.imshow('Processed Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Save events to text file
events_df = pd.DataFrame(events, columns=["Time", "Quadrant Number", "Ball Colour", "Type"])
events_df.to_csv(r"C:\Users\hp\Documents\Assigments\Event_Record.txt", index=False)

print("Processing complete. Processed video and events file saved.")
