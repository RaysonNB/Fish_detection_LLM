from collections import defaultdict
import os
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("video3.pt")

# Open the video file
video_path = r"C:\Users\rayso\Downloads\127713-739309133_medium.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(list)
cropped_count = 0
output_dir = r"C:\Users\rayso\Desktop\python\evertings_for_final\file_saving"
desired_width = 16 * 60
desired_height = 9 * 60

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (desired_width, desired_height))
    h, w, c = frame.shape
    frame = cv2.resize(frame, (int(w * 0.75), int(h * 0.75)))
    try:
        results = model.track(frame, persist=True)
        frame_cap = frame.copy()

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        conff = results[0].boxes.conf

        # Plot the tracks
        for box, track_id, conf in zip(boxes, track_ids, conff):
            x, y, w, h = box
            x1, y1, x2, y2 = int(x - w // 2), int(y - h // 2), int(x + w // 2), int(y + h // 2)

            # Crop the region of the detected object
            cropped_image = frame_cap[y1:y2, x1:x2]
            h1,w1,c1=cropped_image.shape
            if cropped_image.size != 0 and conf>=0.4 and h1<=w1:
                cropped_path = os.path.join(output_dir, f"cropped_fish(id)_{track_id}.jpg")
                cv2.imwrite(cropped_path, cropped_image)
                cropped_count += 1
            #if conf<=0.3: continue
            # Draw bounding box and label
            label_text = f"fish, id:{track_id}, conf:{conf:.2f}"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), (255, 0, 0), -1)
            cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Update track history
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)

            # Draw the tracking lines
            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, points, isClosed=False, color=(230, 230, 230), thickness=3)

        # Display the frame
        cv2.imshow("YOLO Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except Exception as e:
        print(f"Error processing frame: {e}")
        continue

cap.release()
cv2.destroyAllWindows()