from collections import defaultdict
import os
import cv2
import numpy as np

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("hhh.pt")

# Open the video file
video_path = r"C:\Users\rayso\Downloads\Fish Divine In Under Water Video. Ocean Sea Fish Background Video. [No Copyright] #fish #background.mp4"

cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])
cropped_count=0
# Loop through the video frames
output_dir = r"C:\Users\rayso\Desktop\python\evertings_for_final\file_saving"
desired_width = 16 * 60
desired_height = 9 * 60
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame = cv2.resize(frame, (desired_width, desired_height))
    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        frame_cap = frame.copy()
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        conff = results[0].boxes.conf
        # Visualize the results on the frame
        #annotated_frame = results[0].plot()
        #print("result",results[0])
        # Plot the tracks
        for box, track_id, conf in zip(boxes, track_ids, conff):
            #print("id",track_id)
            x, y, w, h = box
            #print(x, y, w, h)
            x1, y1, x2, y2 = x-w//2,y-h//2,x+w//2,y+h//2
            x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
            if conf<0.7: continue
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            label_text = f"fish {conf:.2f}"
            cropped_image = frame_cap[y1:y2, x1:x2]  # Crop the region of the detected object
            if cropped_image.size != 0:  # Ensure the cropped region is valid
                cropped_path = os.path.join(output_dir, f"cropped_{track_id}.jpg")
                cv2.imwrite(cropped_path, cropped_image)
                cropped_count += 1
            text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Create a filled rectangle for the label background
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), (255, 0, 0), -1)
            cv2.putText(frame, "fish_id:%d %.3f" % (track_id, conf), (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
            #print("position",x1,y1,x2,y2)
            # print("Percentage:", conf)
            # print(cls, i.names[cls])
            # Draw the bounding box on the frame

            track = track_history[track_id]

            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)
            #print(track)
            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=3)
        cv2.imshow("YOLO11 Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()