import cv2
from ffpyplayer.player import MediaPlayer
from ultralytics import YOLO
import os

# Input video file
file = r"C:\Users\rayso\Downloads\Fish Divine In Under Water Video. Ocean Sea Fish Background Video. [No Copyright] #fish #background.mp4"

# Output directory for saving images
output_dir = r"C:\Users\rayso\Desktop\python\evertings_for_final\file_saving"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open video and audio
video = cv2.VideoCapture(file)
player = MediaPlayer(file)

# Desired width and height for resizing
desired_width = 16 * 60
desired_height = 9 * 60

# Load YOLO model
model = YOLO("Easynigga.pt")

frame_count = 0  # To keep track of frame numbers
cropped_count = 0  # To keep track of cropped images

#getting fish images
while True:
    ret, frame = video.read()
    audio_frame, val = player.get_frame()

    if not ret:
        print("End of video")
        break

    # Resize the frame
    frame = cv2.resize(frame, (desired_width, desired_height))

    # Run YOLO model on the frame
    results = model(frame)
    for i in results:
        for (x1, y1, x2, y2), cls, conf in zip(i.boxes.xyxy, i.boxes.cls, i.boxes.conf):
            x1, y1, x2, y2, cls = map(int, [x1, y1, x2, y2, cls])
            #print("Percentage:", conf)
            #print(cls, i.names[cls])

            # Draw the bounding box on the frame
            color = (0, 255, 0)
            if cls == 0:
                color = (0, 0, 255)

            results = model.track(frame, persist=True)
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            print("boxes",boxes)
            print("track_ids",track_ids)
            print("position",x1,y1,x2,y2)
            cropped_image = frame[y1:y2, x1:x2]  # Crop the region of the detected object
            if cropped_image.size != 0:  # Ensure the cropped region is valid
                cropped_path = os.path.join(output_dir, f"cropped_{cropped_count:04d}_{i.names[cls]}_{conf:.2f}.jpg")
                cv2.imwrite(cropped_path, cropped_image)
                cropped_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, "%s %.3f" % (i.names[cls], conf), (x1, y1 + 20), 0, 0.7, color, 2)

    # Display the frame
    cv2.imshow("Video", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord("q"):
        break

    # Handle audio
    if val != 'eof' and audio_frame is not None:
        img, t = audio_frame
    frame_count += 1
video.release()
cv2.destroyAllWindows()
