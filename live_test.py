import cv2
from ultralytics import YOLO

# Load your YOLOv8 model
model = YOLO('./runs/detect/train2/weights/best.pt')  # Replace with your model's path

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or replace with the correct index

# Set webcam resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Draw the results on the frame
    annotated_frame = results[0].plot()

    # Display the frame with annotations
    cv2.imshow('Fire Detection', annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
