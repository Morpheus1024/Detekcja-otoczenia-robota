import cv2
# IP address of the camera
ip_address = '192.168.0.82'

# Video capture from IP camera
cap = cv2.VideoCapture(f"http://{ip_address}:12344/video_feed")

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Failed to open camera")
    exit()

# Read and display frames from the camera
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame from camera")
        break

    cv2.imshow("Camera", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
