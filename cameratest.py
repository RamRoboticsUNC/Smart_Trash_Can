import cv2

# Open the default webcam (0 is the first webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was not captured successfully, break the loop
    if not ret:
        print("Error: Could not read frame")
        break

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # Press 'q' to quit the webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the last captured frame as an image
cv2.imwrite('image.jpg', frame)

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
