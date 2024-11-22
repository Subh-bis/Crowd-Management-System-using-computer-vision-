#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#voice assistant feature
import cv2
import pyttsx3

# Load the pre-trained Haar Cascade classifiers for face and upper body detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Set the threshold for overcrowding
crowd_threshold = 5  # You can adjust this limit as needed
last_announcement = ""  # Track the last announcement to avoid repetitive announcements

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is read correctly, ret is True
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break
    
    # Convert frame to grayscale (Haar Cascade works on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    num_individuals = len(faces)  # Start counting the number of detected individuals with faces

    # If no faces are detected, try detecting upper bodies
    if num_individuals == 0:
        upper_bodies = upperbody_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        num_individuals += len(upper_bodies)  # Add the detected upper bodies to the count
        for (x, y, w, h) in upper_bodies:
            # Draw a rectangle around the detected upper body (back of the head approximation)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Back of Head Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Save the image when back of the head is detected
            cv2.imwrite('back_of_head.png', frame)
    else:
        # Draw a rectangle around each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle for face
            cv2.putText(frame, 'Face Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            # Save the image when a face is detected
            cv2.imwrite('face_detected.png', frame)

    # Check if the count exceeds the threshold and display overcrowding warning
    if num_individuals > crowd_threshold:
        cv2.putText(frame, 'Warning: Overcrowded!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        announcement = f"Warning: Overcrowded! {num_individuals} individuals detected."
    else:
        cv2.putText(frame, f'Occupancy: {num_individuals}/{crowd_threshold}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        announcement = f"{num_individuals} individuals detected. The occupancy is within the safe limit."

    # Make the voice announcement only if it's different from the last announcement
    if announcement != last_announcement:
        engine.say(announcement)
        engine.runAndWait()
        last_announcement = announcement  # Update last announcement to avoid repetition

    # Display the frame with rectangles
    cv2.imshow('Face and Head Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

