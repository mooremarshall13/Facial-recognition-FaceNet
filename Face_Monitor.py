import cv2
import os
import time
import numpy as np
from threading import Thread
from keras_facenet import FaceNet
import pickle

# Initialize OpenCV face detection using Haar Cascade
haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Load FaceNet model
face_net = FaceNet()

# Load face database
with open("data.pkl", "rb") as file:
    database = pickle.load(file)

# Directory to save unknown face images
unknown_faces_dir = "unknown_faces"
os.makedirs(unknown_faces_dir, exist_ok=True)

# Function to process each frame
def process_frame(frame):
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    
    for (x, y, w, h) in faces:
        face_img = cv2.resize(frame[y:y+h, x:x+w], (160, 160))
        face_signature = face_net.embeddings(np.expand_dims(face_img, axis=0))
        
        min_dist = 0.94
        identity = 'Unknown'
        
        # Check distance to known faces in the database
        for key, value in database.items():
            dist = np.linalg.norm(value - face_signature)
            if dist < min_dist:
                min_dist = dist
                identity = key
        
        # Draw rectangle around the face
        color = (0, 0, 255) if identity == 'Unknown' else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, identity, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Save frame if unknown face detected
        if identity == 'Unknown':
            unknown_face_filename = os.path.join(unknown_faces_dir, f"unknown_face_{time.strftime('%Y%m%d%H%M%S')}.jpg")
            cv2.imwrite(unknown_face_filename, frame)
            print("Unknown face captured and saved.")

# Function to read frames from the camera
def read_frames(cap):
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            current_datetime = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, current_datetime, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Process every frame
            process_frame(frame)
            
            # Display the resulting frame
            cv2.imshow('frame', frame)
            
            # Break the loop on 'ESC' key press
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        # Release video capture and close windows
        cap.release()
        cv2.destroyAllWindows()

# Main function
def main():
    cap = cv2.VideoCapture(0)
    
    # Create and start thread for reading frames
    frame_thread = Thread(target=read_frames, args=(cap,))
    frame_thread.start()

    # Wait for the frame thread to finish
    frame_thread.join()

if __name__ == "__main__":
    main()
