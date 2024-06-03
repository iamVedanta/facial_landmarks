import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Function to detect and print a list of facial landmarks for each frame of a video
def detect_and_print_landmarks(video_path, landmark_indices):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return
    
    # Loop through each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no more frames

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect facial landmarks
        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for landmark_index in landmark_indices:
                        if landmark_index < len(face_landmarks.landmark):
                            landmark = face_landmarks.landmark[landmark_index]
                            # Convert normalized coordinates to pixel coordinates
                            ih, iw, _ = frame.shape
                            x = int(landmark.x * iw)
                            y = int(landmark.y * ih)
                            z = landmark.z  # Depth (not used here)

                            # Print facial point
                            print(f"Facial Landmark {landmark_index}: ({x}, {y})")

                            # Draw circle on the landmark point
                            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # Display the frame
        cv2.imshow('Facial Landmarks', frame)
        
        # Check for user input to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Path to the video file
video_path = 'my_vid.mp4'

# List of landmark indices to display
landmark_indices = [101,50]  # Example list, modify with your desired indices

# Call the function to detect and print a list of facial landmarks for each frame of the video
detect_and_print_landmarks(video_path, landmark_indices)
