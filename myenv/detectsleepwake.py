import cv2
import mediapipe as mp

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Video input
video_path = './my_vid.mp4'
cap = cv2.VideoCapture(video_path)

# Define eye landmark indices for left and right eyes
left_eye_indices = [33, 246, 161, 160, 159, 158, 157, 173, 133]
right_eye_indices = [362, 398, 384, 385, 386, 387, 263, 249, 390, 373]

def eye_aspect_ratio(eye_landmarks):
    # Calculate EAR (Eye Aspect Ratio) for a given set of eye landmarks
    vertical_dist1 = ((eye_landmarks[1].x - eye_landmarks[5].x) ** 2 + (eye_landmarks[1].y - eye_landmarks[5].y) ** 2) ** 0.5
    vertical_dist2 = ((eye_landmarks[2].x - eye_landmarks[4].x) ** 2 + (eye_landmarks[2].y - eye_landmarks[4].y) ** 2) ** 0.5
    horizontal_dist = ((eye_landmarks[0].x - eye_landmarks[3].x) ** 2 + (eye_landmarks[0].y - eye_landmarks[3].y) ** 2) ** 0.5
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with FaceMesh
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract left and right eye landmarks
            left_eye_landmarks = [face_landmarks.landmark[idx] for idx in left_eye_indices]
            right_eye_landmarks = [face_landmarks.landmark[idx] for idx in right_eye_indices]
            
            # Calculate Eye Aspect Ratio (EAR) for left and right eyes
            left_ear = eye_aspect_ratio(left_eye_landmarks)
            right_ear = eye_aspect_ratio(right_eye_landmarks)
            
            # Define threshold for blink detection
            ear_threshold = 0.2
            
            # Determine blink status
            left_eye_blink = left_ear < ear_threshold
            right_eye_blink = right_ear < ear_threshold
            
            # Print blink status
            if left_eye_blink:
                print("Left Eye Blink Detected")
            if right_eye_blink:
                print("Right Eye Blink Detected")
    
    # Display processed frame (optional)
    cv2.imshow('Video', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()
