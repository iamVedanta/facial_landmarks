import cv2
import dlib

# Load the Dlib face detector and the facial landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define a function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold for the eye aspect ratio to determine if eyes are closed
EYE_AR_THRESHOLD = 0.25
EYE_AR_CONSEC_FRAMES = 3

# Initialize counters
COUNTER = 0
SLEEPING = False

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    
    for face in faces:
        landmarks = landmark_predictor(gray, face)
        landmarks = [(p.x, p.y) for p in landmarks.parts()]
        
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        ear = (left_ear + right_ear) / 2.0
        
        # Check if the person is sleeping
        if ear < EYE_AR_THRESHOLD:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                SLEEPING = True
                cv2.putText(frame, "SLEEPING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            COUNTER = 0
            SLEEPING = False
            cv2.putText(frame, "AWAKE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw landmarks on eyes
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
