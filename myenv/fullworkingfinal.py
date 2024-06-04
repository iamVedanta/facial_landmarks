import cv2
from scipy.spatial import distance as dist
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [20, 50], invert=True)

EYE_AR_THRESH = 0.2  # Eye aspect ratio threshold for closed eyes
COUNTER_BLINK = 0
COUNTER_SLEEP = 0
COUNTER_AWAKE = 0
color = (255, 0, 255)

cap = cv2.VideoCapture('babyblink.mp4')

# Get the video frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object for MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

while True:
    success, img = cap.read()
    if not success:
        break
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        left_eye = [face[i] for i in range(159, 145, -1)]  # Landmarks for the left eye
        right_eye = [face[i] for i in range(386, 374, -1)]  # Landmarks for the right eye

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        ear = (left_ear + right_ear) / 2.0

        if ear < EYE_AR_THRESH:
            cv2.putText(img, 'Sleeping', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            COUNTER_BLINK = 0
            COUNTER_SLEEP = 0
            COUNTER_AWAKE += 1
            if COUNTER_AWAKE >= 20:
                cv2.putText(img, 'Awake', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        imgPlot = plotY.update(ear, color)
        img = cv2.resize(img, (640, 640))
        imgStack = cv2.vconcat([img, imgPlot])
    else:
        img = cv2.resize(img, (640, 640))
        imgStack = cv2.vconcat([img, img])

    # Resize imgStack to the original video dimensions before writing
    imgStack = cv2.resize(imgStack, (frame_width, frame_height))
    out.write(imgStack)
    cv2.imshow("Image", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
