import numpy as np
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

mixer.init()
mixer.music.load("music.wav")


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[3], mouth[9])
    B = distance.euclidean(mouth[2], mouth[10])
    C = distance.euclidean(mouth[4], mouth[8])
    D = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B + C) / (3.0 * D)
    return mar


eye_thresh = 0.25
mouth_thresh = 0.75
frame_check = 20

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

cap = cv2.VideoCapture(0)
flag = 0
yawn_flag = 0
blink_count = 0
alert_count = 0
yawn_count = 0
ear = 0
mar = 0


def draw_count_boxes(frame, ear, mar, ear_progress, mar_progress):
    # Blink Count
    cv2.rectangle(frame, (440, 20), (680, 70), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"Blink Count: {blink_count}",
        (450, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )

    # Alert Count
    cv2.rectangle(frame, (440, 80), (680, 130), (0, 0, 255), 2)
    cv2.putText(
        frame,
        f"Alert Count: {alert_count}",
        (450, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )

    # Yawn Count
    cv2.rectangle(frame, (440, 140), (680, 190), (0, 255, 255), 2)
    cv2.putText(
        frame,
        f"Yawn Count: {yawn_count}",
        (450, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )

    # EAR and MAR values at the bottom of the screen
    cv2.putText(
        frame,
        f"EAR: {ear:.2f}",
        (10, 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"MAR: {mar:.2f}",
        (300, 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # Progress bar for drowsiness alert (EAR)
    ear_progress_width = int((ear_progress / frame_check) * 200)
    cv2.rectangle(frame, (10, 450), (10 + ear_progress_width, 480), (0, 0, 255), -1)

    # Progress bar for yawning alert (MAR)
    mar_progress_width = int((mar_progress / frame_check) * 200)
    cv2.rectangle(frame, (10, 420), (10 + mar_progress_width, 440), (0, 255, 255), -1)


while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        mar = mouth_aspect_ratio(mouth)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 2)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 2)
        cv2.drawContours(frame, [mouthHull], -1, (255, 0, 0), 2)

        if ear < eye_thresh:
            flag += 1
            if flag == 1:
                blink_count += 1

            if flag >= frame_check:
                cv2.putText(
                    frame,
                    "************ DROWSINESS ALERT! ************",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "You are feeling asleep!",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                mixer.music.play()
                alert_count += 1
        else:
            flag = 0

        if mar > mouth_thresh:
            yawn_flag += 1
            if yawn_flag == 1:
                yawn_count += 1
            cv2.putText(
                frame,
                "************ YAWNING ALERT! ************",
                (10, 400),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )
            mixer.music.play()
        else:
            yawn_flag = 0

        # Draw the EAR, MAR values, and progress bars
        draw_count_boxes(frame, ear, mar, flag, yawn_flag)

    cv2.imshow("Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
