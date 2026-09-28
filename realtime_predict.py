import cv2
import mediapipe as mp
import joblib
import time

model = joblib.load("models/asl_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

caption = ""
last_added = ""
last_time = time.time()
STABLE_TIME = 1.2  # seconds a prediction must hold steady before committing

print("Controls: 'q' = quit | 'c' = clear caption")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_pred = None
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            current_pred = model.predict([row])[0]

        cv2.putText(frame, f"Detected: {current_pred}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if current_pred == last_added:
            pass  # waiting for hand to reset before re-adding same sign
        elif time.time() - last_time > STABLE_TIME:
            if current_pred == "SPACE":
                caption += " "
            elif current_pred == "DEL":
                caption = caption[:-1]
            elif current_pred == "NOTHING":
                pass
            else:
                caption += current_pred
            last_added = current_pred
            last_time = time.time()
    else:
        last_added = ""  # hand left frame — allow same sign to be added again next time

    # caption bar
    cv2.rectangle(frame, (0, frame.shape[0]-50), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.putText(frame, caption[-60:], (10, frame.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("ASL Recognition - Live Captioning", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        caption = ""

cap.release()
cv2.destroyAllWindows()