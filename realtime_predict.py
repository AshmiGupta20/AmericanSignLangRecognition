import cv2
import mediapipe as mp
import joblib
import numpy as np
import time
from word_suggester import suggest_words
from dictionary_lookup import get_definition

# ---- Load models ----
model = joblib.load("models/asl_model.pkl")
autoencoder = joblib.load("models/autoencoder_pca.pkl")
scaler = joblib.load("models/autoencoder_scaler.pkl")
threshold = joblib.load("models/autoencoder_threshold.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

caption = ""
current_word = ""
last_added = ""
last_time = time.time()
STABLE_TIME = 1.2

suggestions = []
definition_text = ""

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
            row = np.array(row).reshape(1, -1)

            # --- PCA-based autoencoder check: is this a known hand pose? ---
            row_scaled = scaler.transform(row)
            compressed = autoencoder.transform(row_scaled)
            reconstruction = autoencoder.inverse_transform(compressed)
            error = np.mean(np.square(row_scaled - reconstruction))

            if error > threshold:
                current_pred = "UNKNOWN"
            else:
                current_pred = model.predict(row)[0]

        label_color = (0, 0, 255) if current_pred == "UNKNOWN" else (0, 255, 0)
        cv2.putText(frame, f"Detected: {current_pred}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)

        if current_pred == last_added:
            pass
        elif time.time() - last_time > STABLE_TIME:
            if current_pred == "SPACE":
                caption += current_word + " "
                definition_text = get_definition(current_word) if current_word else ""
                current_word = ""
                suggestions = []
            elif current_pred == "DEL":
                if current_word:
                    current_word = current_word[:-1]
                else:
                    caption = caption[:-1]
            elif current_pred in ("NOTHING", "UNKNOWN"):
                pass
            else:
                current_word += current_pred
                suggestions = suggest_words(current_word)

            last_added = current_pred
            last_time = time.time()
    else:
        last_added = ""

    # --- Caption bar ---
    cv2.rectangle(frame, (0, frame.shape[0]-90), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    full_display = (caption + current_word)[-60:]
    cv2.putText(frame, full_display, (10, frame.shape[0]-60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # --- Suggestions line ---
    if suggestions:
        sugg_text = "Suggestions: " + ", ".join(suggestions)
        cv2.putText(frame, sugg_text, (10, frame.shape[0]-35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # --- Definition line ---
    if definition_text:
        def_display = definition_text[:70]
        cv2.putText(frame, def_display, (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("ASL Recognition - Live Captioning", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        caption = ""
        current_word = ""
        suggestions = []
        definition_text = ""

cap.release()
cv2.destroyAllWindows()
