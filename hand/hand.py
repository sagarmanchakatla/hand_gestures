import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyautogui as key


mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


model = load_model('mp_hand_gesture')

f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

gestures = {
    # 0:'RIGHT',
    8:'RIGHT',
    1:'LEFT',
    9:'LEFT',
    5:'UP',
    7:'UP'
}

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error in camera")
    
while True:
    _, frame = cap.read()

    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # get hand landmark prediction
    result = hands.process(framergb)
    # print(result)
    className = ''
    classID = int()
    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            print(classID)
            className = classNames[classID]
            print(className)
            
            
    ges_label = gestures.get(classID)  
    # print(ges_label)      
    
    if ges_label is not None:
        cv2.putText(frame, ges_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if ges_label == 'RIGHT':
            key.press('right')
        elif ges_label == 'UP':
            key.press('up')
        elif ges_label == 'LEFT':
            key.press('left')       
        
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()