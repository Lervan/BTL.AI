import cv2
import mediapipe as mp
import time 
import math
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")

try:
    with open('model_ban_tim.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Đã load model AI thành công!")
except FileNotFoundError:
    model = None
    print("CẢNH BÁO: Không tìm thấy file 'model_ban_tim.pkl'. Chức năng bắn tim sẽ tắt.")

try:
    with open('model_mong_rong.pkl', 'rb') as f:
        model1 = pickle.load(f)
    print("Đã load model AI thành công!")
except FileNotFoundError:
    model1 = None
    print("CẢNH BÁO: Không tìm thấy file 'model_mong_rong.pkl'. Chức năng bắn tim sẽ tắt.")

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(max_num_hands = 2, min_detection_confidence = 0.7, min_tracking_confidence = 0.7)
pose_dectector = mp_pose.Pose(min_detection_confidence = 0.7)

count = 0
direction = 0
form = 0   

def caculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if (angle > 180.0):
        angle = 360 - angle
    return angle

def gio_ngon_tro(lm_list):
    if not lm_list:
        return False
    
    if lm_list[8][2] > lm_list[6][2]:
        return False

    for id in [12, 16, 20]:
        if lm_list[id][2] < lm_list[id - 2][2]: 
            return False

    if lm_list[4][2] < lm_list[6][2]:
        return False

    return True

def countFingers(hand_landmarks, hand_label):
    fingers_status = [] 

    thumb_tip_x = hand_landmarks.landmark[4].x
    thumb_ip_x = hand_landmarks.landmark[3].x

    if hand_label == "Right":
        fingers_status.append(1 if thumb_tip_x < thumb_ip_x else 0)
    else:
        fingers_status.append(1 if thumb_tip_x > thumb_ip_x else 0)

    finger_tips_ids = [8, 12, 16, 20]
    
    for tip_id in finger_tips_ids:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers_status.append(1)
        else:
            fingers_status.append(0)
            
    return fingers_status, sum(fingers_status)

def is_both_hands_up(landmarks):
    if not landmarks:
        return False
    
    tay_trai = landmarks[15].y
    vai_trai = landmarks[11].y

    tay_phai = landmarks[16].y
    vai_phai = landmarks[12].y

    return (tay_phai < vai_phai) and (tay_trai < vai_trai)

def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def recognize_gesture(landmarks, fnger_up_list, model = None, model1 = None):
    if model is not None:
        row = []
        for lm in landmarks:
            row.extend([lm.x, lm.y])
        X = np.array([row])
        try:
            prediction = model.predict(X)[0]
            if prediction == 1:
                return "Ban Tim :3"
        except Exception as e:
            pass

    if model1 is not None:
        row = []
        for lm in landmarks: 
            row.extend([lm.x, lm.y])
        X = np.array([row])
        try:
            prediction = model1.predict(X)[0]
            if prediction == 1:
                return "Dragon Nail"
        except Exception as e:
            pass

    d = math.hypot(landmarks[4].x - landmarks[8].x, landmarks[4].y - landmarks[8].y)
    if d < 0.05: return "OK"
    
    if (finger_up_list == [0,1,0,0,1]):
        return "ROCK"
    
    if (finger_up_list == [1,0,0,0,0]):
        return "Like"
    
    if (finger_up_list == [0,1,1,0,0]):
        return "Say Hi"
    
    if sum(finger_up_list) == 5:
        return "FIVE"
    
    if (finger_up_list == [0,0,1,0,0]):
        return "WARNING!"     
    return "UNKNOWN"
    

cap = cv2.VideoCapture(0)

current_mode = "HAND"
start_time = 0
holding = False


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    count_time = 0
    if current_mode == "HAND":
        results = hands_detector.process(frame_rgb)
        detected_gesture_switch = False
        total_finger_count = 0
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lms, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                label = hand_info.classification[0].label

                finger_up_list, total_fingers = countFingers(hand_lms, label)
                total_finger_count += total_fingers

                gesture_name = recognize_gesture(hand_lms.landmark, finger_up_list, model, model1)
                
                lm_list = []
                for id, lm in enumerate(hand_lms.landmark):
                    lm_list.append([id, int(lm.x * w), int(lm.y * h)])

                if gio_ngon_tro(lm_list):
                    detected_gesture_switch = True

                wrist_x = int(hand_lms.landmark[0].x * w)
                wrist_y = int(hand_lms.landmark[0].y * h)
                
                cv2.putText(frame, f"{label} Hand", (wrist_x - 40, wrist_y + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Fingers: {total_fingers}", (wrist_x - 40, wrist_y + 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"{gesture_name}", (wrist_x - 40, wrist_y - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.putText(frame, f"Total Fingers: {total_finger_count}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        thoi_gian_can = 5
        if (detected_gesture_switch):
            if not holding:
                bat_dau = time.time()
                holding = True
            
            count_time = time.time() - bat_dau

            cv2.putText(frame, f"Giu Ngon Tro: {int(count_time)}s/{thoi_gian_can}s", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if (count_time > thoi_gian_can):
                current_mode = "POSE"
                holding = False
                bat_dau = 0
                print ("Chuyen sang che do POSE")
        else:
            holding = False

    elif current_mode == "POSE":
        results = pose_dectector.process(frame_rgb)
        detected_gesture = False

        if (results.pose_landmarks):
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if is_both_hands_up(results.pose_landmarks.landmark):
                detected_gesture = True

            landmarks = results.pose_landmarks.landmark

            vai = [landmarks[11].x, landmarks[11].y]
            khuyu_tay = [landmarks[13].x, landmarks[13].y]
            co_tay = [landmarks[15].x, landmarks[15].y]

            angle = caculate_angle(vai, khuyu_tay, co_tay)

            h, w, c = frame.shape
            cv2.putText(frame, str(int(angle)),
                        (int(khuyu_tay[0] * w), int(khuyu_tay[1] * h)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (111,111,111), 2)

            per = np.interp(angle, (70, 160), (100, 0))
            bar_x = w - 50
            bar_y = np.interp(angle, (70, 160), (400 , 100))

            if (per == 100):
                if (direction == 0):
                    direction = 1

            if (per == 0):
                if (direction == 1):
                    count += 1
                    direction = 0
            
            cv2.rectangle(frame, (bar_x, 100), (bar_x+25, 400), (0, 255, 0), 2)
            cv2.rectangle(frame, (bar_x, int(bar_y)), (bar_x+25, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, f'{int(per)}%', (bar_x-10, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)        
        
            cv2.rectangle(frame, (0, h-100), (150, h), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, str(int(count)), (30, h-20), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)


        thoi_gian_can = 5
        if detected_gesture:
            if not holding:
                start_time = time.time()
                holding = True

            count_time = time.time() - start_time


            cv2.putText(frame, f"Gio 2 tay: {int(count_time)}s/{(thoi_gian_can)}s", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            if count_time > thoi_gian_can:
                current_mode = "HAND"
                holding = False
                start_time = 0
                print("Chuyen sang che do HAND")
                count = 0 
        else:
            holding = False
        
    cv2.putText(frame, f"MODE : {current_mode}", (420, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,244,0), 2)
    
    cv2.imshow("Smart", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()