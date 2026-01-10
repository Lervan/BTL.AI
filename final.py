import cv2
import mediapipe as mp
import time 
import math
import numpy as np

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(max_num_hands = 2, min_detection_confidence = 0.7, min_tracking_confidence = 0.7)
pose_dectector = mp_pose.Pose(min_detection_confidence = 0.7)

def tinh_goc(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radian = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radian * 180.0 / np.pi)

    if (angle > 180.0):
        angle = 360 - angle
    return angle

def do_hai_tay(landmarks):
    if not landmarks:
        return False
    
    tay_trai = landmarks[15].y
    vai_trai = landmarks[11].y

    tay_phai = landmarks[16].y
    vai_phai = landmarks[12].y

    return (tay_phai < vai_phai) and (tay_trai < vai_trai)

def kiem_tra_ngon_cai(landmarks, label):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    if label == "Right":
        if thumb_tip.x > thumb_ip.x:
            return True
        else:
            return False 
    elif label == "Left":
        if thumb_tip.x < thumb_ip.x:
            return True 
        else:
            return False 
            
    return False

def gio_ngon_tro(landmarks, label):
    if landmarks[8].y > landmarks[6].y:
        return False
    if landmarks[12].y < landmarks[10].y: 
        return False
    if landmarks[16].y < landmarks[14].y: 
        return False 
    if landmarks[20].y < landmarks[18].y: 
        return False
    if kiem_tra_ngon_cai(landmarks, label) == 0:
        return False
    return True

def dem_ngon_tay(landmarks, label):
    trang_thai_ngon_tay = [] 
    if (kiem_tra_ngon_cai(landmarks, label)):
        trang_thai_ngon_tay.append(0)
    else:    
        trang_thai_ngon_tay.append(1)

    toa_do_dau_ngon_tay = [8, 12, 16, 20]
    
    for dau_ngon_id in toa_do_dau_ngon_tay:
        if landmarks[dau_ngon_id].y < landmarks[dau_ngon_id - 2].y:
            trang_thai_ngon_tay.append(1)
        else:
            trang_thai_ngon_tay.append(0)
            
    return trang_thai_ngon_tay, sum(trang_thai_ngon_tay)

def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def nhan_dien_cu_chi(landmarks, list_ngon_dang_gio):
    size_long_ban_tay = distance(landmarks[0], landmarks[9])
    if (size_long_ban_tay == 0): 
        size_long_ban_tay = 0.0000001
    khoang_cach_cham_ncai_ntro = distance(landmarks[4], landmarks[8])
    dang_cham = (khoang_cach_cham_ncai_ntro / size_long_ban_tay) < 0.2
    ngon_giua_dong = landmarks[12].y > landmarks[9].y
    ngon_ap_ut_dong = landmarks[16].y > landmarks[13].y
    ngon_ut_dong = landmarks[20].y > landmarks[17].y

    if (dang_cham and ngon_giua_dong and ngon_ap_ut_dong and ngon_ut_dong):
        return "Ban Tim <3"

    if (dang_cham and (ngon_giua_dong == 0) and (ngon_ap_ut_dong == 0) and (ngon_ut_dong == 0)): 
        return "OK"

    if (list_ngon_dang_gio == [0,1,0,0,1]):
        return "ROCK"
    
    if (list_ngon_dang_gio == [1,0,0,0,1]):
        return "Smoke"
    
    if (list_ngon_dang_gio == [0,1,1,0,0]):
        return "Say Hi"
    
    if sum(list_ngon_dang_gio) == 0:
        return "Nam Dam"

    if sum(list_ngon_dang_gio) == 5:
        return "FIVE (Xoe ban tay)"
    
    if (list_ngon_dang_gio == [0,0,1,0,0]):
        return "WARNING!"     
    return "UNKNOWN"

def main():
    cap = cv2.VideoCapture(0)
    count = 0
    direction = 0
    # form = 0   

    current_mode = "HAND"
    start_time = 0
    holding = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        frame_lam_mo = cv2.GaussianBlur(frame, (5, 5), 0)
        frame_rgb = cv2.cvtColor(frame_lam_mo, cv2.COLOR_BGR2RGB)

        count_time = 0
        if current_mode == "HAND":
            results = hands_detector.process(frame_rgb)
            detected_gesture_switch = False
            tong_2_ban_tay = 0
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lms, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                    mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                    label = hand_info.classification[0].label

                    list_ngon_dang_gio, tong_ngon_1_ban_tay = dem_ngon_tay(hand_lms.landmark, label)
                    tong_2_ban_tay += tong_ngon_1_ban_tay

                    ten_cu_chi = nhan_dien_cu_chi(hand_lms.landmark, list_ngon_dang_gio)

                    if gio_ngon_tro(hand_lms.landmark, label):
                        detected_gesture_switch = True

                    co_tay_x = int(hand_lms.landmark[0].x * w)
                    co_tay_y = int(hand_lms.landmark[0].y * h)
                    
                    cv2.putText(frame, f"{label} Hand", (co_tay_x - 40, co_tay_y + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Fingers: {tong_ngon_1_ban_tay}", (co_tay_x - 40, co_tay_y + 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"{ten_cu_chi}", (co_tay_x - 40, co_tay_y - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                cv2.putText(frame, f"Total Fingers: {tong_2_ban_tay}", (10, 100),
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

                if do_hai_tay(results.pose_landmarks.landmark):
                    detected_gesture = True

                landmarks = results.pose_landmarks.landmark

                vai = [landmarks[11].x, landmarks[11].y]
                khuyu_tay = [landmarks[13].x, landmarks[13].y]
                co_tay = [landmarks[15].x, landmarks[15].y]

                angle = tinh_goc(vai, khuyu_tay, co_tay)

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

if __name__ == "__main__":
    main()