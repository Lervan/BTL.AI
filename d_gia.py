import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import os
from sklearn.metrics import confusion_matrix, accuracy_score
import final 

'''ĐIỀU KIỆN ĐÁNH GIÁ

Đủ ánh sáng
- Điều kiện phòng có ánh sáng ổn định
- Khoảng cách với webcam từ 25 - 70cm
- Được chụp từ nhiều vị trí trong khoảng webcam đoc được
- Mỗi cử chỉ sẽ được thử 300 lần (Mỗi bên tay 150 lần)
- Tập trung chính ở các góc độ đối diện lòng bàn tay (dễ đọc được các khớp ngón tay)
- Hạn chế lật ngược bàn tay hoặc gập bàn tay

Thiếu ánh sáng
- Điều kiện phòng thiếu ánh sáng
- Khoảng cách với web camsẽ gần hơn từ 25 - 70cm
- Được chụp từ nhiều vị trí trong khoảng webcam đọc được
- Mỗi cử chỉ sẽ được thử 300 lần (Mỗi bên tay 100 lần)
- Tập trung chính ở các góc độ đối diện lòng bàn tay (dễ đọc được các khớp ngón tay)
- Hạn chế lật ngược bàn tay hoặc gập bàn tay'''

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

Nhan_cu_chi = [
    "UNKNOWN",     
    "Smoke",         
    "OK",           
    "Ban Tim <3",  
    "ROCK",        
    "Say Hi",      
    "FIVE (Xoe ban tay)",        
    "WARNING!",     
    "Nam Dam"    
]

FILE_NAME_CSV = "du_lieu_toa_do.csv"

def khoi_tao_csv(filename):
    if not os.path.exists(filename):
        with open(filename, mode='w', newline = '') as f:
            writer = csv.writer(f)
            headers = ['label']
            for i in range(21):
                headers.extend([f'x{i}', f'y{i}'])
            writer.writerow(headers)

def luu_vao_csv(label_id, hand_landmarks):
    with open(FILE_NAME_CSV, mode='a', newline = '') as f:
        writer = csv.writer(f)
        row = [label_id]
        for lm in hand_landmarks.landmark:
            row.extend([lm.x, lm.y])
        writer.writerow(row)

khoi_tao_csv(FILE_NAME_CSV) 

y_chinh_xac = []
y_du_doan = []
gesture_counts = {label: 0 for label in Nhan_cu_chi}

cap = cv2.VideoCapture(0)

print(f"--- DA KHOI TAO FILE: {FILE_NAME_CSV} ---")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frame_rgb)
    
    nhan_hien_tai = "UNKNOWN"
    toa_do_hien_tai = None 
    
    if results.multi_hand_landmarks:
        for hand_lms, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            toa_do_hien_tai = hand_lms
            
            label_info = hand_info.classification[0].label
            finger_up_list, _ = final.dem_ngon_tay(hand_lms.landmark, label_info) 
            nhan_hien_tai = final.nhan_dien_cu_chi(hand_lms.landmark, finger_up_list)
            break 
    
    cv2.putText(frame, f"Doc duoc: {nhan_hien_tai}", (180, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (99, 255, 199), 2)

    y_offset = 20
    for idx, label in enumerate(Nhan_cu_chi):
        count = gesture_counts[label]
        color = (0, 255, 255) if count > 0 else (180, 180, 180)
        cv2.putText(frame, f"{idx}: {label} ({count})", (10, 30 + idx * 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    cv2.imshow("Thu thap du lieu", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if ord('0') <= key <= ord('8'):
        label_id = key - ord('0')
        label_name = Nhan_cu_chi[label_id]
        

        y_chinh_xac.append(label_name)
        y_du_doan.append(nhan_hien_tai)
        gesture_counts[label_name] += 1
        

        if toa_do_hien_tai:
            luu_vao_csv(label_id, toa_do_hien_tai)
            print(f">> Da luu toa do cho nhan {label_id} ({label_name}) vao file CSV")
        else:
            print("!! Khong tim thay tay de luu toa do !!")
            
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if len(y_chinh_xac) > 0:
    Unique_nhan_cu_chi = sorted(list(set(y_chinh_xac + y_du_doan)))
    ma_tran_nham_lan = confusion_matrix(y_chinh_xac, y_du_doan, labels = Unique_nhan_cu_chi)
    acc_total = accuracy_score(y_chinh_xac, y_du_doan)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        class_accuracies = ma_tran_nham_lan.diagonal() / ma_tran_nham_lan.sum(axis=1)

    report = []
    report.append(f"Tong mau: {len(y_chinh_xac)}")
    report.append(f"Do chinh xac: {acc_total*100:.2f}%")
    for i, label in enumerate(Unique_nhan_cu_chi):
        acc = class_accuracies[i]
        acc_str = "Khong co mau nao" if np.isnan(acc) else f"{acc*100:.2f}%"
        report.append(f"- {label}: {acc_str}")
    
    full_text = "\n".join(report)
    print("\n" + full_text)
    
    with open("bao_cao_ket_qua.txt", "w", encoding="utf-8") as f:
        f.write(full_text)

    plt.figure(figsize=(10, 8))
    sns.heatmap(ma_tran_nham_lan, annot=True, fmt='d', cmap='Reds', xticklabels = Unique_nhan_cu_chi, yticklabels = Unique_nhan_cu_chi)
    plt.title(f'Ma tran nham lan - Ty le chinh xac: {acc_total*100:.2f}%')
    plt.savefig('bieu_do_ket_qua.png')
    plt.show()
