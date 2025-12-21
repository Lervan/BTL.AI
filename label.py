import cv2
import mediapipe as mp
import csv
import os

# Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Tạo file CSV nếu chưa có
if not os.path.exists('du_lieu_tay1.csv'):
    with open('du_lieu_tay1.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        # Tạo header: label, x0, y0, ..., x20, y20
        header = ['label']
        for i in range(21):
            header += [f'x{i}', f'y{i}']
        writer.writerow(header)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Lấy tọa độ 21 điểm
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y])
            
            # HƯỚNG DẪN THU THẬP:
            # Nhấn phím '1' khi đang Bắn Tim
            # Nhấn phím '0' khi đang làm cử chỉ Khác (để AI phân biệt)
            k = cv2.waitKey(1)
            if k == ord('1'):
                with open('du_lieu_tay1.csv', mode='a', newline='') as f:
                    csv.writer(f).writerow([1] + row)
                cv2.putText(frame, "Da luu: BAN TIM (1)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            elif k == ord('0'):
                with open('du_lieu_tay1.csv', mode='a', newline='') as f:
                    csv.writer(f).writerow([0] + row)
                cv2.putText(frame, "Da luu: KHONG PHAI TIM (0)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Thu thap du lieu", frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()