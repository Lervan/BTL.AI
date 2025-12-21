import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

# 1. Đọc dữ liệu
data = pd.read_csv('du_lieu_tay1.csv')



# 2. Chia dữ liệu: X là tọa độ, y là nhãn (0 hoặc 1)
X = data.drop('label', axis=1)
y = data['label']

# 3. Chia tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

noise = np.random.normal(0, 0.02, X_test.shape) 

# Cộng nhiễu vào dữ liệu Test
X_test_noisy = X_test + noise

# 4. Train mô hình (Dùng Random Forest - nhẹ và chính xác)
model = RandomForestClassifier(n_estimators=20,     # Giảm số lượng cây xuống (mặc định là 100)
    max_depth=4,         # Cực quan trọng: Chỉ cho cây sâu 3 tầng thôi (nó sẽ không học được chi tiết thừa)
    min_samples_split=5, # Phải có ít nhất 5 mẫu mới được chia nhánh tiếp
    random_state=42
)
model.fit(X_train, y_train)
 

# 5. Kiểm tra độ chính xác
y_pred = model.predict(X_test_noisy)
print(f"Độ chính xác: {(accuracy_score(y_test, y_pred) * 100)}%")

# 6. Lưu mô hình vào file để dùng sau này
with open('model_mong_rong.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("Đã lưu file model_mong_rong.pkl thành công!")