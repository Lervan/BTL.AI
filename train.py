import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

data = pd.read_csv('du_lieu_tay1.csv')

X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

noise = np.random.normal(0, 0.02, X_test.shape) 

X_test_noisy = X_test + noise

model = RandomForestClassifier(
    n_estimators=20,    
    max_depth=4,       
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)
 
y_pred = model.predict(X_test_noisy)
print(f"Độ chính xác: {(accuracy_score(y_test, y_pred) * 100)}%")

with open('model_mong_rong.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("Đã lưu file model_mong_rong.pkl thành công!")