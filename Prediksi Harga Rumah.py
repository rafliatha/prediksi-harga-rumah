import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Data yang diberikan
data = {
    'NAMA_RUMAH': ['Rumah Murah Hook Tebet Timur', 'Rumah Modern di Tebet dekat Stasiun', 'Rumah Mewah 2 Lantai', 
                'Rumah Baru Tebet', 'Rumah Bagus Tebet'],
    'HARGA': [3800000000, 4600000000, 3000000000, 430000000, 9000000000],
    'LB': [220, 180, 267, 40, 400],
    'LT': [220, 137, 250, 25, 355],
    'KT': [3, 4, 4, 2, 6],
    'KM': [3, 3, 4, 2, 5],
    'GRS': [0, 2, 4, 0, 3]
}

# Membuat DataFrame
df = pd.DataFrame(data)

# Fitur dan target
X = df[['LB', 'LT', 'KT', 'KM', 'GRS']]  # Fitur
y = df['HARGA']  # Target

# Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model regresi linier
model = LinearRegression()

# Melatih model dengan data pelatihan
model.fit(X_train, y_train)

# Prediksi harga rumah untuk data pelatihan dan data pengujian
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

# Menghitung MAE, MSE, dan R2 Score untuk data pelatihan
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Menghitung MAE, MSE, dan R2 Score untuk data pengujian
mae_test = mean_absolute_error(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

# Menampilkan hasil evaluasi
print("Evaluasi Model pada Data Pelatihan:")
print(f'MAE: {mae_train}')
print(f'MSE: {mse_train}')
print(f'R2 Score: {r2_train}')

print("\nEvaluasi Model pada Data Pengujian:")
print(f'MAE: {mae_test}')
print(f'MSE: {mse_test}')
print(f'R2 Score: {r2_test}')

# Prediksi harga rumah baru berdasarkan fitur
predicted_price = model.predict([[500, 1000, 2, 2, 1]])  # Contoh data fitur baru: LB=250, LT=200, KT=4, KM=3, GRS=2
print("\nPrediksi Harga Rumah Baru:")
print(f'Harga rumah impian anda diperkirakan sekitar IDR {predicted_price[0]:,.0f}')
