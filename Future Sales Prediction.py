# Gerekli Kütüphaneleri Yükleme
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Veri Setlerinin Yüklenmesi
item_categories = pd.read_csv('item_categories.csv')
items = pd.read_csv('items.csv')
shops = pd.read_csv('shops.csv')
train = pd.read_csv('sales_train.csv')
test = pd.read_csv('test.csv')

# Verilerin Birleştirilmesi
train = train.merge(items, on='item_id', how='left')
train = train.merge(item_categories, on='item_category_id', how='left')
test = test.merge(items, on='item_id', how='left')
test = test.merge(item_categories, on='item_category_id', how='left')

# Tarih Formatının Dönüştürülmesi
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month

# Toplam Satış Sütununun Eklenmesi
train['total_sales'] = train['item_price'] * train['item_cnt_day']

# Aykırı Değerlerin Temizlenmesi
train = train[(train['item_price'] > 0) & 
              (train['item_price'] < train['item_price'].quantile(0.999))]
train = train[(train['item_cnt_day'] < train['item_cnt_day'].quantile(0.999))]

# Aykırı Değerlerin Kutu Grafikleri
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=train['item_price'])
plt.title("Ürün Fiyatları (Aykırı Değerler Hariç)")
plt.subplot(1, 2, 2)
sns.boxplot(x=train['item_cnt_day'])
plt.title("Günlük Satış Miktarları (Aykırı Değerler Hariç)")
plt.tight_layout()
plt.show()

# Zaman Serisi Analizi
sales_daily = train.groupby('date')['item_cnt_day'].sum()

# Günlük Satışların Trendi
plt.figure(figsize=(14, 6))
plt.plot(sales_daily, label='Günlük Satışlar', color='blue')
plt.title("Günlük Satışların Zamanla Değişimi")
plt.xlabel("Tarih")
plt.ylabel("Satış Miktarı")
plt.legend()
plt.show()

# Özellik Mühendisliği
monthly_sales = train.groupby(['date_block_num', 'shop_id'], 
                              as_index=False).agg({'total_sales': 'sum'})

# Özelliklerin Ayrılması
X = monthly_sales[['date_block_num', 'shop_id']]  # Girdi Özellikleri
y = monthly_sales['total_sales']                  # Hedef Değişken

# Eğitim ve Test Verilerinin Ayrılması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                  random_state=42)

# XGBoost Modeli için GridSearchCV ile Hiperparametre Optimizasyonu
param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.2],
    'max_depth': [3, 5],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0],
    'gamma': [0, 0.2]
}

xgb_model = xgb.XGBRegressor()
grid_search = GridSearchCV(estimator=xgb_model, 
                           param_grid=param_grid, 
                           scoring='neg_mean_squared_error', 
                           cv=5, 
                           verbose=1, 
                           n_jobs=-1)
grid_search.fit(X_train, y_train)

# En İyi Parametrelerin Seçilmesi
best_model = grid_search.best_estimator_

# Model Performansının Değerlendirilmesi
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Validation RMSE:", rmse)

# Model Doğruluk Grafiği
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Tahminler')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', label='Doğru Çizgi')
plt.title("Gerçek vs Tahmin Değerleri")
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Değerleri")
plt.legend()
plt.show()

# Test Verisiyle Tahmin Yapılması
test['date_block_num'] = max(train['date_block_num']) + 1
test_X = test[['date_block_num', 'shop_id']]
test['item_cnt_month'] = best_model.predict(test_X)

# Tahmin Sonuçlarının Kaydedilmesi
submission = test[['ID', 'item_cnt_month']]
submission.to_csv('submission.csv', index=False)

print("Tahminler submission.csv dosyasına kaydedildi.")
