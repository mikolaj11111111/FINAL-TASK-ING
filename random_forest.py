import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Załaduj dane
df = pd.read_csv('Main_task_data_training_cleaned.csv', encoding='ISO-8859-1', sep=';', on_bad_lines='skip')

# Przetwarzanie kolumny tekstowej
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].apply(
        lambda x: x.encode('ISO-8859-1').decode('utf-8') if isinstance(x, str) else x
    )

# Określamy kolumny, które nas interesują
columns_df = ['GUS_population', 'GUS_population_density', 'GUS_urbanisation_rate', 'GUS_population_change',
              'sal_void_eoq', 'sal_county_y_mean', 'unemp_void_y_pct', 'VALUATION_VALUE']

# Filtracja danych według MARKET_TYPE
df_pierw = df[df['MARKET_TYPE'] == 1].copy()
df_wtorn = df[df['MARKET_TYPE'] == 2].copy()
df_nieokr = df[df['MARKET_TYPE'] == 0].copy()

# Usuwamy kolumny, które nie są w columns_df
for column in df.columns:
    if column not in columns_df:
        df_pierw.drop(columns=column, inplace=True, errors='ignore')
        df_wtorn.drop(columns=column, inplace=True, errors='ignore')
        df_nieokr.drop(columns=column, inplace=True, errors='ignore')

df_pierw = pd.get_dummies(df_pierw, drop_first=True)
df_wtorn = pd.get_dummies(df_wtorn, drop_first=True)
df_nieokr = pd.get_dummies(df_nieokr, drop_first=True)

columns_to_remove = ['MARKET_TYPE', 'PREMISSES_ID', 'ADDRESS_ID', 'IS_CAPITAL']
df_pierw.drop(columns=columns_to_remove, inplace=True, errors='ignore')
df_wtorn.drop(columns=columns_to_remove, inplace=True, errors='ignore')
df_nieokr.drop(columns=columns_to_remove, inplace=True, errors='ignore')

# Obliczenie średniej lub mediany z GUS_population oraz GUS_population_density jako target_population
target_population = df_wtorn['GUS_population'].median() 
target_density = df_wtorn['GUS_population_density'].median()  

print(f"Wybrana wartość target_population: {target_population}")
print(f"Wybrana wartość target_density: {target_density}")

# Filtracja danych na podstawie różnicy w populacji i gęstości zaludnienia
population_tolerance = 0.2  
density_tolerance = 0.1  

filtered_data = df_wtorn[
    (abs(df_wtorn['GUS_population'] - target_population) / target_population <= population_tolerance) &
    (abs(df_wtorn['GUS_population_density'] - target_density) / target_density <= density_tolerance)
]

# Sprawdź liczbę próbek po filtracji
print(f"Liczba próbek po filtracji: {len(filtered_data)}")

if len(filtered_data) == 0:
    print("Brak danych po zastosowaniu 10% tolerancji. Zmniejszam tolerancję do 5%.")
    population_tolerance = 0.05  
    density_tolerance = 0.05 
    filtered_data = df_wtorn[
        (abs(df_wtorn['GUS_population'] - target_population) / target_population <= population_tolerance) &
        (abs(df_wtorn['GUS_population_density'] - target_density) / target_density <= density_tolerance)
    ]
    print(f"Liczba próbek po filtracji przy 5% tolerancji: {len(filtered_data)}")

filtered_data['GUS_combined'] = (filtered_data['GUS_population'] + filtered_data['GUS_population_density']) / 2

# Kodowanie zmiennych kategorycznych (One-Hot Encoding) na filtrowanych danych
filtered_data = pd.get_dummies(filtered_data, drop_first=True)

# Usuwanie niepotrzebnych kolumn po get_dummies()
filtered_data.drop(columns=columns_to_remove, inplace=True, errors='ignore')

# Przypisanie zmiennej y
y_filtered = filtered_data['VALUATION_VALUE']
X_filtered = filtered_data.drop(columns=['VALUATION_VALUE'])

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# Model Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Ocena modelu
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Ważność cech
print("Ważność cech (feature importances):", model.feature_importances_)

threshold = 0.05  
correct_predictions = abs(y_pred - y_test) / y_test < threshold
accuracy_percentage = correct_predictions.mean() * 100

