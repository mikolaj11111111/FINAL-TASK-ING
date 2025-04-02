import pandas as pd

df = pd.read_csv('Main_task_data_training.csv', encoding='ISO-8859-1', sep=';', on_bad_lines='skip')

"""
print("HEAD")
print(df.head())
print('==" * 100')
print("SHAPE")
print(df.shape)
print("==" * 100)
print("INFO")
print(df.info())
print("==" * 100)
print(df)
print(df.columns)

"""
print(len(df.columns))
for column in df.columns:
    if df[column].count()/ 500000 < 0.51:
        print(f"{column}:", df[column].count()/500000)

print(df.head())
#print(df['SUBTYPE','SHARE', 'RENEWABLE_ENERGY_HEATING', 'RENEWABLE_ENERGY_ELECTRIC', 'CERTIFICATE_PHI',"COMMUNITY", ''])

df.drop(columns=['SUBTYPE','SHARE', 'RENEWABLE_ENERGY_HEATING', 'RENEWABLE_ENERGY_ELECTRIC', 'CERTIFICATE_PHI',"COMMUNITY", 'UTIL_WATER_INTAKE', 'UTIL_WATER_SUPPLY', 'UTIL_ELECTRICITY',
        'UTIL_GAS', 'UTIL_SEWAGE_SYSTEM_CONNECTION', 'UTIL_SEWAGE_SYSTEM_OWN', 'UTIL_SEWAGE_SYSTEM_CONNECTION', 'PREMISSES_STANDARD', 'PREMISSES_INDEX_PED', 'PREMISSES_INDEX_FED', 'PREMISSES_INDEX_UED',
        'PREMISSES_ENERGY_PERF_CERT_DATE', 'PREMISSES_ENERGY_PERF_CERT_VALI', 'PREMISSES_RES_SHARE', 'PREMISSES_CO2_EMMISSION', 'BUILDING_CHAMBER_NO', 'BUILDING_INDEX_PED', 'BUILDING_INDEX_FED',
        'BUILDING_INDEX_UED', 'BUILDING_ENERGY_PERF_CERT_DATE', 'BUILDING_ENERGY_PERF_CERT_VALI', 'BUILDING_RES_SHARE', 'BUILDING_CO2_EMMISSION', 'PARKING_SPACE_ID', 'PARKING_KIND', 'NUMBER'],
        inplace =True)
print(len(df.columns))

for column in df.columns:
    print(f"{column}:", df[column].count()/500000)

