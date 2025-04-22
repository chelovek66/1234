import pandas as pd


# Load your data
ml = pd.read_csv(r"C:\\Users\\user\\OneDrive\\Рабочий стол\\ОМО ЛР\\2\\test.csv", encoding='CP1251')

print(ml.shape)
print(ml.iloc[388])
# ml.info()
ml['error'] = (ml['error'] == "в\\234\\223").astype(int)

print(ml.iloc[388])

ml.info()

ml['Info'] = ml['Info'].fillna('0')

ml.info()

print(ml.describe)