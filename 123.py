import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import string 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download('stopwords')
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from matplotlib import pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
import numpy as np
from sklearn.model_selection import GridSearchCV

df = pd.read_csv(r"C:\Users\user\OneDrive\Рабочий стол\ОМО ЛР\2\labeled.csv", sep=',')
# print(df)
print(df.shape)
print(df.info)
print(df.head(10))

df["toxic"] = df["toxic"].apply(int)

print(df.head(10))

print("-----------0 ",df["toxic"].value_counts())

for i in df[df["toxic"]==1 ]["comment"].head(3):
    print(i,end='\n')
print('\n')
for i in df[df["toxic"]==0 ]["comment"].head(3):
    print(i,end='\n')

train_df, test_df = train_test_split(df, test_size=500, stratify = df["toxic"])

print(test_df.shape)
print(train_df.shape)

print(test_df["toxic"].value_counts())
print(train_df["toxic"].value_counts())

sentence_example = df.iloc[1]["comment"]
tokens = word_tokenize(sentence_example, language="russian")
token_without_punctuation = [i for i in tokens if i not in string.punctuation]
russian_stop_words = stopwords.words("russian")
tokens_without_stop_words_and_punctuation = [i for i in token_without_punctuation if i not in russian_stop_words]
snowball = SnowballStemmer(language="russian")
stemmed_tokens = [snowball.stem(i) for i in tokens_without_stop_words_and_punctuation]

print(f"Исходный текс: {sentence_example}")
print("------------------------------")
print(f"Токены: {tokens}")
print("------------------------------")
print(f"Токен без пунктуации: {token_without_punctuation}")
print("------------------------------")
print(f"Токен без пунктуации и стоп слова: {tokens_without_stop_words_and_punctuation}")
print("------------------------------")
print(f"Токен после стемминга: {stemmed_tokens}")

snowball = SnowballStemmer(language='russian')
russian_stop_words = stopwords.words("russian")

def tokenize_sentense(sentence: str):
    tokens = word_tokenize(sentence, language="russian")
    tokens = [i for i in tokens if i not in string.punctuation]
    tokens = [i for i in tokens if i not in russian_stop_words]
    tokens = [snowball.stem(i) for i in tokens]
    return tokens

z = tokenize_sentense(sentence_example)
print(z)

vectorizer = TfidfVectorizer(tokenizer= lambda x: tokenize_sentense(x), token_pattern=None)
features = vectorizer.fit_transform(train_df["comment"])


model = LogisticRegression(random_state = 0)
model.fit(features,train_df["toxic"])

model_pipline = Pipeline([
    ("vectorizer",TfidfVectorizer(tokenizer=lambda x: tokenize_sentense(x),token_pattern=None)),
    ("model",LogisticRegression(random_state = 0))
])

model_pipline.fit(train_df["comment"],train_df["toxic"])

z = model_pipline.predict(["А вот я люблю когда меня ебут"])

print(z)

print(precision_score(y_true=test_df["toxic"],y_pred=model_pipline.predict(test_df["comment"])))

print(recall_score(y_true=test_df["toxic"],y_pred=model_pipline.predict(test_df["comment"])))

prec, rec, thresholds = precision_recall_curve(y_true=test_df["toxic"],probas_pred=model_pipline.predict_proba(test_df["comment"])[:,1])

# Получаем предсказанные вероятности
y_scores = model_pipline.predict_proba(test_df["comment"])[:, 1]

# Создаем кривую Precision-Recall
precision, recall, _ = precision_recall_curve(test_df["toxic"], y_scores)

# Отображаем график
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
plt.show()


z = np.where( prec > 0.95)[0][0]
print(z)
z1 = precision_score(y_true=test_df["toxic"], y_pred=model_pipline.predict_proba(test_df["comment"])[:,1] > thresholds[z])
z2 = recall_score(y_true=test_df["toxic"], y_pred=model_pipline.predict_proba(test_df["comment"])[:,1]> thresholds[z])

print(z1)

print(z2)

grid_pipline=Pipeline([
    ("vectorizer",TfidfVectorizer(tokenizer=lambda x: tokenize_sentense(x),token_pattern=None)),
    ("model",
    GridSearchCV(
        LogisticRegression(random_state=0),
        param_grid={"C" : [0.1,1,10]},
        cv = 3,
        verbose= 4
    ) 
    )

])

grid_pipline.fit(train_df["comment"],train_df["toxic"])

model_pipline_C10 = Pipeline([
    ("vectorizer",TfidfVectorizer(tokenizer=lambda x: tokenize_sentense(x),token_pattern=None)),
    ("model",LogisticRegression(random_state = 0,C=10))
])

model_pipline_C10.fit(train_df["comment"],train_df["toxic"])

print(precision_score(y_true=test_df["toxic"],y_pred=model_pipline_C10.predict(test_df["comment"])))

print(recall_score(y_true=test_df["toxic"],y_pred=model_pipline_C10.predict(test_df["comment"])))

prec_C10, rec_c10, thresholds_c10 = precision_recall_curve(y_true=test_df["toxic"],probas_pred=model_pipline_C10.predict_proba(test_df["comment"])[:,1])

# Получаем предсказанные вероятности
y_scores_c10 = model_pipline_C10.predict_proba(test_df["comment"])[:, 1]

# Создаем кривую Precision-Recall
precision_c10, recall_c10, _ = precision_recall_curve(test_df["toxic"], y_scores)

# Отображаем график
disp = PrecisionRecallDisplay(precision=precision_c10, recall=recall_c10)
disp.plot()
plt.show()

z = np.where( prec > 0.95)[0][0]
print(z)
z1 = precision_score(y_true=test_df["toxic"], y_pred=model_pipline_C10.predict_proba(test_df["comment"])[:,1] > thresholds[z])
z2 = recall_score(y_true=test_df["toxic"], y_pred=model_pipline_C10.predict_proba(test_df["comment"])[:,1]> thresholds[z])

print(z1)

print(z2)