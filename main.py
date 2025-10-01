import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

df = pd.read_csv("D:\Downloads\\bank-full.csv", sep=';').drop_duplicates()
categorical_values = df[
    ['Работа', 'Семейный статус', 'Образование', 'Контакт', 'Месяц', 'День недели', 'Доходность']].values.ravel()
unique_values = pd.unique(categorical_values)
df = df.replace({"День": 999,
                 "Работа": "Неизвестно",
                 "Семейный статус": "Неизвестно",
                 "Образование": "Неизвестно",
                 "Месяц": "Неизвестно",
                 "День недели": "Неизвестно",
                 "Доходность": "Неизвестно",
                 }, np.nan)
# print(df.isna().mean()*100)
df.drop(columns=['День'], inplace=True)

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
categorical_columns = ["Работа", "Семейный статус", "Образование", "Месяц", "День недели", "Доходность"]
bool_columns = ["Кредитный дефолт", "Ипотека", "Займ"]
df[categorical_columns] = imputer.fit_transform(df[categorical_columns].values)
df[bool_columns] = imputer.fit_transform(df[bool_columns].values)
imputer_num = SimpleImputer(missing_values=np.nan, strategy='mean')
num_columns = ["Возраст", "Длительность", "Кампания", 'Предыдущий контакт', "Колебание уровня безработицы",
                "Индекс потребительских цен", "Индекс потребительской уверенности",
                "Европейская межбанковская ставка", "Количество сотрудников в компании"]
df[num_columns] = imputer_num.fit_transform(df[num_columns].values)

# print(df.boxplot(column='Возраст', figsize=(5, 10)))
# print(df.boxplot(column='Длительность', figsize=(5, 10)))
# print(df.boxplot(column='Кампания', figsize=(5, 10)))
# print(df.boxplot(column='Предыдущий контакт', figsize=(5, 10)))
# print(df.boxplot(column='Колебание уровня безработицы', figsize=(2, 7)))
# print(df.boxplot(column='Индекс потребительских цен', figsize=(5, 10)))
# print(df.boxplot(column='Индекс потребительской уверенности', figsize=(5, 10)))
# print(df.boxplot(column='Европейская межбанковская ставка', figsize=(5, 10)))
# print(df.boxplot(column='Количество сотрудников в компании', figsize=(5, 10)))
# plt.show()
q = df["Возраст"].quantile(0.99)
q2 = df["Длительность"].quantile(0.99)
q3 = df["Кампания"].quantile(0.99)
q4 = df["Предыдущий контакт"].quantile(0.99)
q5 = df["Колебание уровня безработицы"].quantile(0.99)
q6 = df["Индекс потребительских цен"].quantile(0.99)
q7 = df["Индекс потребительской уверенности"].quantile(0.99)
q8 = df["Европейская межбанковская ставка"].quantile(0.99)
q9 = df["Количество сотрудников в компании"].quantile(0.99)
df = df[df["Возраст"] < q]
df = df[df["Длительность"] < q2]
df = df[df["Кампания"] < q3]
df = df[df["Предыдущий контакт"] < q4]
df = df[df["Колебание уровня безработицы"] < q5]
df = df[df["Индекс потребительских цен"] < q6]
df = df[df["Индекс потребительской уверенности"] < q7]
df = df[df["Европейская межбанковская ставка"] < q8]
df = df[df["Количество сотрудников в компании"] < q9]

df["День недели"] = df["День недели"].replace({"Понедельник": "1", "Вторник": "2", "Среда": "3",
                                               "Четверг": "4", "Пятница": "5"})
X = df[['Возраст', 'Длительность', 'Кампания', 'День недели', 'Предыдущий контакт', 'Индекс потребительских цен',
        'Европейская межбанковская ставка', 'Количество сотрудников в компании']]
y = df.iloc[:, -1]
fit = SelectKBest(score_func=chi2, k='all').fit(X, y)
dfScores = pd.DataFrame(fit.scores_)
dfColumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfColumns, dfScores], axis=1)
featureScores.columns = ['Признак', 'Важность']
# print(featureScores.nlargest(10, 'Важность'))

categoricalColumns = ['Работа', 'Семейный статус', 'Образование', 'Контакт', 'Месяц', 'День недели', 'Доходность', 'y']
df[categoricalColumns] = df[categoricalColumns].astype('category')
df[categoricalColumns] = df[categoricalColumns].apply(lambda x: x.cat.codes)
booleanColumns = ['Кредитный дефолт', 'Ипотека', 'Займ']
df[booleanColumns] = df[booleanColumns].astype('category')
df[booleanColumns] = df[booleanColumns].apply(lambda x: x.cat.codes)

X = df[df.columns[:-1]]
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25, shuffle=True)
clf = DecisionTreeClassifier(max_depth=3, random_state=25)
clf.fit(X_train, y_train)
clf_pred = clf.predict(X_test)
cv_clf = cross_val_score(clf, X_train, y_train, cv=10).mean()
print('Доля правильных ответов: %.3f' % clf.score(X_test, y_test))
print('Доля правильных ответов во время кросс-валидации: %0.3f' % cv_clf)
print('Точность результата измерений: %.3f' % precision_score(y_test, clf_pred))
print('Полнота: %.3f' % recall_score(y_test, clf_pred))
print('Оценка F1: %.3f' % f1_score(y_test, clf_pred))

