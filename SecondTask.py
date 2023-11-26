import tkinter as tk
import random
from tkinter import messagebox

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

filename = 'naive_bayes_data.txt'

class SecondTask:
    def __init__(self, master):
        self.master = master
        master.title("training with teacher Shirshov Igor")

        # Создание виджетов
        self.text_result = tk.Text(master, height=20, width=80)
        self.text_result.pack()

        self.generate_button = tk.Button(master, text="Сгенерировать для баейсовского", command=self.generate_data)
        self.generate_button.pack()

        self.generate_button = tk.Button(master, text="Создать наивный баейсовский классификатор", command=self.train_naive_bayes_classifier)
        self.generate_button.pack()

        self.generate_button = tk.Button(master, text="Сгенерировать для svm", command=self.generate_svm_data)
        self.generate_button.pack()

        self.generate_button = tk.Button(master, text="Создать классификатор на основе машины опорных векторов",command=self.classify_data)
        self.generate_button.pack()
    def generate_data(self):
        try:
            np.random.seed(42)
            data = np.zeros((500, 3))

            # Генерация первого столбца (признак 1)
            data[:, 0] = np.random.uniform(low=0, high=10, size=500)

            # Генерация второго столбца (признак 2)
            data[:, 1] = np.random.uniform(low=-1, high=6, size=500)

            # Генерация третьего столбца (метка класса)
            data[:, 2] = np.random.randint(0, 4, size=500)

            # Сохранение данных в файл
            np.savetxt(filename, data, delimiter=',', fmt='%.2f,%.2f,%d')

            messagebox.showinfo("Генерация данных", "Данные успешно сгенерированы.")
        except ValueError:
            messagebox.showerror("Ошибка", "Пожалуйста, введите корректные значения.")

    def train_naive_bayes_classifier(self):
        # Чтение данных из файла
        data = np.loadtxt(filename, delimiter=',')
        X = data[:, :-1]  # Входные признаки
        y = data[:, -1]  # Выходные метки

        # Разделение данных на тренировочный и тестовый наборы
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Создание и обучение наивного байесовского классификатора
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)

        # Предсказание на тестовом наборе
        y_pred = classifier.predict(X_test)

        # Оценка качества классификатора
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        # Вывод результатов
        self.text_result.delete(1.0, tk.END)
        self.text_result.insert(tk.END, f"Точность: {accuracy}\n\nОтчет о классификации:\n{classification_rep}")

    def generate_svm_data(file_path, num_records=500):
        # Списки для возможных значений
        workclass_options = ["State-gov", "Self-emp-not-inc", "Private", "Federal-gov", "Local-gov", "Never-worked",
                             "Without-pay"]
        education_options = ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm",
                             "Assoc-voc", "7th-8th", "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
                             "Preschool", "12th"]
        marital_status_options = ["Never-married", "Married-civ-spouse", "Divorced", "Married-spouse-absent",
                                  "Separated", "Married-AF-spouse", "Widowed"]
        occupation_options = ["Adm-clerical", "Exec-managerial", "Handlers-cleaners", "Prof-specialty", "Other-service",
                              "Sales", "Craft-repair", "Transport-moving", "Farming-fishing", "Machine-op-inspct",
                              "Tech-support", "Protective-serv", "Armed-Forces", "Priv-house-serv"]
        relationship_options = ["Not-in-family", "Husband", "Wife", "Own-child", "Unmarried", "Other-relative"]
        race_options = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
        sex_options = ["Male", "Female"]
        native_country_options = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                                  "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba",
                                  "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico",
                                  "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan",
                                  "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand",
                                  "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]
        income_options = ["<=50K", ">50K"]

        with open("svm_data.txt", "w") as file:
            for _ in range(num_records):
                record = [
                    random.randint(18, 65),
                    random.choice(workclass_options),
                    random.randint(10000, 99999),
                    random.choice(education_options),
                    random.randint(1, 16),
                    random.choice(marital_status_options),
                    random.choice(occupation_options),
                    random.choice(relationship_options),
                    random.choice(race_options),
                    random.choice(sex_options),
                    random.randint(0, 9999),
                    random.randint(0, 99),
                    random.randint(20, 60),
                    random.choice(native_country_options),
                    random.choice(income_options)
                ]
                record_str = ', '.join(map(str, record))
                file.write(record_str + '\n')
        messagebox.showinfo("Генерация данных", "Данные успешно сгенерированы.")

    def classify_data(self):
        # Чтение данных из файла
        file_path = "svm_data.txt"
        X, y = self.read_data(file_path)

        # Разделение данных на тренировочный и тестовый наборы
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Создание и обучение классификатора SVM
        classifier = SVC()
        classifier.fit(X_train, y_train)

        # Предсказание на тестовом наборе
        y_pred = classifier.predict(X_test)

        # Оценка качества классификатора
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        # Вывод результатов в виджет Text
        result_text = f"Точность: {accuracy}\n\nОтчет о классификации:\n{classification_rep}"
        self.text_result.delete(1.0, tk.END)  # Очистка текущего содержимого
        self.text_result.insert(tk.END, result_text)

    def read_data(self, file_path):
        X = []
        y = []
        count_class1 = 0
        count_class2 = 0
        max_datapoints = 13000

        with open(file_path, 'r') as f:
            for line in f.readlines():
                if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
                    break
                if '?' in line:
                    continue

                data = line[:-1].split(', ')

                if data[-1] == '<=50K' and count_class1 < max_datapoints:
                    X.append(data)
                    count_class1 += 1
                if data[-1] == '>50K' and count_class2 < max_datapoints:
                    X.append(data)
                    count_class2 += 1

        X = np.array(X)

        # Преобразование строковых данных в числовые
        label_encoder = []
        X_encoded = np.empty(X.shape)
        for i, item in enumerate(X[0]):
            if item.isdigit():
                X_encoded[:, i] = X[:, i]
            else:
                label_encoder.append(preprocessing.LabelEncoder())
                X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

        X = X_encoded[:, :-1].astype(int)
        y = X_encoded[:, -1].astype(int)
        return X,y

if __name__ == '__main__':
    root = tk.Tk()
    app = SecondTask(root)
    root.mainloop()