from tkinter import messagebox

import numpy as np
import tkinter as tk

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class ThirdTask:
    def __init__(self, master):
        self.master = master
        master.title("Изучение возможностей ансамблевого обучения Shirshov Igor")

        # Создание виджетов
        self.text_result = tk.Text(master, height=20, width=80)
        self.text_result.pack()

        self.generate_button = tk.Button(master, text="Сгенерировать для линейного", command=self.generate_linear_regression_data)
        self.generate_button.pack()

        self.generate_button = tk.Button(master, text="Создать линейный классификатор", command=self.train_model)
        self.generate_button.pack()

        self.generate_button = tk.Button(master, text="Сгенерировать для многомерного",
                                         command=self.generate_multivariate_regression_data)
        self.generate_button.pack()

        self.generate_button = tk.Button(master, text="Создать многомерный классификатор",
                                         command=self.train_model_multivariate)
        self.generate_button.pack()

        self.generate_button = tk.Button(master, text="Сгенерировать для дерева",
                                         command=self.generate_tree_data)
        self.generate_button.pack()

        self.generate_button = tk.Button(master, text="Создать классификатор на основе деревьев решений",
                                         command=self.train_model_tree)
        self.generate_button.pack()

        self.generate_button = tk.Button(master, text="Создать классификатор на основе случайного леса",
                                         command=self.train_random_forest)
        self.generate_button.pack()
        self.generate_button = tk.Button(master,
                                         text="Создать классификатор на основе предельно случайного леса",
                                         command=self.train_extra_trees )
        self.generate_button.pack()

    def generate_linear_regression_data(self):
        try:
            low = -5
            high = 5
            # Генерация данных
            data = np.random.uniform(low=low, high=high, size=(500, 2))

            # Сохранение данных в файл
            with open("linear_regression_data.txt", "w") as file:
                for row in data:
                    file.write(f"{row[0]:.2f},{row[1]:.2f}\n")
            messagebox.showinfo("Генерация данных", "Данные успешно сгенерированы.")
        except ValueError:
            messagebox.showerror("Ошибка", "Пожалуйста, введите корректные значения.")

    def train_model(self):
        try:
            X, y = self.read_data()
            # Разделение данных на тренировочный и тестовый наборы
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Создание и обучение линейной регрессионной модели
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Предсказание на тестовом наборе
            y_pred = model.predict(X_test)

            # Оценка качества модели
            mse = mean_squared_error(y_test, y_pred)

            # Вывод результатов в виджет Text
            result_text = f"Среднеквадратичная ошибка: {mse}"
            self.text_result.delete(1.0, tk.END)
            self.text_result.insert(tk.END, result_text)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")

    def read_data(self):
        try:
            with open("linear_regression_data.txt", 'r') as file:
                lines = file.readlines()

            data = [line.strip().split(',') for line in lines]
            X = np.array([list(map(float, record[:-1])) for record in data])
            y = np.array([float(record[-1]) for record in data])
            return X,y

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при чтении файла: {str(e)}")

    def generate_multivariate_regression_data(self):
        low = -5
        high = 5
        # Генерация данных
        data = np.random.uniform(low=low, high=high, size=(500, 4))

        # Сохранение данных в файл
        with open("multivariate_regression_data.txt", "w") as file:
            for row in data:
                file.write(f"{row[0]:.2f},{row[1]:.2f},{row[2]:.2f},{row[3]:.2f}\n")

    def read_multivariate_regression_data(self):
        try:
            with open("multivariate_regression_data.txt", 'r') as file:
                lines = file.readlines()

            data = [list(map(float, line.strip().split(','))) for line in lines]
            X = np.array([row[:-1] for row in data])
            y = np.array([row[-1] for row in data])
            return X,y
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при чтении файла: {str(e)}")
    def train_model_multivariate(self):

            X, y = self.read_multivariate_regression_data()
            # Разделение данных на тренировочный и тестовый наборы
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Создание и обучение многомерной регрессионной модели
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Предсказание на тестовом наборе
            y_pred = model.predict(X_test)

            # Оценка качества модели
            mse = mean_squared_error(y_test, y_pred)

            # Вывод результатов в виджет Text
            result_text = f"Среднеквадратичная ошибка (MSE): {mse}"
            self.text_result.delete(1.0, tk.END)  # Очистка текущего содержимого
            self.text_result.insert(tk.END, result_text)

    def generate_tree_data(self):
        # Генерация данных
        data = []
        for _ in range(500):
            x1 = np.random.uniform(1, 10)
            x2 = np.random.uniform(1, 10)
            label = np.random.choice([0, 1])  # случайный выбор метки класса (0 или 1)
            data.append([x1, x2, label])

        # Сохранение данных в файл
        with open("tree_data.txt", "w") as file:
            for row in data:
                file.write(f"{row[0]:.2f},{row[1]:.2f},{int(row[2])}\n")

    def train_model_tree(self):
        try:
            X,y = self.read_data_tree()

            # Разделение данных на тренировочный и тестовый наборы
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Создание и обучение классификатора на основе дерева решений
            clf = DecisionTreeClassifier(random_state=42)
            clf.fit(X_train, y_train)

            # Предсказание на тестовом наборе
            y_pred = clf.predict(X_test)

            # Оценка качества модели
            accuracy = accuracy_score(y_test, y_pred)
            confusion_mat = confusion_matrix(y_test, y_pred)

            # Вывод результатов в виджет Text
            result_text = f"Точность (Accuracy): {accuracy:.2f}\n"
            result_text += f"Матрица ошибок:\n{confusion_mat}"
            self.text_result.delete(1.0, tk.END)
            self.text_result.insert(tk.END, result_text)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")

    def read_data_tree(self):
        try:
            with open("tree_data.txt", 'r') as file:
                lines = file.readlines()

            data = [list(map(float, line.strip().split(','))) for line in lines]
            X = np.array([row[:-1] for row in data])
            y = np.array([int(row[-1]) for row in data])
            return X, y
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при чтении файла: {str(e)}")

    def train_random_forest(self):
        self.train_classifier(RandomForestClassifier(), "Random Forest")

    def train_extra_trees(self):
        self.train_classifier(ExtraTreesClassifier(), "Extra Trees")

    def train_classifier(self, classifier, classifier_name):
        try:
            X,y = self.read_data_tree()

            # Разделение данных на тренировочный и тестовый наборы
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Обучение классификатора
            classifier.fit(X_train, y_train)

            # Предсказание на тестовом наборе
            y_pred = classifier.predict(X_test)

            # Оценка качества модели
            accuracy = accuracy_score(y_test, y_pred)
            confusion_mat = confusion_matrix(y_test, y_pred)

            # Вывод результатов в виджет Text
            result_text = f"Точность ({classifier_name}): {accuracy:.2f}\n"
            result_text += f"Матрица ошибок:\n{confusion_mat}"
            self.text_result.delete(1.0, tk.END)  # Очистка текущего содержимого
            self.text_result.insert(tk.END, result_text)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")

if __name__ == '__main__':
    root = tk.Tk()
    app = ThirdTask(root)
    root.mainloop()