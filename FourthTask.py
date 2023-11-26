import tkinter as tk
from tkinter import filedialog, messagebox, Text
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class FourthTask:
    def __init__(self, master):
        self.master = master
        master.title("Изучение возможностей методов обучения без учителя Shirshov Igor")

        # Добавляем виджет Text для вывода результатов
        self.text_result = Text(master, height=30, width=60)
        self.text_result.pack()

        self.train_button = tk.Button(master, text="Сгенерировать данные KMeans", command=self.generate_kmeans_data)
        self.train_button.pack()

        self.train_button = tk.Button(master, text="Обучить KMeans", command=self.train_kmeans)
        self.train_button.pack()

    def generate_kmeans_data(self):
        # Генерация данных
        data = []
        for _ in range(500):
            x1 = np.random.uniform(0, 10)
            x2 = np.random.uniform(0, 10)
            data.append([x1, x2])

        # Сохранение данных в файл
        with open("kmeans_data.txt", "w") as file:
            for row in data:
                file.write(f"{row[0]:.2f},{row[1]:.2f}\n")

    def train_kmeans(self):
        try:
            X = self.read_data()

            # Определение оптимального числа кластеров с использованием метода локтя
            num_clusters = self.elbow_method(X)

            # Обучение модели KMeans
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            labels = kmeans.fit_predict(X)

            # Вывод результатов в виджет Text
            result_text = f"Оптимальное число кластеров: {num_clusters}\n"
            result_text += f"Метки кластеров:\n{labels}"
            self.text_result.delete(1.0, tk.END)  # Очистка текущего содержимого
            self.text_result.insert(tk.END, result_text)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")

    def read_data(self):
        try:
            with open("kmeans_data.txt", 'r') as file:
                lines = file.readlines()

            X = np.array([list(map(float, line.strip().split(','))) for line in lines])
            return X

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при чтении файла: {str(e)}")


    def elbow_method(self, X, max_clusters=10):
        distortions = []
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(X)
            distortions.append(kmeans.inertia_)

        # Определение оптимального числа кластеров с использованием метода локтя
        optimal_clusters = np.argmin(np.diff(distortions)) + 1

        return optimal_clusters

if __name__ == '__main__':
    root = tk.Tk()
    app = FourthTask(root)
    root.mainloop()