import tkinter as tk
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, OrdinalEncoder, KBinsDiscretizer, Binarizer
import numpy as np

class FirstTask:
    def __init__(self, master):
        self.master = master
        master.title("DataPreprocessing Shirshov Igor")
        self.label = tk.Label(master, text="Предобработка данных")
        self.label.pack()

        # Создание виджетов
        self.text_result = tk.Text(master, height=40, width=80)
        self.text_result.pack()

        self.button_process_data = tk.Button(master, text="FirstSet", command=self.process_data)
        self.button_process_data.pack()

    def process_data(self):
        # Задание 1: Произвольный набор данных 5x5
        dataSetOne = np.random.rand(5, 5)
        self.display_result("Задание 1: Произвольный набор данных 5x5", dataSetOne)

        # Задание 2: Стандартизация набора данных
        scaler = StandardScaler()
        standartDataSetOne = scaler.fit_transform(dataSetOne)
        self.display_result("\nЗадание 2: Стандартизация набора данных", standartDataSetOne)

        # Задание 3: Задание и стандартизация аналогичного набора данных
        dataSetTwo = np.random.rand(5, 5)
        standartDataSetTwo = scaler.transform(dataSetTwo)
        self.display_result("\nЗадание 3: Задание и стандартизация аналогичного набора данных", standartDataSetTwo)

        # Задание 4: Стандартизация набора 1 альтернативными способами
        min_max_scaler = MinMaxScaler()
        dataSetOneMinMax = min_max_scaler.fit_transform(dataSetOne)
        max_abs_scaler = MaxAbsScaler()
        dataSetOneAbs = max_abs_scaler.fit_transform(dataSetOne)
        self.display_result("\nЗадание 4 (Мин-Макс): Стандартизация набора 1 альтернативными способами", dataSetOneMinMax)
        self.display_result("\nЗадание 4 (Макс-абс): Стандартизация набора 1 альтернативными способами", dataSetOneAbs)

        # Задание 5: Нормализация данных набора 1 двумя способами
        data_normalized_l1 = preprocessing.normalize(dataSetOne, norm='l1')
        data_normalized_l2 = preprocessing.normalize(dataSetOne, norm='l2')
        self.display_result("\nЗадание 5 (L1): Нормализация данных набора 1 двумя способами", data_normalized_l1)
        self.display_result("\nЗадание 5 (L2): Нормализация данных набора 1 двумя способами", data_normalized_l2)

        # Задание 6: Кодирование данных набора 3
        dataSetThree = np.array([['а', 4], ['е', 6], ['b', 8], ['c', 10], ['d', 12]])
        encoder = OrdinalEncoder()
        dataSetThreeEncoded = encoder.fit_transform(dataSetThree)
        self.display_result("\nЗадание 6: Кодирование данных набора 3", dataSetThreeEncoded)

        # Задание 7: Дискретизация набора
        dataSetTwoDiscret = []
        for n_bins in range(2, 6):
            for strategy in ['uniform', 'quantile', 'kmeans']:
                discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
                dataSetTwoDiscret.append(discretizer.fit_transform(dataSetTwo))
                self.display_result(f"\nЗадание 7 Дискретизация набора (n_bins={n_bins}, strategy={strategy}):", dataSetTwoDiscret)

        # Задание 8: Бинаризация набора
        threshold_values = [0.3, 0.5, 0.7]
        for i, threshold in enumerate(threshold_values):
            dataSetOneBinarized = Binarizer(threshold=threshold).fit_transform(dataSetOne)
            self.display_result(f"\nЗадание 8  Бинаризация набора (Эксперимент {i + 2}, threshold={threshold}):", dataSetOneBinarized)

    def display_result(self, task_title, result):
        self.text_result.insert(tk.END, task_title + "\n")
        self.text_result.insert(tk.END, str(result) + "\n\n")
        self.text_result.yview(tk.END)

if __name__ == '__main__':
    root = tk.Tk()
    app = FirstTask(root)
    root.mainloop()