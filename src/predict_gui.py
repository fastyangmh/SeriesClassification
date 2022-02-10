# import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.predict_gui import BasePredictGUI
from src.predict import Predict
import pandas as pd
from DeepLearningTemplate.data_preparation import parse_transforms
import tkinter as tk
from tkinter import Label, Entry, messagebox


# class
class PredictGUI(BasePredictGUI):
    def __init__(self, project_parameters) -> None:
        super().__init__(extensions=None)
        self.predictor = Predict(project_parameters=project_parameters)
        self.classes = project_parameters.classes
        self.loader = pd.read_csv
        self.transform = parse_transforms(
            transforms_config=project_parameters.transforms_config)['predict']

        # label
        self.input_label = Label(
            master=self.window,
            text='enter your data here. (note, please use comma as delimiter)')

        # entry
        self.input_entry = Entry(master=self.window)

    def recognize(self):
        if self.input_entry.get() != '':
            filepath = []
            for v in self.input_entry.get().split(','):
                try:
                    v = float(v)
                    filepath.append(v)
                except:
                    messagebox.showerror(
                        title='Error!',
                        message='the input data exist non-digital!\ndata: {}'.
                        format(v))
            self.filepath = filepath
            predicted = self.predictor.predict(filepath=self.filepath)
            text = ''
            for idx, (c, p) in enumerate(zip(self.classes, predicted)):
                text += '{}: {}, '.format(c, p.round(3))
                if (idx + 1) < len(self.classes) and (idx + 1) % 5 == 0:
                    text += '\n'
            # remove last commas and space
            text = text[:-2]
            self.predicted_label.config(text='probability:\n{}'.format(text))
            self.result_label.config(text=self.classes[predicted.argmax(-1)])
        else:
            messagebox.showerror(title='Error!', message='please input data!')

    def run(self):
        # NW
        self.recognize_button.pack(anchor=tk.NW)

        # N
        self.input_label.pack(anchor=tk.N)
        self.input_entry.pack(anchor=tk.N)
        self.predicted_label.pack(anchor=tk.N)
        self.result_label.pack(anchor=tk.N)

        # run
        super().run()


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # launch prediction gui
    PredictGUI(project_parameters=project_parameters).run()
