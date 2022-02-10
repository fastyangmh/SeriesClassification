# import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.predict import SeriesPredictDataset
from src.model import create_model
import torch
from DeepLearningTemplate.data_preparation import parse_transforms
import pandas as pd
from typing import Any
from os.path import isfile
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


#class
class Predict:
    def __init__(self, project_parameters) -> None:
        self.model = create_model(project_parameters=project_parameters).eval()
        if project_parameters.device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.transform = parse_transforms(
            transforms_config=project_parameters.transforms_config)['predict']
        self.device = project_parameters.device
        self.batch_size = project_parameters.batch_size
        self.num_workers = project_parameters.num_workers
        self.classes = project_parameters.classes
        self.loader = pd.read_csv

    def predict(self, filepath) -> Any:
        result = []
        if type(filepath) == list:
            sample = torch.tensor(filepath).float()
            if self.transform is not None:
                sample = self.transform(sample)
            sample = sample[None]
            if self.device == 'cuda' and torch.cuda.is_available():
                sample = sample.cuda()
            with torch.no_grad():
                result.append(self.model(sample).tolist()[0])
        elif isfile(path=filepath):
            dataset = SeriesPredictDataset(filepath=filepath,
                                           loader=self.loader,
                                           transform=self.transform)
            pin_memory = True if self.device == 'cuda' and torch.cuda.is_available(
            ) else False
            data_loader = DataLoader(dataset=dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=self.num_workers,
                                     pin_memory=pin_memory)
            with torch.no_grad():
                for sample in tqdm(data_loader):
                    if self.device == 'cuda' and torch.cuda.is_available():
                        sample = sample.cuda()
                    result.append(self.model(sample).tolist())
        else:
            assert 0, 'please check the filepath.\nfilepath: {}'.format(
                filepath)
        result = np.concatenate(result, 0)
        print(', '.join(self.classes))
        print(result)
        return result


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # predict file
    result = Predict(project_parameters=project_parameters).predict(
        filepath=project_parameters.root)
