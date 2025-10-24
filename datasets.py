import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import os
from utils import transform  # твой transform из utils.py

class PascalVOCDataset(Dataset):
    def __init__(self, data_folder, split):
        """
        data_folder: путь к папке с JSON (например, BCCD_JSON)
        split: 'TRAIN' или 'TEST'
        """
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}

        # Загружаем пути к изображениям и объекты
        with open(os.path.join(data_folder, f'{self.split}_images.json')) as j:
            self.images = json.load(j)

        with open(os.path.join(data_folder, f'{self.split}_objects.json')) as j:
            self.objects = json.load(j)

        # Загружаем label_map
        with open(os.path.join(data_folder, 'label_map.json')) as j:
            self.label_map = json.load(j)

    def __getitem__(self, index):
        # Загружаем изображение
        img_path = self.images[index]
        image = Image.open(img_path).convert('RGB')

        # Загружаем bounding boxes, labels, difficulties
        obj = self.objects[index]
        boxes = torch.FloatTensor(obj['boxes'])         # (n_objects, 4)
        labels = torch.LongTensor(obj['labels'])       # (n_objects)
        difficulties = torch.ByteTensor(obj['difficulties'])  # (n_objects)

        # Применяем аугментации и преобразования
        image, boxes, labels, difficulties = transform(
            image, boxes, labels, difficulties, self.split
        )

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)
