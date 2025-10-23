import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET

class BCCDDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: путь к папке BCCD_Dataset
        transform: torchvision.transforms для изображений
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = sorted(os.listdir(os.path.join(root_dir, "images")))
        self.annots = sorted(os.listdir(os.path.join(root_dir, "annotations")))

        # карта классов
        self.class_dict = {"RBC": 1, "WBC": 2, "Platelets": 3}  # 0 зарезервирован под фон

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # загрузка изображения
        img_path = os.path.join(self.root_dir, "images", self.images[idx])
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # загрузка аннотации
        annot_path = os.path.join(self.root_dir, "annotations", self.annots[idx])
        boxes = []
        labels = []
        tree = ET.parse(annot_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            label = obj.find("name").text
            labels.append(self.class_dict[label])
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text) / width
            ymin = float(bndbox.find("ymin").text) / height
            xmax = float(bndbox.find("xmax").text) / width
            ymax = float(bndbox.find("ymax").text) / height
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        target = {"boxes": boxes, "labels": labels}

        return image, target
