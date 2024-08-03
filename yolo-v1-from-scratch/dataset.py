import torch
import os
import pandas as pd
from PIL import Image
import numpy as np

class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, S=7, B=2, C=7, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.label_files = os.listdir(label_dir)    

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.label_files[index])
        img_path = os.path.join(self.img_dir, self.label_files[index].replace("txt", "jpg"))
        labels = []
        
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                labels.append([class_label, x, y, width, height])

        labels = torch.tensor(labels)

        image = Image.open(img_path)
        if self.transform:
            image, labels = self.transform(image, labels)   
        
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))   
        
        for box in labels:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            if label_matrix[i, j, 7] == 0:
                label_matrix[i, j, 7] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, 8:12] = box_coordinates
                label_matrix[i, j, class_label] = 1 

        image = np.array(image)
        return image, label_matrix
       
    
def test():
    transform = None
    dataset = YOLODataset(
        img_dir="AquariumDataset/train/images",
        label_dir="AquariumDataset/train/labels",
        transform=transform,
    )
    img, target = dataset[0]
    # print(img.shape, target.shape)

    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    for x, y in loader:
        print(x.shape, y.shape)
        break

if __name__ == "__main__":
    test()
