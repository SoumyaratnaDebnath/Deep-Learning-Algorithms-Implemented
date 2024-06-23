from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class HorseZebraDataset(Dataset):
    def __init__(self, root_zebra, root_horse, transform=None):
        self.root_zebra = root_zebra
        self.root_horse = root_horse
        self.transform = transform

        self.zebra_images = os.listdir(root_zebra)
        self.horse_images = os.listdir(root_horse)
        self.length_dataset = max(len(self.zebra_images), len(self.horse_images)) # 1000, 1500
        self.zebra_len = len(self.zebra_images)
        self.horse_len = len(self.horse_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        zebra_img = self.zebra_images[index % self.zebra_len]
        horse_img = self.horse_images[index % self.horse_len]

        zebra_path = os.path.join(self.root_zebra, zebra_img)
        horse_path = os.path.join(self.root_horse, horse_img)

        zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
        horse_img = np.array(Image.open(horse_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=zebra_img, image0=horse_img)
            zebra_img = augmentations["image"]
            horse_img = augmentations["image0"]

        return zebra_img, horse_img





    

import matplotlib.pyplot as plt
import numpy as np

def test():
    map_dataset = HorseZebraDataset("./data/train/horses", "./data/train/zebras")
    print(f"Number of samples in the dataset: {len(map_dataset)}")
    
    # Retrieve the first input-output pair
    inp, outp = map_dataset[10]
    
    # Convert to NumPy if inp and outp are not already NumPy arrays
    if not isinstance(inp, np.ndarray):
        inp = inp.numpy()
    if not isinstance(outp, np.ndarray):
        outp = outp.numpy()
    
    # Transpose the image data if it's in channel-first format
    if inp.shape[0] == 3:  # Assuming the shape is (3, H, W)
        inp = np.transpose(inp, (1, 2, 0))
    if outp.shape[0] == 3:  # Assuming the shape is (3, H, W)
        outp = np.transpose(outp, (1, 2, 0))
    
    # Plot the images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(inp)
    axes[0].set_title("Horse")
    axes[0].axis("off")
    
    axes[1].imshow(outp)
    axes[1].set_title("Zeebra")
    axes[1].axis("off")
    
    plt.show()
    print(inp.shape, outp.shape)

# Example usage
if __name__ == "__main__":
    test()
