from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import config

class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(root_dir)
        self.list_files.sort()
    
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :600, :] # image is (3, 1200, 600)
        target_image = image[:, 600:, :]

        argumentations = config.both_trasform(image=input_image, image0 = target_image)
        input_image, target_image = argumentations["image"], argumentations["image0"] 

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_output(image=target_image)["image"]

        return input_image, target_image
    

import matplotlib.pyplot as plt
import numpy as np

def test():
    map_dataset = MapDataset("./map_dataset/train")
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
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    axes[1].imshow(outp)
    axes[1].set_title("Output Image")
    axes[1].axis("off")
    
    plt.show()

# Example usage
if __name__ == "__main__":
    test()
