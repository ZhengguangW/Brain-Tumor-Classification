from PIL import Image
import os
# Set the path to the folders containing the images and labels

class ReadImages():
    ##"/Users/zhengguang/Desktop/OneDrive - University of Virginia/Desktop/CS 4774/Brain-Tumoer-Classification/archive"
    def __init__(self,parent_folder_path):
        self.parent_folder_path = parent_folder_path
        self.images = []
        self.labels = []
        self.label_map = {"glioma_tumor": 0, "meningioma_tumor": 1, "no_tumor": 2, "pituitary_tumor": 3}
    # Loop through each subfolder and read in the images and labels
    def Reading(self):
        for subfolder in os.listdir(self.parent_folder_path):
            if subfolder.startswith('.'):
                continue  
            if not os.path.isdir(os.path.join(self.parent_folder_path, subfolder)):
                continue
            subfolder_path = os.path.join(self.parent_folder_path, subfolder)
            for folder in os.listdir(subfolder_path):
                if folder.startswith('.'):
                    continue  
                if not os.path.isdir(os.path.join(subfolder_path, folder)):
                    continue
                folder_path = os.path.join(subfolder_path, folder)
                for file in os.listdir(folder_path):
                    if file.startswith('.'):
                        continue  # skip hidden files and folders
                    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                        img = Image.open(os.path.join(folder_path, file))
                        self.images.append(img)
                        label_str = folder
                        label = self.label_map[label_str]
                        self.labels.append(label)



