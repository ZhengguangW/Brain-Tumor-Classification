from PIL import Image
import os
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
# Set the path to the folders containing the images and labels

class ReadImages():
    #"/Users/zhengguang/Desktop/OneDrive - University of Virginia/Desktop/CS 4774/Brain-Tumoer-Classification/archive"
    def __init__(self,parent_folder_path):
        self.parent_folder_path = parent_folder_path
        self.images = []
        self.labels = []
        self.label_map = {"glioma_tumor": 0, "meningioma_tumor": 1, "no_tumor": 2, "pituitary_tumor": 3}
        self.training_images=None
        self.training_labels=None
        self.testing_images=None
        self.testing_labels=None
        self.validation_images=None
        self.validation_labels=None

    # Loop through each subfolder and read in the images and labels
    def Reading(self):
        pca = PCA(n_components=128)
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
                        img = img.convert("L")
                        img=img.resize((380,360))
                        img_array = np.array(img)
                        self.images.append(img_array)  # add the image array to the list
                        label_str = folder
                        label = self.label_map[label_str]
                        self.labels.append(label)
        self.images=np.array(self.images)
        pca = PCA(n_components=128)
        self.images = pca.fit_transform(self.images.reshape(-1, 360*380))
        self.images = self.images.reshape(-1, 128)
        


    def Split(self):
        self.training_images, X_test_val, self.training_labels, y_test_val = train_test_split(self.images, self.labels, test_size=0.2, random_state=42)
        self.testing_images,self.validation_images,self.testing_labels,self.validation_labels=train_test_split(
            X_test_val, y_test_val, test_size=0.1, random_state=42)
        
    def get_training(self):
        return self.training_images,self.training_labels
    
    def get_testing(self):
        return self.testing_images,self.testing_labels
    
    def get_validation(self):
        return self.validation_images,self.validation_labels
