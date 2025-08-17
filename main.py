import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim
from PIL import Image

class DEWM_Guesser(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(640*480,3072), #1% of number of pixles in 480p image
            nn.ReLU(),
            nn.Linear(3072,3072),
            nn.ReLU(),
            nn.Linear(3072,24), #2 images per DE/WM, which there will be 12 of
        )

    def forward(self,image):
        image = self.flatten(image)
        lin_relu_stk = self.linear_relu_stack(image)
        return lin_relu_stk

    def train_model(self, training_data, device):
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), 1e-3)

        for epoch in range(10):
            for images, class_labels in training_data:
                images = images.to(device)
                class_labels = class_labels.to(device)

                outputs = self(images)
                loss = criterion(outputs, class_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")

    def predict(self, images, device):
        self.to(device)
        self.eval()
        images = images.to(device)

        with torch.no_grad():
            outputs = self(images)
            pred = outputs.argmax(1)
        
        return pred.item()

class Data_Giver():
    @staticmethod
    def get(directory):
        transform = transforms.Compose([
            transforms.Resize((480,640)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        training_data = datasets.ImageFolder(directory, transform=transform)

        loader = DataLoader(training_data, batch_size=1, shuffle=True)

        return loader
    
    @staticmethod
    def get_img_from_user():
        image_path = input("Enter the full path to the image you want me to guess: \n")

        transform = transforms.Compose([
            transforms.Resize((480,640)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        
        image = Image.open(image_path)
        image = transform(image)
        image = image.unsqueeze(0)

        return image
    
def main():
    directory = input("What directory shall I use as training data:\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DEWM_Guesser().to(device)

    training_data = Data_Giver().get(directory)

    model.train_model(training_data, device)

    pred_index = model.predict(Data_Giver().get_img_from_user(), device)
    pred_name = training_data.dataset.classes[pred_index]

    print(f"Thats probably {pred_name}")

if __name__=="__main__":
    main()
