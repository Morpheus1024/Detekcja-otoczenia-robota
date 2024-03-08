import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import cv2
import os
 
# # load the dataset, split into input (X) and output (y) variables
# dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
# X = dataset[:,0:8]
# y = dataset[:,8]

dataset_path = '/home/mikolaj/Github/Detekcja-otoczenia-robota/dataset/butelki/all'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

dataset = os.listdir(dataset_path)

X = []
y = []

for file in dataset:
    img = cv2.imread(os.path.join(dataset_path, file))

    label = os.path.splitext(file)[0][-1]  # Extract the last character from the file name
    y.append(int(label))  #wydobywanie etykiety z nazwy pliku: 1-jest butelka, 0-nie ma butelki

    img = cv2.resize(img, (320, 320))  
    img = img[:,:,0:3] / 255.0  # Normalize the image data to [0, 1] range  
    img = torch.from_numpy(img).permute(2, 0, 1)  # Convert the image to PyTorch tensor and change dimensions order to CxHxW
    X.append(img)


# Convert the list of file names into a tensor
# X = torch.tensor(dataset, dtype=torch.float32)
# Convert the list of images into a tensor
X = torch.stack([torch.from_numpy(np.array(i)) for i in X]).float()
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# define the model
model = nn.Sequential(
    nn.Linear(3, 32),
    nn.ReLU(),
    nn.Linear(32, 20),
    nn.ReLU(),
    nn.Linear(20, 25),
    nn.ReLU(),
    nn.Linear(25, 8),
    nn.Sigmoid(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)
# print(model)
model = model.to(device)

# train the model
loss_fn   = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
n_epochs = 100

batch_size = 1

ilosc_zdjec = X.shape[0]
kolor_zdjecia = X.shape[1]
h = X.shape[2]
l = X.shape[3]

# trzeba dostosować wielkości tensorów w predykcji


for epoch in range(n_epochs):
    for i in range(0, h, batch_size):
        for j in range(0, l, batch_size):
            Xbatch = X[i:i+batch_size].to(device)
            Xbatch = Xbatch.view(batch_size, kolor_zdjecia, h, l)  # Reshape the input to match the model's input size
            Xbatch = Xbatch.permute(0, 2, 3, 1)  # Change the dimensions order to match the model's input
            y_pred = model(Xbatch)  # Pass the reshaped input to the model
            ybatch = y[i:i+batch_size].view(-1, 1).to(device)  # Reshape the target tensor
            ybatch = ybatch.view(-1)  # Reshape the target tensor to match the shape of y_pred
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')
 
# Save the trained model
torch.save(model.state_dict(), '/home/mikolaj/Github/Detekcja-otoczenia-robota/model.pth')
print("Model saved successfully.")