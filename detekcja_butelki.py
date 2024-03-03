import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import os
 
# # load the dataset, split into input (X) and output (y) variables
# dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
# X = dataset[:,0:8]
# y = dataset[:,8]

dataset_path = '/home/mikolaj/Github/Detekcja-otoczenia-robota/dataset/butelki/all'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

dataset = os.listdir(dataset_path)

X = []
y=[]

for file in dataset:
    img = cv2.imread(os.path.join(dataset_path, file))
    img = cv2.resize(img, (320, 320))  # Resize the image to 64x64 pixels
    X.append(img)
    file = os.path.splitext(file)[0]
    y.append(float(file[-1]))


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
print(model)
model = model.to(device)
# train the model
loss_fn   = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
n_epochs = 100
batch_size = 1

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size].to(device)
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size].view(-1, 1).to(device)
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')
 
# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")