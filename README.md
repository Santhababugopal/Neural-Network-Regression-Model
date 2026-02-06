##EX:01 Neural-Network-Regression-Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

<img width="954" height="633" alt="image" src="https://github.com/user-attachments/assets/8d9ac77c-869d-4d23-a4fe-49621970ff2a" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:SANTHABABU G
### Register Number:212224040292
```

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


dataset1 = pd.read_csv('/content/input_output_table.csv')
X = dataset1[['Input']].values
y = dataset1[['Output']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here since it's a regression task
        return x

ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
print("Name: SANTHABABU  G")
print("Reg no: 212224040292")
train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)
with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')



loss_df = pd.DataFrame(ai_brain.history)

print("Name: SANTHABABU  G")
print("Reg no: 212224040292")
import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

print("Name: SANTHABABU  G")
print("Reg no: 212224040292")
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')

```
## Dataset Information

<img width="600" height="578" alt="image" src="https://github.com/user-attachments/assets/7314a0e3-ef61-4f1e-a2e7-a16f12f34825" />



## OUTPUT

<img width="314" height="448" alt="image" src="https://github.com/user-attachments/assets/82778078-89f8-4dc5-ad76-0ac76a2b60c5" />









<img width="457" height="279" alt="image" src="https://github.com/user-attachments/assets/a3c4e36e-5032-4390-9b9f-a8fb6a754f65" />

### Training Loss Vs Iteration Plot

<img width="821" height="632" alt="image" src="https://github.com/user-attachments/assets/22f6a02b-7b59-405a-b47a-70b4422aa997" />



### New Sample Data Prediction


<img width="968" height="148" alt="image" src="https://github.com/user-attachments/assets/6d885349-2686-487a-97a1-c45c14f8445e" />




##RESULT:

Thus,the code was successfully executed  to develop a neural network regression model...

