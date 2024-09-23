import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Load the Iris dataset
iris_data = load_iris()
X = iris_data.data
y = iris_data.target
labels = iris_data.target_names
features = iris_data.feature_names

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

#Creat PyTorch model
class IrisModel(nn.Module):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = IrisModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr =0.01)

num_epochs = 120

# Lists to store losses
train_losses = []
val_losses = []

#Train the model
for epoch in range(num_epochs):

    #Model.train:  Sets the model to training mode, enabling features like dropout and batch normalization.
    model.train()

    #Forward pass:  Passes the training data through the model to get predictions.
    outputs = model(X_train_tensor) 

    #Calculate loss: Computes the loss by comparing model predictions to the actual labels.
    loss = criterion(outputs, y_train_tensor)

    #Optimizer zero grad: Clears the gradients of all optimized tensors to prevent accumulation from previous iterations.
    optimizer.zero_grad()

     #Loss backward: Computes the gradient of the loss with respect to model parameters through backpropagation.
    loss.backward()

    #Optimizer step:  Updates the model parameters based on the computed gradients.
    optimizer.step()

    # Store training loss
    train_losses.append(loss.item())

    # Validate the model
    model.eval()
    with torch.inference_mode():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)
        val_losses.append(val_loss.item())

    #Display training info
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


#Test the model
model.eval()
with torch.inference_mode():

    # Forward pass
    y_pred = model(X_test_tensor)

    # Retrieves the predicted class labels by taking the index of the maximum logit for each sample
    _, predicted = torch.max(y_pred, 1)


# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, predicted)
print("Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_test, predicted)
print("Classification Report:\n", class_report)

accuracy = accuracy_score(y_test, predicted)
print(f"Accuracy: {accuracy:.2f}")

# Plot learning curves
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Learning Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
