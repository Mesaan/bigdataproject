#Project
#%%
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt #visualize
import seaborn as sns #also visualize
#NN imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
sns.set_theme()

raw_data = pd.read_csv('healthcare-dataset-stroke-data.csv')
dataset = raw_data.drop(columns=['ever_married','Residence_type'])
dataset = dataset.dropna()

#Inspect data
dataset.head(10)
dataset.columns
dataset.describe().T
dataset.info()

#Adjust data type for string ones
dataset['gender'] = dataset['gender'].astype('string')
dataset['work_type'] = dataset['work_type'].astype('string')
dataset['smoking_status'] = dataset['smoking_status'].astype('string')
dataset.info()

#Sort out columns
columns=[i.lower() for i in dataset.columns]
dataset.columns=columns

#Add weight status group variable to dataset for datavis
dataset["weight"]=["underweight" if each <18.5 else "normal" if (each>18.5 and each<24.9) else "overweight" if (each>25 and each<29.9) else "obese" if (each>30) else "nan" for each in dataset.bmi]

#Histogram datavisualisation of ppl w strokes
stroke_data = dataset[dataset['stroke'] == 1]
def plot_hist(variable):
    plt.figure(figsize=(9,3))
    plt.hist(stroke_data[variable],bins=50,density=True)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f"{variable} histogram distribution")
    plt.show()
numerical=["age","bmi","weight"]

for i in numerical:
    plot_hist(i)

#Convert the data to numerical to use it in NN
dataset = dataset.drop(columns=['weight'])
dataset_nb = dataset[['gender','work_type','smoking_status']]
dataset_nb = dataset_nb.astype('category')
dataset_nb = dataset_nb.apply(lambda x : x.cat.codes)
dataset[dataset_nb.columns] = dataset_nb.copy()

#Split into input and output
datainput = dataset.drop(columns=['stroke']) #all but stroke
dataoutput = dataset['stroke']

#Convert into tensors
datainput = torch.tensor(datainput.values, dtype=torch.float32)
dataoutput = torch.tensor(dataoutput.values, dtype=torch.float32)

# Split into training and testing
dataset = TensorDataset(datainput, dataoutput)
train_len = int(0.8 * len(dataset))
test_len = len(dataset) - train_len
train_set, test_set = random_split(dataset, [train_len, test_len])


#Definition of NN
class StrokePredict(nn.Module):
  def __init__(self, input_size):
      super(StrokePredict, self).__init__()
      self.fc1 = nn.Linear(input_size, 32)
      self.relu1 = nn.ReLU()
      self.fc2 = nn.Linear(32, 1)
      self.sigmoid = nn.Sigmoid()

  def forward(self, x):
      x = self.fc1(x)
      x = self.relu1(x)
      x = self.fc2(x)
      x = self.sigmoid(x)
      return x

#Parameters
num_epochs = 10
batch_size = 32
learning_rate=0.001

#Dataset loaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

#Initialize  model, loss function, and optimizer
model = StrokePredict(input_size=datainput.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Training loop
for epoch in range(num_epochs):
    for datainput, dataoutput in train_loader:
        optimizer.zero_grad()
        output = model(datainput)
        loss = criterion(output, dataoutput.view(-1, 1))
        loss.backward()
        optimizer.step()

    # Printing loss after each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

#Evaluation loop
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for testinput, testoutput in test_loader:
        output = model(testinput)
        predicted = (output > 0.5).float()
        total += testoutput.size(0)
        correct += (predicted == testoutput.view(-1, 1)).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy}')