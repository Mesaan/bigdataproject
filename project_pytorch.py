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
from torch.utils.data import DataLoader



import warnings
warnings.filterwarnings('ignore')
sns.set_theme()

raw_data = pd.read_csv('healthcare-dataset-stroke-data.csv')
dataset = raw_data.drop(columns=['ever_married','Residence_type'])

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
dataset.head(10)

#Split into input and output
datainput = dataset.iloc[:,0:9]
dataoutput = dataset.iloc[:,9]
dataoutput.head(1000)
input_train, input_test, output_train, output_test = train_test_split(datainput, dataoutput, test_size=0.1, random_state=42)

#Definition of NN
class StrokePredict(nn.Module):
  def __init__(self):
    super(StrokePredict, self).__init__()
    self.fc1 = nn.Linear(input_size, 64)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(64, 1)
    self.sigmoid = nn.Sigmoid()
  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.sigmoid(x)
    return x

#Definition of Dataset
class StrokeDataset(torch.utils.data.Dataset):
    def __init__(self, datainput, dataoutput):
        self.datainput = datainput
        self.dataoutput = dataoutput

    def __len__(self):
        return len(self.datainput)

    def __getitem__(self, idx):
        return torch.Tensor(self.datainput[idx]), torch.Tensor([self.dataoutput[idx]])

# Load dataset and create data loaders
train_dataset = StrokeDataset(input_train, output_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Hyperparameters
batch_size = 64
input_size = len(datainput[0])
learning_rate = 0.01
num_epochs = 10

# Initialize the model, loss function, and optimizer
model = StrokePredict()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# Training loop
for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    optimizer.zero_grad() # Zero the gradients
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward() # Backpropagation
    optimizer.step() # Update weights

    #Make predictions with model
    predict = model(input_test)
    predict_proba = predict.item()
    
    threshold = 0.5
    predict_label = 1 if predicted_probability > threshold else 0

    
    print(f"Predicted Probability: {predict_proba}")
    print(f"Predicted Label (0: No Stroke, 1: Stroke): {predict_label}")

