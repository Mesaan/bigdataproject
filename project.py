#%%
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt #visualize
import seaborn as sns #also visualize
#NN imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

#Definition of model, fully connected NN
model = Sequential()
#Hidden layers
model.add(Dense(16, input_shape=(9,), activation='relu'))
model.add(Dense(8, activation='relu'))
#Output layer
model.add(Dense(1, activation='sigmoid'))
#Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#Train model
model.fit(datainput, dataoutput, epochs=25, batch_size=10)


#Evaluate model
_, accuracy = model.evaluate(datainput, dataoutput)
print('Accuracy: %.2f' % (accuracy*100))

#todo : try to make predictions
#todo : try to separate input into training & testing