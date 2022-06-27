# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 22:02:50 2022

@author: User
"""

import tensorflow as tf
import keras 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import os
import datetime
from keras.callbacks import EarlyStopping, TensorBoard
from keras import regularizers
#Importing Garment Worker Productivity Data
filepath = r"C:\Users\User\Downloads\garments_worker_productivity.csv"
garment_data = pd.read_csv(filepath)
garment_data.head(10)
#Finding total unique value for quarter, department and day columns
print(f"Unique value of {garment_data.columns[1]}:{len(np.unique(garment_data.quarter))}")
print(f"Unique value of {garment_data.columns[2]}:{len(np.unique(garment_data.department))}")
print(f"Unique value of {garment_data.columns[3]}:{len(np.unique(garment_data.day))}")


#Print the value of the unique values for quarter, department and day columns
print(garment_data.department.unique())
print(garment_data.quarter.unique())
print(garment_data.day.unique())

#Replacing the errors in department column
garment_data['department'] = garment_data['department'].replace(['sweing'],['sewing'])
garment_data['department'] = [string.strip() for string in garment_data.department]

print(garment_data.department.unique())

#print the total null values for every column
print(garment_data.isna().sum())

garment_data['wip'] = garment_data['wip'].replace(np.nan, 0)

print(garment_data.isna().sum())
print(garment_data.shape)


#creating new column of month and year for data visualization later
garment_data['date'] = pd.to_datetime(garment_data['date'])

garment_data[['month']] = pd.DataFrame(garment_data.date.dt.month)
# data[['day']] = pd.DataFrame(data.date.dt.day)
#garment_data[['year']] = pd.DataFrame(garment_data.date.dt.year)
#Convert over_time column from using minutes into hours
garment_data['over_time'] = garment_data['over_time'] / 60

garment_data['over_time']

garment_data['margin'] = garment_data['actual_productivity'] - garment_data['targeted_productivity']

garment_data['margin']

garment_data['quarter'] = [int(q[7:]) for q in garment_data.quarter]

garment_data['quarter']

print(garment_data.quarter.unique())
#print the data type for every column
print(garment_data.dtypes)


#Print first 10 rows updated garment_data 
print(garment_data.head(10))

#To visualize the productivity of every teams in term of margin, no of workers, over time and incentive
import seaborn as sns
c = list(np.full(12, 'grey'))
c[5], c[11] = 'orange', 'orange'

f, ax = plt.subplots(2, 2, figsize=(9,6))

sns.barplot(data=garment_data, x='team', y='margin', palette=c, ax=ax[0][0])
ax[0][0].set_title('Productivity', size=15)
sns.barplot(data=garment_data, x='team', y='no_of_workers', palette=c, ax=ax[0][1])
ax[0][1].set_title('Manpower', size=15)
sns.barplot(data=garment_data, x='team', y='over_time', palette=c, ax=ax[1][0])
ax[1][0].set_title('Overtime [hours]', size=15)
sns.barplot(data=garment_data, x='team', y='incentive', palette=c, ax=ax[1][1])
ax[1][1].set_title('Incentive [BDT]', size=15)

plt.tight_layout()

#Pie chart to determnine the total workers in sewing and finishing department
y = np.array([len(garment_data[garment_data['department'] == 'sewing']),
             len(garment_data[garment_data['department'] == 'finishing'])])
mylabels = garment_data.department.unique()
myexplode = [0.2, 0]

plt.pie(y, labels = mylabels, explode = myexplode, shadow = True,startangle=90, autopct='%1.1f%%')
plt.show() 

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))
 
# set height of bar
sewing = [garment_data[(garment_data["department"] == "sewing")&(garment_data["month"] == 1)]["no_of_workers"].sum(),
          garment_data[(garment_data["department"] == "sewing")&(garment_data["month"] == 2)]["no_of_workers"].sum(),
          garment_data[(garment_data["department"] == "sewing")&(garment_data["month"] == 3)]["no_of_workers"].sum()]

finishing = [garment_data[(garment_data["department"] == "finishing")&(garment_data["month"] == 1)]["no_of_workers"].sum(),
             garment_data[(garment_data["department"] == "finishing")&(garment_data["month"] == 2)]["no_of_workers"].sum(),
             garment_data[(garment_data["department"] == "finishing")&(garment_data["month"] == 3)]["no_of_workers"].sum()]
 
# Set position of bar on X axis
br1 = np.arange(len(sewing))
br2 = [x + barWidth for x in br1]
 
# Make the plot
plt.bar(br1, sewing, color ='b', width = barWidth,
        edgecolor ='grey', label ='Sewing')
plt.bar(br2, finishing, color ='y', width = barWidth,
        edgecolor ='grey', label ='Finishing')
 
# Adding Xticks
plt.xlabel('Department', fontweight ='bold', fontsize = 15)
plt.ylabel('Total Workers', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(sewing))],
        ['January', 'February', 'March'])
 
plt.legend()
plt.show()

#To visualize the correlation of the data to determine the degree to which two variables are related
def corr_heatmap(df):    
    plt.figure(figsize=(8,8))

    mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
    sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, cmap='icefire')

corr_heatmap(garment_data)
garment_data.corr()

# Change incentive with values of 0 with very small value 0.0001
garment_data.loc[garment_data.incentive==0, 'incentive'] = 0.0001
garment_data.corr()
# Feature synthesis
features = ['smv', 'wip', 'incentive', 'no_of_workers', 'over_time', 'no_of_style_change']
garment_data_copy = garment_data.copy()

for f in features:
    # Log
    garment_data_copy[f+'_log'] = garment_data_copy[f].apply(lambda x: np.log10(x) if x>0 else np.log10(x+0.0001))
    # Sqrt
    garment_data_copy[f+'_sqrt'] = garment_data_copy[f].apply(lambda x: np.sqrt(x))
    # Cube root
    garment_data_copy[f+'_cbrt'] = garment_data_copy[f].apply(lambda x: np.cbrt(x))
    # Square
    garment_data_copy[f+'^2'] = garment_data_copy[f].apply(lambda x: x**2)
    # Cube
    garment_data_copy[f+'^3'] = garment_data_copy[f].apply(lambda x: x**3)
    for f2 in features:
        if f==f2:
            pass
        else:
            x, y = garment_data_copy[f], garment_data_copy[f2]
            # Multiply
            garment_data_copy[f+'*'+f2] = x*y
            # Divide
            y = y.apply(lambda c: c if c>0 else c+0.0001)
            garment_data_copy[f+'/'+f2] = x/y
            # Take log of multiplication
            xx = garment_data_copy[f+'*'+f2]
            garment_data_copy[f+'*'+f2+'_log'] = xx.apply(lambda x: np.log10(x) if x>0 else np.log10(x+0.0001))

with pd.option_context('display.max_rows', None): 
    print(garment_data_copy.corr().actual_productivity.apply(lambda x: np.abs(x)).sort_values(ascending=False).head(50))

# Feature engineer smv, manpower, and incentive 
garment_data['smv_manpower'] = np.log(garment_data['smv'] / garment_data['no_of_workers'])
garment_data['log_incentive'] = np.log(garment_data['incentive'])
garment_data.corr()
garment_reg = garment_data.drop(columns=['smv', 'no_of_workers', 'incentive', 'idle_time', 'idle_men',
                       'targeted_productivity', 'margin', 'team', 'date'])

# Feature and target
features = garment_reg.drop(columns=['actual_productivity'])
label = garment_reg.actual_productivity

# Encoding
le = LabelEncoder()
categ = ['department', 'day']
features[categ] = features[categ].apply(le.fit_transform)

# Train test split
features_train, features_test, label_train, label_test = train_test_split(features, label, 
                                                                           test_size=0.2, 
                                                                           random_state=12345)
standardizer = StandardScaler()

features_train = standardizer.fit_transform(features_train)
features_test = standardizer.transform(features_test)
print(features_train)

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu',kernel_regularizer=regularizers.L2(0.001)),
    #keras.layers.Dropout(0.25),
    keras.layers.Dense(64, activation='relu',kernel_regularizer=regularizers.L2(0.001)),
    #keras.layers.Dropout(0.25),
    keras.layers.Dense(32, activation='relu',kernel_regularizer=regularizers.L2(0.001)),
    #keras.layers.Dropout(0.5),
    keras.layers.Dense(1),
])
model.compile(optimizer='adam',loss='mse',metrics=['mae'])
history = model.fit(features_train, label_train, validation_data=(features_test, label_test),
                    batch_size=32, epochs=20)
# new instances where we do not know the answer
from sklearn.datasets import make_blobs
features_new, _ = make_blobs(n_samples=10, centers=2, n_features=9, random_state=1)
# make a prediction
label_new = model.predict(features_new)
# show the inputs and predicted outputs
for i in range(len(features_new)):
	print("Features=%s, Predicted=%s" % (features_new[i], label_new[i]))