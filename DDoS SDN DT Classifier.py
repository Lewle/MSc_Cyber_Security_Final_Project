#!/usr/bin/env python
# coding: utf-8

# # K-Nearest Neighbour Classifier

# In[15]:


# Numpy and Pandas
import numpy as np                    # Support with specialized data structures, functions 
import pandas as pd      
# Used to read csv, excel, txt,..., 
import seaborn as sns                 #  A data visualization library

# Ignore Warnings
import warnings
warnings.filterwarnings ('ignore')


# For Plotting 
get_ipython().system('pip install missingno')
get_ipython().system('pip install matplotlib')
import missingno as msno              # Missing data visualizations
import matplotlib
from matplotlib import pyplot as plt  # Multi-platform data visualization library built on Num
get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.display import set_matplotlib_formats  
set_matplotlib_formats ('retina')     # Setting display format to retina for better quality images

from matplotlib.pyplot import figure
warnings.filterwarnings('ignore')
from sklearn import preprocessing

# Default Parameter for plots
matplotlib.rcParams['font.size']= 11
matplotlib.rcParams['figure.titlesize']= 15
matplotlib.rcParams['figure.figsize']= [10,8]

# For Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Oversampling and Undersampling
get_ipython().system('pip install imbalanced-learn')
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler



# Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Modelling and Performance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
get_ipython().system('pip install xgboost')
#from xgboost import xgb
from sklearn.cluster import KMeans
from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

from datetime import datetime
import time


# For Merging .csv files
import glob                            # Importing glob library to read and support merging multiple csv files


# # Exploratory Data Analysis (EDA)

# ###  Read .csv Files

# In[73]:


# Read .csv file

SDN = pd.read_csv('Dataset_sdn.csv')
SDN.head()


# In[74]:


# Read .csv file

#SDN = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
#SDN.head()


# In[75]:


[col for col in SDN.columns] # lets just look at the column titles


# In[76]:


# Check the unique labels in the dataset

print(SDN['label'].unique())


# In[77]:


SDN.shape


# # Data Cleaning

# ### Missing Values

# In[78]:


# Checking number of missing (null) values

SDN_null_count = SDN.isnull().sum()
SDN_null_count


# In[79]:


# Visualisation of missing values in a bar chart
msno.bar(SDN)


# In[80]:


# Visualisation of the location of missing values
msno.matrix(SDN)


# In[81]:


# How many total missing values do we have?
total_cells = np.product(SDN.shape)
total_missing_values = SDN_null_count.sum()
print ('SDN_total_cells',total_cells)
print ('SDN_total_missing_values',total_missing_values)


# In[82]:


# Percent of data that is missing in SDN dataframe
percent_missing = (total_missing_values/total_cells) * 100
print('SDN_percent_missing_values',percent_missing)


# In[83]:


# Percent of data that is missing in each SDN feature

SDN_nulls_percentage = SDN.isnull().sum()/len(SDN)*100

print('the percentages of null values per feature in SDN:\n')
print(round(SDN_nulls_percentage,2))


# In[84]:


# Dropping missing values

print('Dataframe Shape',SDN.shape)

SDN.dropna(inplace=True)

print ('Dataframe Shape after dropping missing values',SDN.shape)


# In[85]:


# Visualisation of dataframe after dropping missing values in a bar chart
msno.bar(SDN)


# ### Duplicated Values

# In[86]:


# Count number of duplicated rows, except the first occurrences

len(SDN[SDN.duplicated(keep='first')])


# In[87]:


# Visualised duplicated rows, except the first occurrences
SDN_duplicateRows = SDN[SDN.duplicated(subset=None, keep='first')]
SDN_duplicateRows


# In[88]:


# Drop only the repeated rows (duplicated rows after the first occurrences)

print ('SDN after dropping missing values',SDN.shape)

SDN.drop_duplicates(inplace = True)

#  Display after removing null values & duplicated rows after the first occurrences

print ('SDN after dropping missing values & duplicated rows',SDN.shape)


# ### Low Variance / Single Value Features

# In[89]:


# Drop columns with constant value (std = 0)


single_value_list = []
for cols in (SDN.select_dtypes(include=['number'])):                     # Select only the 'number' columns
    if (SDN[cols].std())==0:                                             # Calculate std of those 'number' columns = std with 0 value
        single_value_list.append(cols)
                        
    
print('Columns with single value:\n',np.array(single_value_list),'\n')

print ('SDN Shape after dropping missing values & duplicated rows',SDN.shape)

SDN.drop(single_value_list,axis=1,inplace=True) 

print ('SDN Shape after dropping missing values, duplicated rows & low variance rows',SDN.shape)


# In[68]:


# Numeric Features's Structure
SDN_numerical_features = [feature for feature in SDN.columns if SDN[feature].dtypes != 'O']
SDN_unique_numerical_values = SDN[SDN_numerical_features].nunique(axis=0).sort_values()

# Set the color palette
sns.set_palette("pastel")

# Plot information with y-axis in log-scale
SDN_unique_numerical_values.plot.bar(logy=True, title="Unique values per feature in SDN");


# In[90]:


# Select columns with less than 5 unique values

#num_features_less_than_5_unique = [feature for feature in SDN.columns if SDN[feature].dtypes != 'O' and SDN[feature].nunique() < 5]
#num_features_less_than_5_unique


# In[91]:


# Drop columns with less than 5 unique values

print ('SDN Shape after dropping missing values, duplicated rows & low variance rows',SDN.shape)

#SDN.drop(num_features_less_than_5_unique,axis=1,inplace=True) 

print ('SDN Shape after dropping missing values, duplicated rows, low variance rows & constant value columns',SDN.shape)


# ### Identifying and Handling Outliers

# In[95]:


# Identify numeric columns from the SDN dataset
numeric_columns = SDN.select_dtypes(include=[np.number])

# Set the color palette
sns.set_palette("coolwarm")

# Plotting box plots for each numeric column
plt.figure()  

for col in numeric_columns.columns:
    plt.subplot(2, 2, 2)  # Adjust the number of rows and columns for subplots
    sns.boxplot(data=SDN[col])
    plt.title(f'Box Plot of {col}') 
    plt.show()


# In[93]:


# Set the color palette
sns.set_palette("coolwarm")

# Numeric Feature Distribution via Histrograms

cols = 3
rows = 10
num_cols = SDN.select_dtypes(exclude='object').columns
fig = plt.figure( figsize=(cols*5, rows*5))
for i, col in enumerate(num_cols):
    
    ax=fig.add_subplot(rows,cols,i+1)
    
    sns.histplot(x = SDN[col], ax = ax)
    
fig.tight_layout()  
plt.show()


# In[60]:


#CIC2017D.plot(lw=0,
          #marker=".",
          #subplots=True,
          #layout=(-1, 4),
          #figsize=(15, 30),
          #markersize=1);


# In[61]:


#sns.pairplot(SDN, hue='label')


# ### Converting Categorical Features to Numerical Features

# In[96]:


# Check data types
SDN.info()


# In[97]:


# Set the color palette
sns.set_palette("coolwarm")

plt.bar(list(dict(SDN.Protocol.value_counts()).keys()), dict(SDN.Protocol.value_counts()).values())
plt.bar(list(dict(SDN[SDN.label == 1].Protocol.value_counts()).keys()), dict(SDN[SDN.label == 1].Protocol.value_counts()).values())

plt.xlabel('Protocol')
plt.ylabel('Count')
plt.legend(['All', 'Malicious'])
plt.title('The number of requests from different protocols')


# In[98]:


# Set the color palette
sns.set_palette("bwr")

plt.barh(list(dict(SDN.src.value_counts()).keys()), dict(SDN.src.value_counts()).values())
plt.barh(list(dict(SDN[SDN.label == 1].src.value_counts()).keys()), dict(SDN[SDN.label == 1].src.value_counts()).values())

for idx, val in enumerate(dict(SDN.src.value_counts()).values()):
    plt.text(x = val, y = idx-0.2, s = str(val), color='b')

for idx, val in enumerate(dict(SDN[SDN.label == 1].src.value_counts()).values()):
    plt.text(x = val, y = idx-0.2, s = str(val), color='black')


plt.xlabel('Number of Requests')
plt.ylabel('Source IP Address')
plt.legend(['All','Malicious'])
plt.title('Number of requests from different IP adress')


# In[100]:


plt.barh(list(dict(SDN.dst.value_counts()).keys()), dict(SDN.dst.value_counts()).values(), color='yellow')
plt.barh(list(dict(SDN[SDN.label == 1].dst.value_counts()).keys()), dict(SDN[SDN.label == 1].dst.value_counts()).values(), color='orange')

for idx, val in enumerate(dict(SDN.dst.value_counts()).values()):
    plt.text(x = val, y = idx-0.2, s = str(val), color='green')

for idx, val in enumerate(dict(SDN[SDN.label == 1].dst.value_counts()).values()):
    plt.text(x = val, y = idx-0.2, s = str(val), color='b')


plt.xlabel('Number of Requests')
plt.ylabel('Destination IP addres')
plt.legend(['All','Malicious'])
plt.title('Number of requests to different IP adress')


# In[101]:


# Dataset for Label Encoding

SDN_LE = SDN.copy()


# In[102]:


# Creating a instance of label Encoder 
LE = LabelEncoder()
 
# Using .fit_transform function to fit label to return encoded label

SDN_src = LE.fit_transform(SDN['src'])
SDN_dst = LE.fit_transform(SDN['dst'])
SDN_pt = LE.fit_transform(SDN['Protocol'])


# In[103]:


#  Dropping 'object' features from the DataFrame

SDN.drop("dst", axis=1, inplace=True)
SDN.drop("src", axis=1, inplace=True)
SDN.drop("Protocol", axis=1, inplace=True)


# In[104]:


# Appending the array back to the DataFrame

SDN["dst"] = SDN_dst
SDN["src"] = SDN_src
SDN["Protocol"] = SDN_pt


# In[105]:


print('SDN datatypes after LabelEncoding:\n')
SDN.info()


# In[106]:


# Set the color palette
sns.set_palette("coolwarm")

plt.bar(list(dict(SDN_LE.Protocol.value_counts()).keys()), dict(SDN_LE.Protocol.value_counts()).values())
plt.bar(list(dict(SDN_LE[SDN_LE.label == 1].Protocol.value_counts()).keys()), dict(SDN_LE[SDN_LE.label == 1].Protocol.value_counts()).values())

plt.xlabel('Protocol')
plt.ylabel('Count')
plt.legend(['All', 'Malicious'])
plt.title('The number of requests from different protocols after Label Encoding')


# In[99]:


#print ('CIC2017D Shape after dropping missing values, duplicated rows, low variance rows & constant value columns',CIC2017D.shape)
#print ('CIC2017D Shape after dropping missing values, duplicated rows, low variance rows, constant value columns & after Lable Encoding',CIC2017D.shape)


# In[107]:


# Set the color palette
sns.set_palette("bwr")

plt.barh(list(dict(SDN_LE.src.value_counts()).keys()), dict(SDN_LE.src.value_counts()).values())
plt.barh(list(dict(SDN_LE[SDN_LE.label == 1].src.value_counts()).keys()), dict(SDN_LE[SDN_LE.label == 1].src.value_counts()).values())


plt.xlabel('Number of Requests')
plt.ylabel('Source IP Address')
plt.legend(['All','Malicious'])
plt.title('Number of requests from different IP adress after Label Encoding')


# In[108]:


# Set the color palette
sns.set_palette("coolwarm")

plt.barh(list(dict(SDN_LE.dst.value_counts()).keys()), dict(SDN_LE.dst.value_counts()).values(), color='yellow')
plt.barh(list(dict(SDN_LE[SDN_LE.label == 1].dst.value_counts()).keys()), dict(SDN_LE[SDN_LE.label == 1].dst.value_counts()).values(), color='orange')


plt.xlabel('Number of Requests')
plt.ylabel('Destination IP addres')
plt.legend(['All','Malicious'])
plt.title('Number of requests to different IP adress after Label Encoding')


# In[109]:


print ('SDN Shape after dropping missing values, duplicated rows & constant value rows',SDN.shape)
print ('SDN Shape after dropping missing values, duplicated rows & constant value rows & after Lable Encoding',SDN_LE.shape)


# ### Saving the Pre-processed file

# In[110]:


#CIC2017D.to_csv('CIC2017D_preprocessed_data.csv', index=False)


# In[111]:


# Read .csv file

#CIC2017D = pd.read_csv('CIC2017D_preprocessed_data.csv')
#CIC2017D.head()


# In[112]:


# replace infinities with NaN

# CIC2017D.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[113]:


#from scipy.stats.mstats import winsorize

# Apply winsorize to each column
#for col in CIC2017D.columns:
    #CIC2017D[col] = winsorize(CIC2017D[col], limits=[0.05, 0.05])


# ### Handle Inf Values

# In[114]:


# Replace infinite values with NaN

SDN.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[115]:


# Drop rows with NaN 

SDN.dropna(inplace=True)#### Handle Inf Values


# ### Handle str in the Label Column

# In[117]:


# Replace strings with numerical values in the Label column

#SDN[' Label'] = SDN[' Label'].replace({'BENIGN': 0, 'DDoS': 1})


# ### Effect of Scalling on Different ML Model Performance

# #### Performance Evaluation

# In[39]:


# Prepare the data

SDN_ALL_s = SDN.copy() # Std Scalling
SDN_ALL_q = SDN.copy() # QT Scalling

ALLxs = SDN_ALL_s.drop(columns=[' Label'])
ALLys = SDN_ALL_s[' Label']

ALLxq = SDN_ALL_q.drop(columns=[' Label'])
ALLyq = SDN_ALL_q[' Label']


# In[40]:


# Split the data
ALLxs_train, ALLxs_test, ALLys_train, ALLys_test = train_test_split(ALLxs, ALLys, test_size=0.3, random_state=42)
ALLxq_train, ALLxq_test, ALLyq_train, ALLyq_test = train_test_split(ALLxq, ALLyq, test_size=0.3, random_state=42)


# In[41]:


# Scale the Data

from sklearn.preprocessing import StandardScaler, QuantileTransformer

# Scale the Data - Std Scaling
st = StandardScaler()
#ALLxss  = st.fit_transform(ALLxs)

# Fit the scaler only on the training data to avoid data leakage
st.fit(ALLxs_train)


# Transform both training and test data using the fitted scaler
ALLxss_train = st.transform(ALLxs_train)
ALLxss_test = st.transform(ALLxs_test)


# Scale the Data - QT
qt = QuantileTransformer()
#ALLxqs  = qt.fit_transform(ALLxq)

# Fit the scaler only on the training data to avoid data leakage
qt.fit(ALLxq_train)

# Transform both training and test data using the fitted scaler
ALLxqs_train = qt.transform(ALLxq_train)
ALLxqs_test = qt.transform(ALLxq_test)


# In[42]:


#from sklearn.preprocessing import MinMaxScaler
#min_max = MinMaxScaler()
#ALLmx  = min_max.fit(ALLx)


# In[43]:


fig , (ax1,ax2,ax3) = plt.subplots(ncols=3,figsize= (12,5))

ax1.scatter(ALLxs_test['Total Length of Fwd Packets'] , ALLxs_test[' Average Packet Size'] , color = 'blue')
ax1.set_title('Before Scaling')
ax1.set_xlabel('Total Length of Fwd Packets')
ax1.set_ylabel('Average Packet Size')

ax2.scatter(ALLxss_test[:, 4] , ALLxss_test[:, 40] , color = 'red')
ax2.set_title('After Std Scaling')
ax2.set_xlabel('Total Length of Fwd Packets')
ax2.set_ylabel('Average Packet Size')


ax3.scatter(ALLxqs_test[:, 4] , ALLxqs_test[:, 40] , color = 'orange')
ax3.set_title('After QT Scaling')
ax3.set_xlabel('Total Length of Fwd Packets')
ax3.set_ylabel('Average Packet Size')


# In[463]:


fig , (ax1,ax2,ax3) = plt.subplots(ncols=3,figsize= (12,5))

ax1.scatter(ALLxs_test[' Fwd Header Length.1'] , ALLxs_test[' Total Fwd Packets'] , color = 'blue')
ax1.set_title('Before Scaling')
ax1.set_xlabel('Fwd Header Length.1')
ax1.set_ylabel('Total Fwd Packets')

ax2.scatter(ALLxss_test[:, 43] , ALLxss_test[:, 2] , color = 'red')
ax2.set_title('After Std Scaling')
ax2.set_xlabel('Fwd Header Length.1')
ax2.set_ylabel('Total Fwd Packets')


ax3.scatter(ALLxqs_test[:, 43] , ALLxqs_test[:, 2] , color = 'orange')
ax3.set_title('After QT Scaling')
ax3.set_xlabel('Fwd Header Length.1')
ax3.set_ylabel('Total Fwd Packets')



# In[464]:


fig , (ax1,ax2, ax3) = plt.subplots(ncols=3,figsize= (12,5))

ax1.scatter(ALLxs_test[' Packet Length Variance'] , ALLxs_test[' Down/Up Ratio'] , color = 'blue')
ax1.set_title('Before Scaling')
ax1.set_xlabel('Packet Length Variance')
ax1.set_ylabel('Total Fwd Packets')

ax2.scatter(ALLxss_test[:, 4] , ALLxss_test[:, 39] , color = 'red')
ax2.set_title('After Std Scaling')
ax2.set_xlabel('Packet Length Variance')
ax2.set_ylabel('Down/Up Ratio')


ax3.scatter(ALLxqs_test[:, 4] , ALLxqs_test[:, 39] , color = 'orange')
ax3.set_title('After QT Scaling')
ax3.set_xlabel('Packet Length Variance')
ax3.set_ylabel('Down/Up Ratio')


# In[465]:


fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(ncols=6, figsize=(18, 5))

# Before scaling: Packet Length Variance
ax1.set_title('Before Scaling')
sns.kdeplot(ALLx['Total Length of Fwd Packets'], ax=ax1)

# After Std Scaling: Packet Length Variance
ax2.set_title('After Std Scaling')
sns.kdeplot(ALLxss_test[:, 4], ax=ax2, color='red')

# After QT Scaling: Packet Length Variance
ax3.set_title('After QT Scaling')
sns.kdeplot(ALLxqs_test[:, 4], ax=ax3, color='orange')

# Before scaling: Down/Up Ratio
ax4.set_title('Before Scaling')
sns.kdeplot(ALLx[' Average Packet Size'], ax=ax4)

# After Std Scaling: Down/Up Ratio
ax5.set_title('After Std Scaling')
sns.kdeplot(ALLxss_test[:, 40], ax=ax5, color='red')

# After QT Scaling: Down/Up Ratio
ax6.set_title('After QT Scaling')
sns.kdeplot(ALLxqs_test[:, 40], ax=ax6, color='orange')

plt.tight_layout()
plt.show()


# In[466]:


fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(ncols=6, figsize=(18, 5))

# Before scaling: Packet Length Variance
ax1.set_title('Before Scaling')
sns.kdeplot(ALLx[' Packet Length Variance'], ax=ax1)

# After Std Scaling: Packet Length Variance
ax2.set_title('After Std Scaling')
sns.kdeplot(ALLxss_test[:, 38], ax=ax2, color='red')

# After QT Scaling: Packet Length Variance
ax3.set_title('After QT Scaling')
sns.kdeplot(ALLxqs_test[:, 38], ax=ax3, color='orange')

# Before scaling: Down/Up Ratio
ax4.set_title('Before Scaling')
sns.kdeplot(ALLx[' Down/Up Ratio'], ax=ax4)

# After Std Scaling: Down/Up Ratio
ax5.set_title('After Std Scaling')
sns.kdeplot(ALLxss_test[:, 39], ax=ax5, color='red')

# After QT Scaling: Down/Up Ratio
ax6.set_title('After QT Scaling')
sns.kdeplot(ALLxqs_test[:, 39], ax=ax6, color='orange')

plt.tight_layout()
plt.show()



# #### Effect of StandardScalling on Different Algorithms

# In[44]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import time

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as XGB
from sklearn.neural_network import MLPClassifier

# Models
models = {
    'LR': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(),
    'DT': DecisionTreeClassifier(random_state=42),
    'RF': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'XGB': XGB.XGBClassifier(random_state=42),
    'MLP': MLPClassifier(max_iter=1000, random_state=42)
}

datasets = {
    'Unscaled': (ALLxs_train, ALLys_train, ALLxs_test, ALLys_test),  # Add this line for the unscaled dataset
    'StandardScaler': (ALLxss_train, ALLys_train, ALLxss_test, ALLys_test),
    'QuantileTransformer': (ALLxqs_train, ALLyq_train, ALLxqs_test, ALLyq_test)
}


# In[46]:


from sklearn.metrics import confusion_matrix
import numpy as np

def g_mean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return np.sqrt(sensitivity * specificity)


# Evaluate model
def evaluate_model(model, x_train, y_train, x_test, y_test):
    start_time = time.time()

    # Train model
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    # Evaluate model
    gmean = g_mean(y_test, predictions)  # Compute G-mean
    f1 = f1_score(y_test, predictions, average='macro')
    
    end_time = time.time()
    training_time = end_time - start_time
    
    return f1, gmean, training_time  # Return F1 score, G-mean, and training time


# Store results
results = {}

# Loop through datasets and model evaluation
for model_name, model in models.items():
    for dataset_name, (x_train, y_train, x_test, y_test) in datasets.items():
        f1, gmean, train_time = evaluate_model(model, x_train, y_train, x_test, y_test)  
        results[(model_name, dataset_name)] = {'F1 Score': f1, 'G-mean': gmean, 'Training Time': train_time}  

# Display results
for (model_name, dataset_name), metrics in results.items():
    print(f"Model: {model_name}, Dataset: {dataset_name}")
    print(f"F1 Score: {metrics['F1 Score']:.4f}, G-mean: {metrics['G-mean']:.4f}, Training Time: {metrics['Training Time']:.2f} seconds\n")


# In[3]:


# SDN - Scaling plots

# Define a list of pastel colors
#!pip install --upgrade matplotlib

#colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'purple']

models = ['LR', 'KNN', 'DT', 'RF', 'SVM', 'XGB', 'MLP']
f1_scores = [ 0.966, 0.9974, 0.9998, 0.9999, 0.9339, 0.9999, 0.9899]  
g_means = [0.9632, 0.9973, 0.9998, 0.9999, 0.9317, 0.9999, 0.9904 ]  

plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.scatter(g_means[i], f1_scores[i], label=model, s=100)
plt.xlabel('G- Means')
plt.ylabel('F1 Score')
plt.title('Without Scaling: F1 Score vs G-Means')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


# SDN - Scaling plots

models = ['LR', 'KNN', 'DT', 'RF', 'SVM', 'XGB', 'MLP']
f1_scores = [ 0.9978,  0.9996 , 0.9998,  0.9999 , 0.9985,  0.9999 ,0.9994 ]  
g_means = [ 0.9977, 0.9995, 0.9998, 0.9999, 0.9985, 0.9999, 0.9995]  

plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.scatter(g_means[i], f1_scores[i], label=model, s=100)
plt.xlabel('G- Means')
plt.ylabel('F1 Score')
plt.title('STD Scaling: F1 Score vs G-Means')
plt.legend()
plt.grid(True)


# In[ ]:


# SDN - Scaling plots


models = ['LR', 'KNN', 'DT', 'RF', 'SVM', 'XGB', 'MLP']
f1_scores = [0.9989, 0.9998, 0.9998, 0.9999, 0.99965, 0.9999, 0.9998]  
g_means = [0.9989, 0.9998, 0.9998, 0.9999, 0.999675, 0.9999, 0.9998] 

plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.scatter(g_means[i], f1_scores[i], label=model, s=100)
plt.xlabel('G- Means')
plt.ylabel('F1 Score')
plt.title('QT Scaling: F1 Score vs G-Means')
plt.legend()


# In[ ]:


# SDN - Scaling plots


models = ['LR', 'KNN', 'DT', 'RF', 'SVM', 'XGB', 'MLP']
f1_scores = [ 0.966, 0.9974, 0.9998, 0.9999, 0.9339, 0.9999, 0.9899]   
training_times= [  13.19,  19.94, 1.33, 26.29, 1.526.24,  15.58,  33.35 ]  

plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.scatter(training_times[i], f1_scores[i], label=model, s=100)
plt.xlabel('Training Times')
plt.ylabel('F1 Score')
plt.title('Without Scaling: F1 Score vs Training Times')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


# SDN - Scaling plots


models = ['LR', 'KNN', 'DT', 'RF', 'SVM', 'XGB', 'MLP']
g_means = [ 0.9632, 0.9973, 0.9998, 0.9999, 0.9317, 0.9999, 0.9904 ]  
training_times= [ 13.19,  19.94, 1.33, 26.29, 1.526.24,  15.58,  33.35]  

plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.scatter(training_times[i], g_means[i], label=model, s=100)
plt.xlabel('Training Times')
plt.ylabel('G-Means')
plt.title('Without Scaling: G-Means vs Training Times')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


# SDN - Scaling plots

models = ['LR', 'KNN', 'DT', 'RF', 'SVM', 'XGB', 'MLP']
f1_scores = [ 0.9978 , 0.9996,  0.9998 , 0.9999 , 0.9985 , 0.9999,  0.9994 ]  
training_times= [  4.10,  21.57 , 1.18 , 21.92 , 378.54 , 14.74, 18.17 ]  

plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.scatter(training_times[i], f1_scores[i], label=model, s=100)
plt.xlabel('Training Times')
plt.ylabel('F1 Score')
plt.title('STD Scaling: F1 Score vs Training Times')
plt.legend()
plt.grid(True)


# In[ ]:


# SDN - Scaling plots

models = ['LR', 'KNN', 'DT', 'RF', 'SVM', 'XGB', 'MLP'] 
g_means = [0.9977, 0.9995, 0.9998 , 0.9999 , 0.9985 , 0.9999 ,0.9995 ]  
training_times= [ 4.10,  21.57 , 1.18 , 21.92 , 378.54 , 14.74, 18.17]  

plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.scatter(training_times[i], f1_scores[i], label=model, s=100)
plt.xlabel('Training Times')
plt.ylabel('F1 Score')
plt.title('STD Scaling: G Means vs Training Times')
plt.legend()
plt.grid(True)


# In[ ]:


# SDN - Scaling plots

models = ['LR', 'KNN', 'DT', 'RF', 'SVM', 'XGB', 'MLP']
f1_scores = [0.9989, 0.9998 , 0.9998 , 0.9999 , 0.9997 , 0.9999, 0.9998]  
training_times= [  1.93 , 20.33 , 1.39 , 23.33 , 10.95 , 15.05 , 18.41 ]  

plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.scatter(training_times[i], f1_scores[i], label=model, s=100)
plt.xlabel('Training Times')
plt.ylabel('F1 Score')
plt.title('QT Scaling: F1 Score vs Training Times')
plt.legend()
plt.grid(True)


# In[ ]:


# SDN - Scaling plots

models = ['LR', 'KNN', 'DT', 'RF', 'SVM', 'XGB', 'MLP']
g_means = [0.9989, 0.9998 , 0.9998 , 0.9999 , 0.9997 , 0.9999, 0.9998]  
training_times= [ 1.93 , 20.33 , 1.39 , 23.33 , 10.95 , 15.05 , 18.41 ]  

plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.scatter(training_times[i], g_means[i], label=model, s=100)
plt.xlabel('Training Times')
plt.ylabel('F1 Score')
plt.title('QT Scaling: G Means vs Training Times')
plt.legend()


# ### Feature Selection

# In[118]:


# No Feature Selection

SDN_ALL = SDN.copy()


# #### Correlations

# In[121]:


# Set the color palette
#sns.set_palette("coolwarm")

# Feature Selection via Correlations

SDN_CR = SDN.copy()

CRx = SDN_CR.drop(columns=['label'])  #independent columns
CRy = SDN_CR['label']   #target column


# Identifing Correlations of each features
corre_scores = SDN_CR.corr()
top_corr_features = corre_scores.index
plt.figure(figsize=(18,18))

# Heat Map Plotting
g=sns.heatmap(SDN_CR[top_corr_features].corr(),annot=True, cmap="RdYlGn")


# In[122]:


# Calculate the correlation between each feature and the label

correlations = SDN_CR.corr()["label"].sort_values(ascending=False)


# In[124]:


# Select features between 0 - 0.7 and exclude the target itself from the selected features

select= correlations[(correlations > 0) & (correlations < 0.7)]
TOP_CR_features = select.drop('label', errors='ignore')  # Exclude the target if it's in the list


# In[125]:


# Display the ranked features and their correlation values

print(TOP_CR_features)


# In[128]:


# Create a new dataset with the selected TOP features

SDN_CR_T = SDN_CR[TOP_CR_features.index.tolist() + ['label']]


# In[129]:


# 'NEW DATA' now contains only the top features and the target variable

SDN_CR_T


# In[131]:


print ('SDN Shape after dropping missing values, duplicated rows & constant value rows & after Lable Encoding',SDN.shape)
print ('SDN Shape after dropping missing values, duplicated rows & constant value rows & after Lable Encoding & Correlation Feature Selection',SDN_CR_T.shape)


# #### ANOVA F-value

# In[132]:


# Feature Selection via Information Gain

SDN_AF = SDN.copy()


# In[135]:


from sklearn.feature_selection import f_classif

# Feature Selecting via ANOVA F-value

AFx = SDN_AF.drop(columns=['label'])  #independent columns
AFy = SDN_AF['label']   #target column


# Calculate ANOVA F-values for each feature with respect to the target
f_values, _ = f_classif(AFx, AFy)

# Rank features based on F-values
ranked_features = pd.Series(f_values, index=AFx.columns).sort_values(ascending=False)
ranked_features


# In[137]:


from sklearn.feature_selection import f_classif

# Feature Selecting via ANOVA F-value

AFx = SDN_AF.drop(columns=['label'])  #independent columns
AFy = SDN_AF['label']   #target column

selector = SelectKBest(score_func=f_classif, k='all')  # 'all' to get scores for all features

# Fit the selector to the data
selector.fit(AFx, AFy)

# Get the ANOVA F-values (scores) for all features
f_scores = selector.scores_

# Create a DataFrame to store the scores along with their corresponding feature names
AF_results = pd.DataFrame({'Feature': AFx.columns, 'ANOVA F-Score': f_scores})

# Sort the features based on their ANOVA F-scores in descending order
AF_results = AF_results.sort_values(by='ANOVA F-Score',ascending=False)

print(AF_results)


# In[139]:


from sklearn.model_selection import cross_val_score, StratifiedKFold # due to class imbalance
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest


# Evaluate the performance for different numbers of features


# k-fold cross-validation to determine the optimal number of features

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# n_splits=5: Divided into 5 parts folds. In each iteration of the cross-validation, 4 of these parts will be used for training and 1 for validation
# shuffle=True: This shuffles the data before splitting it into folds - ensure each fold is representative of the overall dataset

max_score = -float('inf')

# max_score = -float('inf):  Negative infinity ensures that any observed score will be greater than max_score in the beginning
# A higher F-value indicates a greater difference between the means

optimal_num_features = 0

# store the number of features that gives the highest average cross-validation score

scores_list = []

# store the average cross-validation score for each number in this list, which use later to plot or analyze how the model's performance varies with the number of features


for n in range(1, len(ranked_features) + 1):
    selected_features = ranked_features.index[:n].tolist()

    # Create a pipeline with feature selection and a classifier
    pipe = make_pipeline(SelectKBest(f_classif, k=n), RandomForestClassifier()) #SelectKBest selects the top n features based on the F-values.
    
    # Cross-validation
    scores = cross_val_score(pipe, AFx, AFy, cv=kf)
    avg_score = scores.mean()
    scores_list.append(avg_score)

    if avg_score > max_score:
        max_score = avg_score
        optimal_num_features = n


# In[140]:


# Plot scores_list to see how accuracy changes with number of features

plt.plot(range(1, len(ranked_features) + 1), scores_list)
plt.xlabel("Number of Features")
plt.ylabel("Cross-Validation Score")
plt.show()


# In[141]:


print(f"Optimal number of features: {optimal_num_features}")
print(f"The best cross-validated score with optimal features is: {max_score:.4f}")


# In[142]:


# Create a DataFrame
AF_scores = pd.DataFrame({
    'Number of Features': range(1, len(ranked_features) + 1),
    'Average Cross-Validation Score': scores_list
})

# Display the DataFrame
print(AF_scores)


# In[144]:


# Create a new dataset with the selected TOP features

TOP_AF_features = ranked_features.index[:optimal_num_features].tolist()
SDN_AF_T = SDN_AF[TOP_AF_features + ['label']]
SDN_AF_T 


# In[145]:


print ('SDN Shape after dropping missing values, duplicated rows & constant value rows',SDN.shape)
print ('SDN Shape after dropping missing values, duplicated rows & constant value rows & after AF Feature Selection',SDN_AF_T.shape)


# #### Information Gain

# In[146]:


# Feature Selection via Information Gain

SDN_IG = SDN.copy()


# In[147]:


# Feature Selecting via IG

IGx = SDN_IG.drop(columns=['label'])  #independent columns
IGy = SDN_IG['label']   #target column

# Calculate IG for each feature with respect to the target
ig_values = mutual_info_classif(IGx, IGy)

# Rank features based on IG values
ranked_features = pd.Series(ig_values, index=IGx.columns).sort_values(ascending=False)
ranked_features


# In[148]:


from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Use cross-validation to determine the optimal number of features

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42) # Using RandomForest as an example classifier

# n_splits=5: Divided into 5 parts folds. In each iteration of the cross-validation, 4 of these parts will be used for training and 1 for validation
# shuffle=True: This shuffles the data before splitting it into folds - ensure each fold is representative of the overall dataset

max_mean_accuracy = -float('inf')

# maximum mean accuracy obtained from the cross-validation runs

optimal_num_features = 0

# store the number of features that gives the highest average cross-validation score

scores_list = []

for n in range(1, len(ranked_features) + 1):
    selected_features = ranked_features.index[:n].tolist()
    scores = cross_val_score(clf, IGx[selected_features], IGy, cv=kf, scoring='accuracy')
    
# Store the result
    mean_accuracy = np.mean(scores) 
    
    scores_list.append(mean_accuracy)  # append mean accuracy to scores_list
    
    if mean_accuracy > max_mean_accuracy:
        max_mean_accuracy = mean_accuracy
        optimal_num_features = n

optimal_features = ranked_features.index[:optimal_num_features].tolist()


# In[149]:


print(f"Optimal number of features: {optimal_num_features}")
print(f"Mean Accuracy with optimal features: {max_mean_accuracy}")


# In[150]:


# Plot scores_list to see how accuracy changes with number of features

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(ranked_features) + 1), scores_list, marker='o', linestyle='-')
plt.title('Mean Accuracy vs. Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Mean Accuracy')
plt.grid(True)
plt.show()


# In[151]:


# Create a DataFrame
IG_scores = pd.DataFrame({
    'Number of Features': range(1, len(ranked_features) + 1),
    'Mean Accuracy with optimal features': scores_list
})

# Display the DataFrame
print(IG_scores)


# In[152]:


# Create a new dataset with the selected optimal features

SDN_IG_T = SDN_IG[optimal_features + ['label']]


# In[153]:


print ('SDN Shape after dropping missing values, duplicated rows & constant value rows',SDN.shape)
print ('SDN Shape after dropping missing values, duplicated rows & constant value rows & after IG Feature Selection',SDN_IG_T.shape)


# #### Lasso (Least Absolute Shrinkage and Selection Operator) 

# In[154]:


SDN_LO = SDN.copy()

LOx = SDN_LO.drop(columns=['label'])  #independent columns
LOy = SDN_LO['label']   #target column


# In[155]:


# Step 1: Scale the features (Lasso regularization is sensitive to the scale of the features)

scaler = StandardScaler()
LOx_s = scaler.fit_transform(LOx)


# In[156]:


# Lasso Logistic Regression
lasso_logistic = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000)  # Using 'saga' as it's suitable for large datasets and supports L1 penalty.
lasso_logistic.fit(LOx_s, LOy)


# In[157]:


# Select important features based on non-zero coefficients

LO_features = LOx.columns[lasso_logistic.coef_[0] != 0]
LO_features


# In[158]:


num_selected_features = len(LO_features)
num_selected_features


# In[159]:


# Create new dataset with only selected features
SDN_LO_T = SDN_LO[LO_features.tolist() + ['label']]


# In[160]:


print ('SDN Shape after dropping missing values, duplicated rows & constant value rows',SDN.shape)
print ('SDN Shape after dropping missing values, duplicated rows & constant value rows & after LO Feature Selection',SDN_LO_T.shape)


# #### Recursive Feature Elimination (RFE)

# In[161]:


# Feature Selection via Recursive Feature Elimination (RFE)

SDN_RF = SDN.copy()

RFx = SDN_RF.drop(columns=['label'])  #independent columns
RFy = SDN_RF['label']   #target column


# In[162]:


from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


# Create a Logistic Regression model as the base estimator for RFE (for classification problem)
estimator = LogisticRegression(max_iter=1000, solver='liblinear')

# Using a step of 5 for RFE to remove 5 features at a time
selector = RFECV(estimator=estimator, step=5, cv=StratifiedKFold(5), scoring='accuracy')

# Extends RFE by adding cross-validation to determine the optimal number of features that yield the best performance, as based on the scoring parameter on the validation set

# Fit RFE
selector = selector.fit(RFx, RFy)


# In[163]:


# Get the optimal number of features
optimal_features_count = selector.n_features_

# Score at optimal number of features
#optimal_score = selector.grid_scores_[optimal_features_count - 1]

print(f"Optimal number of features: {optimal_features_count}")
#print(f"Score at optimal number of features: {optimal_score:.4f}")


# In[164]:


TOP_features = RFx.columns[selector.support_]
TOP_features


# In[165]:


# Plotting the cross-validation scores

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.title('Cross-validation Score as Number of Features Increases')
plt.xlabel('Number of Features Selected')
plt.ylabel('Cross-validation Score (Accuracy)')
plt.axvline(x=optimal_features_count, color='r', linestyle='--', label=f"Optimal number: {optimal_features_count}")
plt.legend()
plt.show()


# In[169]:


# Create new dataset with the selected features
SDN_RF_T = SDN_RF[TOP_features.to_list() + ['label']]
SDN_RF_T


# In[170]:


print ('SDN Shape after dropping missing values, duplicated rows & constant value rows',SDN.shape)
print ('SDN Shape after dropping missing values, duplicated rows & constant value rows & after RFE Feature Selection',SDN_RF_T.shape)


# ### Effect of Feature Selection on Different ML Model Performance

# #### Performance Evaluation

# In[171]:


import pandas as pd
import time
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as XGB
from sklearn.neural_network import MLPClassifier


# Models
models = {
    'LR': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(),
    'DT': DecisionTreeClassifier(random_state=42),
    'RF': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'XGB': XGB.XGBClassifier(random_state=42),
    'MLP': MLPClassifier(max_iter=1000, random_state=42)
}

datasets = {
    'Correlation': (SDN_CR_T.drop(columns=['label']), SDN_CR_T['label']),
    'ANOVA F-value': (SDN_AF_T.drop(columns=['label']), SDN_AF_T['label']),
    'Information Gain': (SDN_IG_T.drop(columns=['label']), SDN_IG_T['label']),
    'Lasso': (SDN_LO_T.drop(columns=['label']), SDN_LO_T['label']),
    'RFE': (SDN_RF_T.drop(columns=['label']), SDN_RF_T['label']),
}


# In[172]:


from sklearn.metrics import confusion_matrix
import numpy as np

def g_mean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return np.sqrt(sensitivity * specificity) #** 0.5


# Evaluate model
def evaluate_model(model, x_data, y_data):
    start_time = time.time()

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

    # Train model
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    
    # Evaluate model
    gmean = g_mean(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')
    
    end_time = time.time()
    training_time = end_time - start_time
    
    return f1, gmean, training_time # Return F1 score, G-mean, and training time
    
    
# Store results
results = {}

# Loop through datasets and models to evaluate

for dataset_name, (x_data, y_data) in datasets.items():
    for model_name, model in models.items():
        f1, gmean, train_time = evaluate_model(model, x_data, y_data)
        results[(dataset_name, model_name)] = {'F1 Score': f1, 'G-mean': gmean, 'Training Time': train_time}
        
        
# Display results
for (dataset_name, model_name), metrics in results.items():
    print(f"Feature Selection: {dataset_name}, Model: {model_name}")
    print(f"F1 Score: {metrics['F1 Score']:.4f}, G-mean: {metrics['G-mean']:.4f}, Training Time: {metrics['Training Time']:.2f} seconds\n")


# In[175]:


# No Feature Selection

import pandas as pd
import time
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as XGB
from sklearn.neural_network import MLPClassifier


# Models
models = {
    'LR': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(),
    'DT': DecisionTreeClassifier(random_state=42),
    'RF': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'XGB': XGB.XGBClassifier(random_state=42),
    'MLP': MLPClassifier(max_iter=1000, random_state=42)
}

datasets = {
    'Wihtout Feature Selection': (SDN.drop(columns=['label']), SDN['label']),
}


# In[176]:


def g_mean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return np.sqrt(sensitivity * specificity) #** 0.5


# Evaluate model
def evaluate_model(model, x_data, y_data):
    start_time = time.time()

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

    # Train model
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    
    # Evaluate model
    gmean = g_mean(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')
    
    end_time = time.time()
    training_time = end_time - start_time
    
    return f1, gmean, training_time # Return F1 score, G-mean, and training time
    
    
# Store results
results = {}

# Loop through datasets and models to evaluate

for dataset_name, (x_data, y_data) in datasets.items():
    for model_name, model in models.items():
        f1, gmean, train_time = evaluate_model(model, x_data, y_data)
        results[(dataset_name, model_name)] = {'F1 Score': f1, 'G-mean': gmean, 'Training Time': train_time}
        
        
# Display results
for (dataset_name, model_name), metrics in results.items():
    print(f"Feature Selection: {dataset_name}, Model: {model_name}")
    print(f"F1 Score: {metrics['F1 Score']:.4f}, G-mean: {metrics['G-mean']:.4f}, Training Time: {metrics['Training Time']:.2f} seconds\n")


# In[116]:


Note:

This code is a template. Ensure you adjust paths, dataset-specific treatments, and hyperparameters accordingly.
Proper hyperparameter tuning can be computationally expensive. Adjust the GridSearchCV or RandomizedSearchCV parameters accordingly based on computational constraints.
You might need to handle class imbalance in more sophisticated ways, such as over-sampling, under-sampling, or using algorithms specifically designed for imbalanced classes.

simport pandas as pd


import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

# Assuming you have loaded the classifiers, defined the g_mean function, and loaded datasets

# 1. Data Preparation
data = pd.read_csv("your_dataset_path.csv")
X = data.drop('target', axis=1)
y = data['target']

# Handle missing values - this is a simple imputer, but you might want to get more sophisticated
X.fillna(X.mean(), inplace=True)

# Encode categorical variables (assuming 'protocol_type' is the only categorical variable for this example)
encoder = OneHotEncoder(drop='first')
encoded_features = encoder.fit_transform(X[['protocol_type']])
X = X.join(pd.DataFrame(encoded_features, columns=encoder.get_feature_names(['protocol_type'])))
X.drop('protocol_type', axis=1, inplace=True)

# 2. Feature Selection
# (Based on your choice. Just a placeholder here)

# 3. Data Scaling
# Create a preprocessor that scales numeric columns and leaves others untouched
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ],
    remainder='passthrough'
)

# 4. Model Evaluation
skf = StratifiedKFold(n_splits=10)
results = []

for name, model in models.items():
    # Depending on the model, you may not want to include the preprocessor
    if name in ['SVM', 'kNN', 'MLP']:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
    else:
        pipeline = Pipeline([
            ('classifier', model)
        ])
    
    # Placeholder for hyperparameter tuning. You'll need to define a parameter grid for each model.
    grid_search = GridSearchCV(pipeline, param_grid={}, cv=skf, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X, y)
    best_pipeline = grid_search.best_estimator_
    
    y_pred = cross_val_predict(best_pipeline, X, y, cv=skf)
    
    f1 = f1_score(y, y_pred, average='macro')
    gmean = g_mean(y, y_pred)
    
    results.append({
        'Model': name,
        'F1 Score': f1,
        'G-mean': gmean
    })

results_df = pd.DataFrame(results)
print(results_df)


# # Review Label Data Split

# In[181]:


# Set the color palette
sns.set_palette("coolwarm")

# Visualising the Class Imbalance

SDN.rename(columns={'label': 'label'}, inplace=True)

SDN.label.value_counts()
labels = [ 'Malicious', 'Benign',]
sizes = [dict(SDN.label.value_counts())[0], dict(SDN.label.value_counts())[1]]
plt.figure()
plt.pie(sizes, labels=labels, autopct='%1.1f%%',# Modelling - KNN Classifier
        shadow=True, startangle=80,)
plt.legend([ 'Benign','Malicious'])
plt.title('% of Benign and Maliciuos requests in the SDN Dataset')
plt.show()


# # Modelling - KNN Classifier

# ## Modelling without Feature Selection, Class Balance and Normalisation

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from time import time

def preprocess_data(data):
    # Handle duplicates
    data.drop_duplicates(inplace=True)
    # Handle missing values
    data.fillna(data.mean(), inplace=True)
    # Handle Inf values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    # Label Encoding for target and categorical columns
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])
    categorical_cols = data.select_dtypes(['object']).columns.to_list()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    X = data.drop('label', axis=1)
    y = data['label']
    return X, y


# Load and preprocess the dataset
data = pd.read_csv("Dataset_sdn.csv")
X, y = preprocess_data(data)  # Preprocess the data
X = data.drop('label', axis=1)
y = data['label']



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate correlation matrix
correlations = data.corr()
select = correlations[(correlations > 0) & (correlations < 0.7)]
TOP_CR_features = select.drop('label', errors='ignore')

# Define the hyperparameters and their possible values
param_grid = {
    'C': np.logspace(-4, 4, 10),
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'penalty': ['l1', 'l2', 'elasticnet', 'none']
}

# Function to train and evaluate the model
def train_and_evaluate(X_train, X_test, y_train, y_test, param_search):
    # Hyperparameter optimization using random search
    random_search = RandomizedSearchCV(LogisticRegression(max_iter=1000), param_distributions=param_grid, n_iter=10, cv=5, n_jobs=-1)
    random_search.fit(X_train, y_train)
    best_random_params = random_search.best_params_

    # Hyperparameter optimization using grid search
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_grid_params = grid_search.best_params_

    # Select the best hyperparameters based on the highest F1-score
    if random_search.best_score_ > grid_search.best_score_:
        best_params = best_random_params
    else:
        best_params = best_grid_params

    lr = LogisticRegression(**best_params, max_iter=1000)
    
    # Training time
    start_time = time()
    lr.fit(X_train, y_train)
    end_time = time()
    total_train_time = end_time - start_time
    
    # Testing time
    start_time = time()
    y_pred = lr.predict(X_test)
    end_time = time()
    test_time = end_time - start_time
    
    # Calculate F1 Score and G-mean
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    gmean_val = np.sqrt(tp/(tp+fn) * tn/(tn+fp))
    comp_efficiency_f1 = f1 / total_train_time
    comp_efficiency_gmean = gmean_val / total_train_time
    
    results = {
        'Training Time': total_train_time,
        'Testing Time': test_time,
        'F1 Score': f1,
        'G-mean': gmean_val,
        'Comp. Efficiency (F1)': comp_efficiency_f1,
        'Comp. Efficiency (G-mean)': comp_efficiency_gmean
    }
    
    return results

# Correlation-based feature selection
correlation_selected_features = TOP_CR_features.index.to_list()
X_train_corr = X_train[correlation_selected_features]
X_test_corr = X_test[correlation_selected_features]

# Standardize the features after feature selection
scaler = StandardScaler()
X_train_corr = scaler.fit_transform(X_train_corr)
X_test_corr = scaler.transform(X_test_corr)

# Results for Correlation-based feature selection
results_corr = train_and_evaluate(X_train_corr, X_test_corr, y_train, y_test, param_search=param_grid)
print("Results for Correlation-based feature selection:", results_corr)

# RFE-based feature selection
estimator = LogisticRegression(max_iter=1000, solver='liblinear')
selector = RFECV(estimator=estimator, step=5, cv=StratifiedKFold(5), scoring='accuracy')
selector = selector.fit(X_train, y_train)
optimal_features_count = selector.n_features_
selected_features_indices = selector.get_support(indices=True)
X_train_rfe = X_train.iloc[:, selected_features_indices]
X_test_rfe = X_test.iloc[:, selected_features_indices]

# Standardize the features after feature selection
scaler = StandardScaler()
X_train_rfe = scaler.fit_transform(X_train_rfe)
X_test_rfe = scaler.transform(X_test_rfe)

# Results for RFE-based feature selection
results_rfe = train_and_evaluate(X_train_rfe, X_test_rfe, y_train, y_test, param_search=param_grid)
print(f"Optimal number of features: {optimal_features_count}")
print("Results for RFE-based feature selection:", results_rfe)


# In[185]:


import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest

# g_mean function
def g_mean(y_true, y_pred):
    # calculate sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return np.sqrt(sensitivity * specificity)  # ** 0.5 is the same as taking the square root
    pass

def preprocess_data(data):
    # Handle duplicates
    data.drop_duplicates(inplace=True)
    # Handle missing values
    data.fillna(data.mean(), inplace=True)
    # Handle Inf values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    # Label Encoding for target and categorical columns
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])
    categorical_cols = data.select_dtypes(['object']).columns.to_list()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    X = data.drop('label', axis=1)
    y = data['label']
    return X, y

# Load data
data = pd.read_csv("Dataset_sdn.csv")
X, y = preprocess_data(data)

# Define your models with pipeline
models = {
    'RandomForest': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
}

# Grid search parameters for RandomForest
param_grid = {
    'RandomForest': {
        'classifier__n_estimators': [50, 100, 150],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
}

# Model Evaluation
skf = StratifiedKFold(n_splits=10)
results = []

for name, model in models.items():
    # Measure preprocessing + training time
    start_time_preprocess = time.time()
    
    # Feature Selection with SelectKBest (outside the pipeline to prevent data leakage)
    selector = SelectKBest(score_func=mutual_info_classif, k='all')
    X_new = selector.fit_transform(X, y)
    
    end_time_preprocess = time.time()
    
    # Measure training time
    start_time_train = time.time()
    
    grid_search = GridSearchCV(model, param_grid[name], cv=skf, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_new, y)
    
    end_time_train = time.time()

    total_train_time = (end_time_train - start_time_train) + (end_time_preprocess - start_time_preprocess)
    best_model = grid_search.best_estimator_
    
    # Measure testing time
    start_time_test = time.time()
    y_pred = cross_val_predict(best_model, X_new, y, cv=skf)
    end_time_test = time.time()
    
    test_time = end_time_test - start_time_test
    f1 = f1_score(y, y_pred, average='macro')
    gmean_val = g_mean(y, y_pred)
    
    # Computation efficiency
    comp_efficiency_f1 = f1 / (total_train_time + test_time)
    comp_efficiency_gmean = gmean_val / (total_train_time + test_time)
    
    results.append({
        'Model': name,
        'Training Time': total_train_time,
        'Testing Time': test_time,
        'F1 Score': f1,
        'G-mean': gmean_val,
        'Comp. Efficiency (F1)': comp_efficiency_f1,
        'Comp. Efficiency (G-mean)': comp_efficiency_gmean
    })

results_df = pd.DataFrame(results)
print(results_df)


# In[187]:


# g_mean function
def g_mean(y_true, y_pred):
    # calculate sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return np.sqrt(sensitivity * specificity)  # ** 0.5 is the same as taking the square root


def preprocess_data(data):
    # Handle duplicates
    data.drop_duplicates(inplace=True)
    # Handle missing values
    data.fillna(data.mean(), inplace=True)
    # Handle Inf values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    # Label Encoding for target and categorical columns
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])
    categorical_cols = data.select_dtypes(['object']).columns.to_list()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    X = data.drop('label', axis=1)
    y = data['label']
    return X, y

# Load data
data = pd.read_csv("Dataset_sdn.csv")
X, y = preprocess_data(data)

# Define your models with pipeline
models = {
    'RandomForest': RandomForestClassifier(random_state=42)
}

# Grid search parameters for RandomForest
param_grid = {
    'RandomForest': {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
}

# Model Evaluation
skf = StratifiedKFold(n_splits=10)
results = []

for name, model in models.items():
    # Measure preprocessing + training time
    start_time_preprocess = time.time()
    
    # Feature Selection with SelectKBest (outside the pipeline to prevent data leakage)
    selector = SelectKBest(score_func=mutual_info_classif, k='all')
    X_new = selector.fit_transform(X, y)
    
    end_time_preprocess = time.time()
    
    # Measure training time
    start_time_train = time.time()
    
    grid_search = GridSearchCV(model, param_grid[name], cv=skf, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_new, y)
    
    end_time_train = time.time()

    total_train_time = (end_time_train - start_time_train) + (end_time_preprocess - start_time_preprocess)
    best_model = grid_search.best_estimator_
    
    # Measure testing time
    start_time_test = time.time()
    y_pred = cross_val_predict(best_model, X_new, y, cv=skf)
    end_time_test = time.time()
    
    test_time = end_time_test - start_time_test
    f1 = f1_score(y, y_pred, average='macro')
    gmean_val = g_mean(y, y_pred)
    
    # Computation efficiency
    comp_efficiency_f1 = f1 / (total_train_time + test_time)
    comp_efficiency_gmean = gmean_val / (total_train_time + test_time)
    
    results.append({
        'Model': name,
        'Training Time': total_train_time,
        'Testing Time': test_time,
        'F1 Score': f1,
        'G-mean': gmean_val,
        'Comp. Efficiency (F1)': comp_efficiency_f1,
        'Comp. Efficiency (G-mean)': comp_efficiency_gmean
    })

results_df = pd.DataFrame(results)
print(results_df)


# In[188]:


# g_mean function
def g_mean(y_true, y_pred):
    # calculate sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return np.sqrt(sensitivity * specificity)  # ** 0.5 is the same as taking the square root


def preprocess_data(data):
    # Handle duplicates
    data.drop_duplicates(inplace=True)
    # Handle missing values
    data.fillna(data.mean(), inplace=True)
    # Handle Inf values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    # Label Encoding for target and categorical columns
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])
    categorical_cols = data.select_dtypes(['object']).columns.to_list()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    X = data.drop('label', axis=1)
    y = data['label']
    return X, y

# Load data
data = pd.read_csv("Dataset_sdn.csv")
X, y = preprocess_data(data)

# Define your models with pipeline
models = {
    'RandomForest': RandomForestClassifier(random_state=42)
}

# Grid search parameters for RandomForest
param_grid = {
    'RandomForest': {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
}

# Model Evaluation
skf = StratifiedKFold(n_splits=10)
results = []

for name, model in models.items():
    # Measure preprocessing + training time
    start_time_preprocess = time.time()
    end_time_preprocess = time.time() # Just to maintain consistency
    
    
    # Measure training time
    start_time_train = time.time()
    
    grid_search = GridSearchCV(model, param_grid[name], cv=skf, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_new, y)
    
    end_time_train = time.time()

    total_train_time = (end_time_train - start_time_train) + (end_time_preprocess - start_time_preprocess)
    best_model = grid_search.best_estimator_
    
    # Measure testing time
    start_time_test = time.time()
    y_pred = cross_val_predict(best_model, X_new, y, cv=skf)
    end_time_test = time.time()
    
    test_time = end_time_test - start_time_test
    f1 = f1_score(y, y_pred, average='macro')
    gmean_val = g_mean(y, y_pred)
    
    # Computation efficiency
    comp_efficiency_f1 = f1 / (total_train_time + test_time)
    comp_efficiency_gmean = gmean_val / (total_train_time + test_time)
    
    results.append({
        'Model': name,
        'Training Time': total_train_time,
        'Testing Time': test_time,
        'F1 Score': f1,
        'G-mean': gmean_val,
        'Comp. Efficiency (F1)': comp_efficiency_f1,
        'Comp. Efficiency (G-mean)': comp_efficiency_gmean
    })

results_df = pd.DataFrame(results)
print(results_df)


# In[190]:


sns.pairplot(SDN, hue='label' )


# In[192]:


from sklearn.model_selection import RandomizedSearchCV

# g_mean function
def g_mean(y_true, y_pred):
    # calculate sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return np.sqrt(sensitivity * specificity)  # ** 0.5 is the same as taking the square root


def preprocess_data(data):
    # Handle duplicates
    data.drop_duplicates(inplace=True)
    # Handle missing values
    data.fillna(data.mean(), inplace=True)
    # Handle Inf values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    # Label Encoding for target and categorical columns
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])
    categorical_cols = data.select_dtypes(['object']).columns.to_list()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    X = data.drop('label', axis=1)
    y = data['label']
    return X, y

# Load data
data = pd.read_csv("Dataset_sdn.csv")
X, y = preprocess_data(data)

# Define your models with pipeline
models = {
    'RandomForest': RandomForestClassifier(random_state=42)
}


# Random Search Parameters for RandomForest
param_dist = {
    'RandomForest': {
        'n_estimators': range(10, 200, 10),
        'max_depth': [None] + list(range(1, 50)),
        'min_samples_split': range(2, 20, 2),
        'min_samples_leaf': range(1, 20, 2)
    }
}

# Placeholder for Grid Search Parameters for RandomForest
param_grid = {
    'RandomForest': {}
}

# Model Evaluation
skf = StratifiedKFold(n_splits=10)
results = []

for name, model in models.items():
    # Randomized Search
    random_search = RandomizedSearchCV(model, param_distributions=param_dist[name], n_iter=20, 
                                       cv=skf, scoring='f1_macro', n_jobs=-1, random_state=42)
    random_search.fit(X, y)
    
    # Using the best params from Randomized Search to define a new grid for Grid Search
    best_params = random_search.best_params_
    param_grid[name] = {param: [value-1, value, value+1] for param, value in best_params.items() if isinstance(value, int)}
    for param, value in best_params.items():
        if param not in param_grid[name]:
            param_grid[name][param] = [value]
    
    # Grid Search
    start_time_train = time.time()
    grid_search = GridSearchCV(model, param_grid[name], cv=skf, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X, y)
    end_time_train = time.time()

    total_train_time = end_time_train - start_time_train
    best_model = grid_search.best_estimator_
    
    # Testing
    start_time_test = time.time()
    y_pred = cross_val_predict(best_model, X, y, cv=skf)
    end_time_test = time.time()
    
    test_time = end_time_test - start_time_test
    f1 = f1_score(y, y_pred, average='macro')
    gmean_val = g_mean(y, y_pred)
    
    # Computation efficiency
    comp_efficiency_f1 = f1 / (total_train_time + test_time)
    comp_efficiency_gmean = gmean_val / (total_train_time + test_time)
    
    results.append({
        'Model': name,
        'Training Time': total_train_time,
        'Testing Time': test_time,
        'F1 Score': f1,
        'G-mean': gmean_val,
        'Comp. Efficiency (F1)': comp_efficiency_f1,
        'Comp. Efficiency (G-mean)': comp_efficiency_gmean
    })

results_df = pd.DataFrame(results)
print(results_df)


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from time import time
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings

# Preprocess data
def preprocess_data(data):
    # Handle duplicates
    data.drop_duplicates(inplace=True)
    # Handle missing values
    data.fillna(data.mean(numeric_only=True), inplace=True)  # Explicitly set numeric_only
    # Handle Inf values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    # Label Encoding
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])
    categorical_cols = data.select_dtypes(['object']).columns.to_list()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    X = data.drop('label', axis=1)
    y = data['label']
    return X, y

# Load and preprocess the dataset
data = pd.read_csv("Dataset_sdn.csv")
X, y = preprocess_data(data)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Use SelectKBest with Information Gain
selector = SelectKBest(mutual_info_classif, k='all')
selector.fit(X_train, y_train)
X_train_ig = selector.transform(X_train)
X_test_ig = selector.transform(X_test)

# Define the hyperparameters
param_grid = {
    'C': np.logspace(-4, 4, 10),
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'penalty': ['l1', 'l2', 'elasticnet', 'none']
}

# Define StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Train and evaluate the model
def train_and_evaluate(X_train, X_test, y_train, y_test, param_search):
    random_search = RandomizedSearchCV(LogisticRegression(max_iter=2000), param_distributions=param_grid, n_iter=10, cv=skf, n_jobs=-1)
    random_search.fit(X_train, y_train)
    best_random_params = random_search.best_params_

    grid_search = GridSearchCV(LogisticRegression(max_iter=2000), param_grid=param_grid, cv=skf, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_grid_params = grid_search.best_params_

    if random_search.best_score_ > grid_search.best_score_:
        best_params = best_random_params
    else:
        best_params = best_grid_params

    lr = LogisticRegression(**best_params, max_iter=2000)
    start_time = time()
    lr.fit(X_train, y_train)
    end_time = time()
    total_train_time = end_time - start_time

    start_time = time()
    y_pred = lr.predict(X_test)
    end_time = time()
    test_time = end_time - start_time

    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    gmean_val = np.sqrt(tp/(tp+fn) * tn/(tn+fp))
    comp_efficiency_f1 = f1 / total_train_time
    comp_efficiency_gmean = gmean_val / total_train_time

    results = {
        'Training Time': total_train_time,
        'Testing Time': test_time,
        'F1 Score': f1,
        'G-mean': gmean_val,
        'Comp. Efficiency (F1)': comp_efficiency_f1,
        'Comp. Efficiency (G-mean)': comp_efficiency_gmean
    }
    return results

results_ig = train_and_evaluate(X_train_ig, X_test_ig, y_train, y_test, param_search=param_grid)
print("Results for Information Gain-based feature selection:", results_ig)



# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from time import time
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings

# Preprocess data
def preprocess_data(data):
    # Handle duplicates
    data.drop_duplicates(inplace=True)
    # Handle missing values
    data.fillna(data.mean(numeric_only=True), inplace=True)  # Explicitly set numeric_only
    # Handle Inf values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    # Label Encoding
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])
    categorical_cols = data.select_dtypes(['object']).columns.to_list()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    X = data.drop('label', axis=1)
    y = data['label']
    return X, y

# Load and preprocess the dataset
data = pd.read_csv("Dataset_sdn.csv")
X, y = preprocess_data(data)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Use SelectKBest with Information Gain
selector = SelectKBest(mutual_info_classif, k='all')
selector.fit(X_train, y_train)
X_train_ig = selector.transform(X_train)
X_test_ig = selector.transform(X_test)

# Standard Scaling
scaler = StandardScaler()
X_train_ig = scaler.fit_transform(X_train_ig)
X_test_ig = scaler.transform(X_test_ig)

# Define the hyperparameters for Random Forest
param_grid = {
    'n_estimators': [10, 50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Define StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Train and evaluate the model
def train_and_evaluate(X_train, X_test, y_train, y_test, param_search):
    random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_grid, n_iter=10, cv=skf, n_jobs=-1)
    random_search.fit(X_train, y_train)
    best_random_params = random_search.best_params_

    grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=skf, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_grid_params = grid_search.best_params_

    if random_search.best_score_ > grid_search.best_score_:
        best_params = best_random_params
    else:
        best_params = best_grid_params

    rf = RandomForestClassifier(**best_params)
    start_time = time()
    rf.fit(X_train, y_train)
    end_time = time()
    total_train_time = end_time - start_time

    start_time = time()
    y_pred = rf.predict(X_test)
    end_time = time()
    test_time = end_time - start_time

    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    gmean_val = np.sqrt(tp/(tp+fn) * tn/(tn+fp))
    comp_efficiency_f1 = f1 / total_train_time
    comp_efficiency_gmean = gmean_val / total_train_time

    results = {
        'Training Time': total_train_time,
        'Testing Time': test_time,
        'F1 Score': f1,
        'G-mean': gmean_val,
        'Comp. Efficiency (F1)': comp_efficiency_f1,
        'Comp. Efficiency (G-mean)': comp_efficiency_gmean
    }
    return results

results_ig = train_and_evaluate(X_train_ig, X_test_ig, y_train, y_test, param_search=param_grid)
print("Results for Information Gain-based feature selection:", results_ig)


# ## Randome Forest : Modelling with IG Feature Selection & STD Scaler

# In[16]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV, cross_val_predict
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

def preprocess_data(data):
    data.drop_duplicates(inplace=True)
    data.fillna(data.mean(numeric_only=True), inplace=True)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])
    categorical_cols = data.select_dtypes(['object']).columns.to_list()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    X = data.drop('label', axis=1)
    y = data['label']
    return X, y

def g_mean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return np.sqrt(sensitivity * specificity)

data = pd.read_csv("Dataset_sdn.csv")
X, y = preprocess_data(data)

selector = SelectKBest(mutual_info_classif, k='all')
selector.fit(X, y)
X_selected = selector.transform(X)

# Apply feature scaling only to the selected features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

param_grid = {
    'n_estimators': [10, 50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

start_time_train = time()

random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_grid, n_iter=20, cv=skf, n_jobs=-1)
random_search.fit(X_scaled, y)
best_random_params = random_search.best_params_

refined_param_grid = {f'{param}': [value-1, value, value+1] for param, value in best_random_params.items() if isinstance(value, int)}
for param, value in best_random_params.items():
    if f'{param}' not in refined_param_grid:
        refined_param_grid[f'{param}'] = [value]

grid_search = GridSearchCV(RandomForestClassifier(), param_grid=refined_param_grid, cv=skf, n_jobs=-1)
grid_search.fit(X_scaled, y)

total_train_time = time() - start_time_train

best_model = grid_search.best_estimator_
y_pred = cross_val_predict(best_model, X_scaled, y, cv=skf)

test_time = time() - total_train_time

f1 = f1_score(y, y_pred, average='macro')
gmean_val = g_mean(y, y_pred)
auc = roc_auc_score(y, y_pred)

comp_efficiency_f1 = f1 / (total_train_time + test_time)
comp_efficiency_gmean = gmean_val / (total_train_time + test_time)

results = {
    'Model': 'RandomForest',
    'F1 Score': f1,
    'G-mean': gmean_val,
    'AUC': auc,
    'Comp. Efficiency (F1)': comp_efficiency_f1,
    'Comp. Efficiency (G-mean)': comp_efficiency_gmean
}

results_df = pd.DataFrame([results])
print(results_df)

cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

feature_importances = best_model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(X.columns[sorted_indices], feature_importances[sorted_indices])
plt.xticks(rotation=90)
plt.show()

selected_features = X.columns[sorted_indices[:10]]
print("Selected Features:", selected_features)


# ## Decision Tree : Modelling with IG Feature Selection & STD Scaler

# In[18]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

def g_mean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return np.sqrt(sensitivity * specificity)

data = pd.read_csv("Dataset_sdn.csv")
data.drop_duplicates(inplace=True)
data.fillna(data.mean(numeric_only=True), inplace=True)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])
categorical_cols = data.select_dtypes(['object']).columns.to_list()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])
X = data.drop('label', axis=1)
y = data['label']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Select 10 best features using information gain (mutual information)
selector = SelectKBest(mutual_info_classif, k=10)
X_train = selector.fit_transform(X_train, y_train)
X_val = selector.transform(X_val)
X_test = selector.transform(X_test)

param_grid = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

start_time_train = time()
random_search = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions=param_grid, n_iter=20, cv=skf, n_jobs=-1)
random_search.fit(X_train, y_train)
end_time_train = time() - start_time_train

best_random_params = random_search.best_params_

refined_param_grid = {f'{param}': [value-1, value, value+1] for param, value in best_random_params.items() if isinstance(value, int)}
for param, value in best_random_params.items():
    if f'{param}' not in refined_param_grid:
        refined_param_grid[f'{param}'] = [value]

start_time_val = time()
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=refined_param_grid, cv=skf, n_jobs=-1)
grid_search.fit(X_val, y_val)
end_time_val = time() - start_time_val

best_model = grid_search.best_estimator_

start_time_test = time()
y_pred = best_model.predict(X_test)
end_time_test = time() - start_time_test

f1 = f1_score(y_test, y_pred, average='macro')
gmean_val = g_mean(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print(f"Training Time: {end_time_train}, Validation Time: {end_time_val}, Test Time: {end_time_test}")

results = {
    'Model': 'Decision Tree',
    'F1 Score': f1,
    'G-mean': gmean_val,
    'AUC': auc
}

results_df = pd.DataFrame([results])
print(results_df)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

feature_importances = best_model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(np.array(X.columns[selector.get_support()])[sorted_indices], feature_importances[sorted_indices])
plt.xticks(rotation=90)
plt.show()

print("Selected Features:", np.array(X.columns[selector.get_support()])[sorted_indices[:10]])


# ## Decision Tree : Modelling with STD Scaler

# In[19]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

def g_mean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return np.sqrt(sensitivity * specificity)

data = pd.read_csv("Dataset_sdn.csv")
data.drop_duplicates(inplace=True)
data.fillna(data.mean(numeric_only=True), inplace=True)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])
categorical_cols = data.select_dtypes(['object']).columns.to_list()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])
X = data.drop('label', axis=1)
y = data['label']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

param_grid = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

start_time_train = time()
random_search = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions=param_grid, n_iter=20, cv=skf, n_jobs=-1)
random_search.fit(X_train, y_train)
end_time_train = time() - start_time_train

best_random_params = random_search.best_params_

refined_param_grid = {f'{param}': [value-1, value, value+1] for param, value in best_random_params.items() if isinstance(value, int)}
for param, value in best_random_params.items():
    if f'{param}' not in refined_param_grid:
        refined_param_grid[f'{param}'] = [value]

start_time_val = time()
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=refined_param_grid, cv=skf, n_jobs=-1)
grid_search.fit(X_val, y_val)
end_time_val = time() - start_time_val

best_model = grid_search.best_estimator_

start_time_test = time()
y_pred = best_model.predict(X_test)
end_time_test = time() - start_time_test

f1 = f1_score(y_test, y_pred, average='macro')
gmean_val = g_mean(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print(f"Training Time: {end_time_train}, Validation Time: {end_time_val}, Test Time: {end_time_test}")

results = {
    'Model': 'Decision Tree',
    'F1 Score': f1,
    'G-mean': gmean_val,
    'AUC': auc
}

results_df = pd.DataFrame([results])
print(results_df)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

feature_importances = best_model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(X.columns[sorted_indices], feature_importances[sorted_indices])
plt.xticks(rotation=90)
plt.show()

print("Selected Features:", X.columns[sorted_indices[:10]])


# ## Decision Tree : Modelling without IG Feature Section & STD Scaler

# In[20]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

def g_mean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return np.sqrt(sensitivity * specificity)

data = pd.read_csv("Dataset_sdn.csv")
data.drop_duplicates(inplace=True)
data.fillna(data.mean(numeric_only=True), inplace=True)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])
categorical_cols = data.select_dtypes(['object']).columns.to_list()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])
X = data.drop('label', axis=1)
y = data['label']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)

param_grid = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

start_time_train = time()
random_search = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions=param_grid, n_iter=20, cv=skf, n_jobs=-1)
random_search.fit(X_train, y_train)
end_time_train = time() - start_time_train

best_random_params = random_search.best_params_

refined_param_grid = {f'{param}': [value-1, value, value+1] for param, value in best_random_params.items() if isinstance(value, int)}
for param, value in best_random_params.items():
    if f'{param}' not in refined_param_grid:
        refined_param_grid[f'{param}'] = [value]

start_time_val = time()
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=refined_param_grid, cv=skf, n_jobs=-1)
grid_search.fit(X_val, y_val)
end_time_val = time() - start_time_val

best_model = grid_search.best_estimator_

start_time_test = time()
y_pred = best_model.predict(X_test)
end_time_test = time() - start_time_test

f1 = f1_score(y_test, y_pred, average='macro')
gmean_val = g_mean(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print(f"Training Time: {end_time_train}, Validation Time: {end_time_val}, Test Time: {end_time_test}")

results = {
    'Model': 'Decision Tree',
    'F1 Score': f1,
    'G-mean': gmean_val,
    'AUC': auc
}

results_df = pd.DataFrame([results])
print(results_df)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

feature_importances = best_model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(X.columns[sorted_indices], feature_importances[sorted_indices])
plt.xticks(rotation=90)
plt.show()

print("Selected Features:", X.columns[sorted_indices[:10]])


# In[ ]:





# In[ ]:





# In[21]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV, cross_val_predict
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

def preprocess_data(data):
    data.drop_duplicates(inplace=True)
    data.fillna(data.mean(numeric_only=True), inplace=True)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])
    categorical_cols = data.select_dtypes(['object']).columns.to_list()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    X = data.drop('label', axis=1)
    y = data['label']
    return X, y

def g_mean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return np.sqrt(sensitivity * specificity)

data = pd.read_csv("Dataset_sdn.csv")
X, y = preprocess_data(data)

# Feature selection
selector = RFE(MLPClassifier(), n_features_to_select=10)
selector.fit(X, y)
X_selected = selector.transform(X)

# Apply feature scaling only to the selected features
scaler = QuantileTransformer()
X_scaled = scaler.fit_transform(X_selected)

param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50), (100, 100)],
    'activation': ['relu', 'logistic', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'max_iter': [100, 200, 300]
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

start_time_train = time()

random_search = RandomizedSearchCV(MLPClassifier(), param_distributions=param_grid, n_iter=20, cv=skf, n_jobs=-1)
random_search.fit(X_scaled, y)
best_random_params = random_search.best_params_

refined_param_grid = {f'{param}': [value-1, value, value+1] for param, value in best_random_params.items() if isinstance(value, int)}
for param, value in best_random_params.items():
    if f'{param}' not in refined_param_grid:
        refined_param_grid[f'{param}'] = [value]

grid_search = GridSearchCV(MLPClassifier(), param_grid=refined_param_grid, cv=skf, n_jobs=-1)
grid_search.fit(X_scaled, y)

total_train_time = time() - start_time_train

best_model = grid_search.best_estimator_
y_pred = cross_val_predict(best_model, X_scaled, y, cv=skf)

test_time = time() - total_train_time

f1 = f1_score(y, y_pred, average='macro')
gmean_val = g_mean(y, y_pred)
auc = roc_auc_score(y, y_pred)

comp_efficiency_f1 = f1 / (total_train_time + test_time)
comp_efficiency_gmean = gmean_val / (total_train_time + test_time)

results = {
    'Model': 'MLP',
    'F1 Score': f1,
    'G-mean': gmean_val,
    'AUC': auc,
    'Comp. Efficiency (F1)': comp_efficiency_f1,
    'Comp. Efficiency (G-mean)': comp_efficiency_gmean
}

results_df = pd.DataFrame([results])
print(results_df)

cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Display selected features
selected_features = X.columns[selector.support_]
print("Selected Features:", selected_features)


# In[ ]:





# In[ ]:




