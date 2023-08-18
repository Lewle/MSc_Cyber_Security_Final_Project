#!/usr/bin/env python
# coding: utf-8

# # K-Nearest Neighbour Classifier

# In[1]:


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

# In[2]:


# Read .csv file

# Use glob to match the pattern 'csv'
files = glob.glob('*_ISCX.csv')

# Create an empty list to store dataframes
dataframes = []

# Loop over the list of csv files
for csv in files:
    # Read csv file and append to the list
    df = pd.read_csv(csv)
    dataframes.append(df)

# Concatenate all dataframes in the list
CIC2017 = pd.concat(dataframes, ignore_index=True)

# Save the combined dataframe to a new csv file
CIC2017.to_csv('combined.csv', index=False)


# In[3]:


# Read .csv file

CIC2017 = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
CIC2017.head()


# In[4]:


[col for col in CIC2017.columns] # lets just look at the column titles


# In[5]:


# Check the unique labels in the dataset

print(CIC2017[' Label'].unique())


# In[6]:


CIC2017.shape


# # Data Cleaning

# ### Missing Values

# In[7]:


# Checking number of missing (null) values

CIC2017_null_count = CIC2017.isnull().sum()
CIC2017_null_count


# In[8]:


# Visualisation of missing values in a bar chart
msno.bar(CIC2017)


# In[9]:


# Visualisation of the location of missing values
msno.matrix(CIC2017)


# In[10]:


# How many total missing values do we have?
total_cells = np.product(CIC2017.shape)
total_missing_values = CIC2017_null_count.sum()
print ('CIC2017_total_cells',total_cells)
print ('CIC2017_total_missing_values',total_missing_values)


# In[11]:


# Percent of data that is missing in SDN dataframe
percent_missing = (total_missing_values/total_cells) * 100
print('CIC2017_percent_missing_values',percent_missing)


# In[12]:


# Percent of data that is missing in each SDN feature

CIC2017_nulls_percentage = CIC2017.isnull().sum()/len(CIC2017)*100

print('the percentages of null values per feature in SDN:\n')
print(round(CIC2017_nulls_percentage,2))


# In[13]:


# Dropping missing values

print('Dataframe Shape',CIC2017.shape)

CIC2017.dropna(inplace=True)

print ('Dataframe Shape after dropping missing values',CIC2017.shape)


# In[14]:


# Visualisation of dataframe after dropping missing values in a bar chart
msno.bar(CIC2017)


# ### Duplicated Values

# In[15]:


# Count number of duplicated rows, except the first occurrences

len(CIC2017[CIC2017.duplicated(keep='first')])


# In[16]:


# Visualised duplicated rows, except the first occurrences
CIC2017_duplicateRows = CIC2017[CIC2017.duplicated(subset=None, keep='first')]
CIC2017_duplicateRows


# In[17]:


# Drop only the repeated rows (duplicated rows after the first occurrences)

print ('CIC2017 after dropping missing values',CIC2017.shape)

CIC2017.drop_duplicates(inplace = True)

#  Display after removing null values & duplicated rows after the first occurrences

print ('CIC2017 after dropping missing values & duplicated rows',CIC2017.shape)


# ### Low Variance / Single Value Features

# In[18]:


# Drop columns with constant value (std = 0)


single_value_list = []
for cols in (CIC2017.select_dtypes(include=['number'])):                     # Select only the 'number' columns
    if (CIC2017[cols].std())==0:                                             # Calculate std of those 'number' columns = std with 0 value
        single_value_list.append(cols)
                        
    
print('Columns with single value:\n',np.array(single_value_list),'\n')

print ('CIC2017 Shape after dropping missing values & duplicated rows',CIC2017.shape)

CIC2017.drop(single_value_list,axis=1,inplace=True) 

print ('CIC2017 Shape after dropping missing values, duplicated rows & low variance rows',CIC2017.shape)


# In[19]:


# Numeric Features's Structure
CIC2017_numerical_features = [feature for feature in CIC2017.columns if CIC2017[feature].dtypes != 'O']
CIC2017_unique_numerical_values = CIC2017[CIC2017_numerical_features].nunique(axis=0).sort_values()

# Set the color palette
sns.set_palette("pastel")

# Plot information with y-axis in log-scale
CIC2017_unique_numerical_values.plot.bar(logy=True, title="Unique values per feature in CIC2017D");


# In[20]:


# Select columns with less than 5 unique values

num_features_less_than_5_unique = [feature for feature in CIC2017.columns if CIC2017[feature].dtypes != 'O' and CIC2017[feature].nunique() < 5]
num_features_less_than_5_unique


# In[21]:


# Drop columns with less than 5 unique values

print ('CIC2017 Shape after dropping missing values, duplicated rows & low variance rows',CIC2017.shape)

CIC2017.drop(num_features_less_than_5_unique,axis=1,inplace=True) 

print ('CIC2017 Shape after dropping missing values, duplicated rows, low variance rows & constant value columns',CIC2017.shape)


# ### Identifying and Handling Outliers

# In[22]:


# Identify numeric columns from the SDN dataset
numeric_columns = CIC2017.select_dtypes(include=[np.number])

# Set the color palette
sns.set_palette("coolwarm")

# Plotting box plots for each numeric column
plt.figure()  

for col in numeric_columns.columns:
    plt.subplot(2, 2, 2)  # Adjust the number of rows and columns for subplots
    sns.boxplot(data=CIC2017[col])
    plt.title(f'Box Plot of {col}') 
    plt.show()


# In[23]:


#Set the color palette
#sns.set_palette("coolwarm")

# Numeric Feature Distribution via Histrograms

#cols = 3
#rows = 10
#num_cols = CIC2017D.select_dtypes(exclude='object').columns
#fig = plt.figure( figsize=(cols*5, rows*5))
#for i, col in enumerate(num_cols):
    
    #ax=fig.add_subplot(rows,cols,i+1)
    
    #sns.histplot(x = CIC2017D[col], ax = ax)
    
#fig.tight_layout()  
#plt.show()


# In[24]:


#CIC2017D.plot(lw=0,
          #marker=".",
          #subplots=True,
          #layout=(-1, 4),
          #figsize=(15, 30),
          #markersize=1);


# In[25]:


#sns.pairplot(SDN, hue='label')


# ### Converting Categorical Features to Numerical Features

# In[26]:


# Check data types
CIC2017.info()


# In[27]:


#print ('CIC2017D Shape after dropping missing values, duplicated rows, low variance rows & constant value columns',CIC2017D.shape)
#print ('CIC2017D Shape after dropping missing values, duplicated rows, low variance rows, constant value columns & after Lable Encoding',CIC2017D.shape)


# ### Saving the Pre-processed file

# In[28]:


#CIC2017D.to_csv('CIC2017D_preprocessed_data.csv', index=False)


# In[29]:


# Read .csv file

#CIC2017D = pd.read_csv('CIC2017D_preprocessed_data.csv')
#CIC2017D.head()


# In[30]:


# replace infinities with NaN

# CIC2017D.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[31]:


#from scipy.stats.mstats import winsorize

# Apply winsorize to each column
#for col in CIC2017D.columns:
    #CIC2017D[col] = winsorize(CIC2017D[col], limits=[0.05, 0.05])


# ### Handle Inf Values

# In[32]:


# Replace infinite values with NaN

CIC2017.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[33]:


# Drop rows with NaN 

CIC2017.dropna(inplace=True)#### Handle Inf Values


# ### Handle str in the Label Column

# In[34]:


# Replace strings with numerical values in the Label column

CIC2017[' Label'] = CIC2017[' Label'].replace({'BENIGN': 0, 'DDoS': 1})


# ### Effect of Scalling on Different ML Model Performance

# #### Effect of StandardScalling on Different ML Model Performance

# In[35]:


# Prepare the data

CIC2017_ALL_s = CIC2017.copy() # Std Scalling
CIC2017_ALL_q = CIC2017.copy() # QT Scalling

ALLxs = CIC2017_ALL_s.drop(columns=[' Label'])
ALLys = CIC2017_ALL_s[' Label']

ALLxq = CIC2017_ALL_q.drop(columns=[' Label'])
ALLyq = CIC2017_ALL_q[' Label']


# In[36]:


# Split the data
ALLxs_train, ALLxs_test, ALLys_train, ALLys_test = train_test_split(ALLxs, ALLys, test_size=0.3, random_state=42)
ALLxq_train, ALLxq_test, ALLyq_train, ALLyq_test = train_test_split(ALLxq, ALLyq, test_size=0.3, random_state=42)


# In[37]:


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


# In[38]:


#from sklearn.preprocessing import MinMaxScaler
#min_max = MinMaxScaler()
#ALLmx  = min_max.fit(ALLx)


# In[39]:


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


# In[40]:


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



# In[41]:


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


# In[54]:


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
sns.kdeplot(ALLxs[' Average Packet Size'], ax=ax4)

# After Std Scaling: Down/Up Ratio
ax5.set_title('After Std Scaling')
sns.kdeplot(ALLxss_test[:, 40], ax=ax5, color='red')

# After QT Scaling: Down/Up Ratio
ax6.set_title('After QT Scaling')
sns.kdeplot(ALLxqs_test[:, 40], ax=ax6, color='orange')

plt.tight_layout()
plt.show()


# In[57]:


fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(ncols=6, figsize=(18, 5))

# Before scaling: Packet Length Variance
ax1.set_title('Before Scaling')
sns.kdeplot(ALLxs[' Packet Length Variance'], ax=ax1)

# After Std Scaling: Packet Length Variance
ax2.set_title('After Std Scaling')
sns.kdeplot(ALLxss_test[:, 38], ax=ax2, color='red')

# After QT Scaling: Packet Length Variance
ax3.set_title('After QT Scaling')
sns.kdeplot(ALLxqs_test[:, 38], ax=ax3, color='orange')

# Before scaling: Down/Up Ratio
ax4.set_title('Before Scaling')
sns.kdeplot(ALLxs[' Down/Up Ratio'], ax=ax4)

# After Std Scaling: Down/Up Ratio
ax5.set_title('After Std Scaling')
sns.kdeplot(ALLxss_test[:, 39], ax=ax5, color='red')

# After QT Scaling: Down/Up Ratio
ax6.set_title('After QT Scaling')
sns.kdeplot(ALLxqs_test[:, 39], ax=ax6, color='orange')

plt.tight_layout()
plt.show()



# #### Effect of StandardScalling on Different Algorithms

# In[58]:


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


# In[59]:


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


# In[43]:


# CICIDS2017 - Scaling plots


# Define a list of pastel colors
#!pip install --upgrade matplotlib

colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'purple']

models = ['LR', 'KNN', 'DT', 'RF', 'SVM', 'XGB', 'MLP']
f1_scores = [ 0.966 , 0.9974 , 0.9998 , 0.9999 , 0.9339 , 0.9999 , 0.9899 ]  
g_means = [ 0.9632 , 0.9973, 0.9998 , 0.9999, 0.9317 , 0.9999,  0.9904 ]  

plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.scatter(g_means[i], f1_scores[i], label=model, s=100)
plt.xlabel('G- Means')
plt.ylabel('F1 Score')
plt.title('Without Scaling: F1 Score vs G-Means')
plt.legend()
plt.grid(True)
plt.show()


# In[44]:


# CICIDS2017 - Scaling plots

models = ['LR', 'KNN', 'DT', 'RF', 'SVM', 'XGB', 'MLP']
f1_scores = [0.9978 , 0.9996 , 0.9998 , 0.9999 , 0.9985 , 0.9999 , 0.9994]  
g_means = [  0.9977 , 0.9995 , 0.9998, 0.9999 ,  0.9985 ,  0.9999,  0.9995 ]  

plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.scatter(g_means[i], f1_scores[i], label=model, s=100)
plt.xlabel('G- Means')
plt.ylabel('F1 Score')
plt.title('STD Scaling: F1 Score vs G-Means')
plt.legend()
plt.grid(True)


# In[45]:


# CICIDS2017 - Scaling plots


models = ['LR', 'KNN', 'DT', 'RF', 'SVM', 'XGB', 'MLP']
f1_scores = [0.9989, 0.9998 , 0.9998 , 0.9999 , 0.9996 , 0.9999, 0.9998]  
g_means = [   0.9989 , 0.9998 , 0.9998 , 0.9999 , 0.9997 , 0.9999 , 0.9998 ] 

plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.scatter(g_means[i], f1_scores[i], label=model, s=100)
plt.xlabel('G- Means')
plt.ylabel('F1 Score')
plt.title('QT Scaling: F1 Score vs G-Means')
plt.legend()


# In[46]:


# CICIDS2017 - Scaling plots


models = ['LR', 'KNN', 'DT', 'RF', 'SVM', 'XGB', 'MLP']
f1_scores = [ 0.966 , 0.9974 , 0.9998 , 0.9999 , 0.9339 , 0.9999 , 0.9899 ]   
training_times= [ 13.18, 19.05 , 1.38 , 26.75 , 1551.99, 14.41 , 30.81]  

plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.scatter(training_times[i], f1_scores[i], label=model, s=100)
plt.xlabel('Training Times')
plt.ylabel('F1 Score')
plt.title('Without Scaling: F1 Score vs Training Times')
plt.legend()
plt.grid(True)
plt.show()


# In[47]:


# CICIDS2017 - Scaling plots


models = ['LR', 'KNN', 'DT', 'RF', 'SVM', 'XGB', 'MLP']
g_means = [ 0.9632 , 0.9973, 0.9998 , 0.9999, 0.9317 , 0.9999,  0.9904 ]  
training_times= [ 13.18, 19.05 , 1.38 , 26.75 , 1551.99, 14.41 , 30.81]  

plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.scatter(training_times[i], g_means[i], label=model, s=100)
plt.xlabel('Training Times')
plt.ylabel('G-Means')
plt.title('Without Scaling: G-Means vs Training Times')
plt.legend()
plt.grid(True)
plt.show()


# In[48]:


# CICIDS2017 - Scaling plots

models = ['LR', 'KNN', 'DT', 'RF', 'SVM', 'XGB', 'MLP']
f1_scores = [0.9978 , 0.9996 , 0.9998 , 0.9999 , 0.9985 , 0.9999 , 0.9994]  
training_times= [ 4.07, 22.28, 1.18, 21.4, 369.66, 13.77, 17.9]  

plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.scatter(training_times[i], f1_scores[i], label=model, s=100)
plt.xlabel('Training Times')
plt.ylabel('F1 Score')
plt.title('STD Scaling: F1 Score vs Training Times')
plt.legend()
plt.grid(True)


# In[49]:


# CICIDS2017 - Scaling plots

models = ['LR', 'KNN', 'DT', 'RF', 'SVM', 'XGB', 'MLP'] 
g_means = [  0.9977 , 0.9995 , 0.9998, 0.9999 ,  0.9985 ,  0.9999,  0.9995 ]  
training_times= [ 4.07, 22.28, 1.18, 21.4, 369.66, 13.77, 17.9]  

plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.scatter(training_times[i], f1_scores[i], label=model, s=100)
plt.xlabel('Training Times')
plt.ylabel('F1 Score')
plt.title('STD Scaling: G Means vs Training Times')
plt.legend()
plt.grid(True)


# In[50]:


# CICIDS2017 - Scaling plots

models = ['LR', 'KNN', 'DT', 'RF', 'SVM', 'XGB', 'MLP']
f1_scores = [0.9989, 0.9998 , 0.9998 , 0.9999 , 0.9996 , 0.9999, 0.9998]  
training_times= [ 2.07, 20.17, 1.48, 23.04, 10.72, 14.16, 17.53]  

plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.scatter(training_times[i], f1_scores[i], label=model, s=100)
plt.xlabel('Training Times')
plt.ylabel('F1 Score')
plt.title('QT Scaling: F1 Score vs Training Times')
plt.legend()
plt.grid(True)


# In[51]:


# CICIDS2017 - Scaling plots

models = ['LR', 'KNN', 'DT', 'RF', 'SVM', 'XGB', 'MLP']
g_means = [0.9989, 0.9998 , 0.9998 , 0.9999 , 0.9996 , 0.9999, 0.9998]  
training_times= [ 2.07, 20.17, 1.48, 23.04, 10.72, 14.16, 17.53]  

plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.scatter(training_times[i], g_means[i], label=model, s=100)
plt.xlabel('Training Times')
plt.ylabel('F1 Score')
plt.title('QT Scaling: G Means vs Training Times')
plt.legend()


# In[52]:


# Friedman test - statistical test used to detect differences in treatments across multiple scalling


from scipy.stats import friedmanchisquare

import numpy as np
from scipy.stats import friedmanchisquare

# F1 scores without scaling, STD scaling, and QT scaling
f1_data = np.array([
    [0.966, 0.9978, 0.9989],   # LR
    [0.9974, 0.9996, 0.9998], # KNN
    [0.9998, 0.9998, 0.9998], # DT
    [0.9999, 0.9999, 0.9999], # RF
    [0.9339, 0.9985, 0.9996], # SVM
    [0.9999, 0.9999, 0.9999], # XGB
    [0.9899, 0.9994, 0.9998]  # MLP
])

# G-means scores without scaling, STD scaling, and QT scaling
g_means_data = np.array([
    [0.9632, 0.9977, 0.9989], # LR
    [0.9973, 0.9995, 0.9998], # KNN
    [0.9998, 0.9998, 0.9998], # DT
    [0.9999, 0.9999, 0.9999], # RF
    [0.9317, 0.9985, 0.9997], # SVM
    [0.9999, 0.9999, 0.9999], # XGB
    [0.9904, 0.9995, 0.9998]  # MLP
])

# Function to perform the Friedman test
def perform_friedman(data):
    stat, p = friedmanchisquare(data[:, 0], data[:, 1], data[:, 2])
    alpha = 0.05
    if p > alpha:
        return "Same distributions (fail to reject H0)"
    else:
        return "Different distributions (reject H0)"

# Results



print(f"F1 scores: {perform_friedman(f1_data)}")
print(f"G-means: {perform_friedman(g_means_data)}")



# In[53]:


# G-means scores for each model using different scaling techniques
without_scaling = [0.9632, 0.9973, 0.9998, 0.9999, 0.9317, 0.9999, 0.9904]
std_scaling = [0.9977, 0.9995, 0.9998, 0.9999, 0.9985, 0.9999, 0.9995]
qt_scaling = [0.9989, 0.9998, 0.9998, 0.9999, 0.9997, 0.9999, 0.9998]

# Conduct the Friedman test
stat, p = friedmanchisquare(without_scaling, std_scaling, qt_scaling)


# ### Feature Selection

# In[60]:


# No Feature Selection

CIC2017_ALL = CIC2017.copy()


# #### Correlations

# In[61]:


# Set the color palette
#sns.set_palette("coolwarm")

# Feature Selection via Correlations

CIC2017_CR = CIC2017.copy()

CRx = CIC2017_CR.drop(columns=[' Label'])  #independent columns
CRy = CIC2017_CR[' Label']   #target column


# Identifing Correlations of each features
corre_scores = CIC2017_CR.corr()
top_corr_features = corre_scores.index
plt.figure(figsize=(18,18))

# Heat Map Plotting
g=sns.heatmap(CIC2017_CR[top_corr_features].corr(),annot=True, cmap="RdYlGn")


# In[62]:


# Calculate the correlation between each feature and the label

correlations = CIC2017_CR.corr()[" Label"].sort_values(ascending=False)


# In[63]:


# Select features between 0 - 0.7 and exclude the target itself from the selected features

selected_features = correlations[(correlations > 0) & (correlations < 0.7)]
TOP_CR_features = selected_features.drop(' Label', errors='ignore')  # Exclude the target if it's in the list


# In[64]:


# Display the ranked features and their correlation values

print(TOP_CR_features)


# In[65]:


# Create a new dataset with the selected TOP features

CIC2017_CR_T = CIC2017_CR[TOP_CR_features.index.tolist() + [' Label']]


# In[66]:


# 'NEW DATA' now contains only the top features and the target variable

CIC2017_CR_T


# In[67]:


print ('SDN Shape after dropping missing values, duplicated rows & constant value rows & after Lable Encoding',CIC2017.shape)
print ('SDN Shape after dropping missing values, duplicated rows & constant value rows & after Lable Encoding & Correlation Feature Selection',CIC2017_CR_T.shape)


# #### ANOVA F-value

# In[68]:


# Feature Selection via Information Gain

CIC2017_AF = CIC2017.copy()


# In[69]:


from sklearn.feature_selection import f_classif

# Feature Selecting via ANOVA F-value

AFx = CIC2017_AF.drop(columns=[' Label'])  #independent columns
AFy = CIC2017_AF[' Label']   #target column


# Calculate ANOVA F-values for each feature with respect to the target
f_values, _ = f_classif(AFx, AFy)

# Rank features based on F-values
ranked_features = pd.Series(f_values, index=AFx.columns).sort_values(ascending=False)
ranked_features


# In[70]:


from sklearn.feature_selection import f_classif

# Feature Selecting via ANOVA F-value

AFx = CIC2017_AF.drop(columns=[' Label'])  #independent columns
AFy = CIC2017_AF[' Label']   #target column

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


# In[71]:


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


# In[72]:


# Plot scores_list to see how accuracy changes with number of features

plt.plot(range(1, len(ranked_features) + 1), scores_list)
plt.xlabel("Number of Features")
plt.ylabel("Cross-Validation Score")
plt.show()


# In[73]:


print(f"Optimal number of features: {optimal_num_features}")
print(f"The best cross-validated score with optimal features is: {max_score:.4f}")


# In[74]:


# Create a DataFrame
AF_scores = pd.DataFrame({
    'Number of Features': range(1, len(ranked_features) + 1),
    'Average Cross-Validation Score': scores_list
})

# Display the DataFrame
print(AF_scores)


# In[75]:


# Create a new dataset with the selected TOP features

TOP_AF_features = ranked_features.index[:optimal_num_features].tolist()
CIC2017_AF_T = CIC2017_AF[TOP_AF_features + [' Label']]
CIC2017_AF_T 


# In[76]:


print ('CIC2017 Shape after dropping missing values, duplicated rows & constant value rows',CIC2017.shape)
print ('CIC2017 Shape after dropping missing values, duplicated rows & constant value rows & after AF Feature Selection',CIC2017_AF_T.shape)


# #### Information Gain

# In[77]:


# Feature Selection via Information Gain

CIC2017_IG = CIC2017.copy()


# In[78]:


# Feature Selecting via IG

IGx = CIC2017_IG.drop(columns=[' Label'])  #independent columns
IGy = CIC2017_IG[' Label']   #target column

# Calculate IG for each feature with respect to the target
ig_values = mutual_info_classif(IGx, IGy)

# Rank features based on IG values
ranked_features = pd.Series(ig_values, index=IGx.columns).sort_values(ascending=False)
ranked_features


# In[79]:


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


# In[80]:


print(f"Optimal number of features: {optimal_num_features}")
print(f"Mean Accuracy with optimal features: {max_mean_accuracy}")


# In[81]:


# Plot scores_list to see how accuracy changes with number of features

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(ranked_features) + 1), scores_list, marker='o', linestyle='-')
plt.title('Mean Accuracy vs. Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Mean Accuracy')
plt.grid(True)
plt.show()


# In[82]:


# Create a DataFrame
IG_scores = pd.DataFrame({
    'Number of Features': range(1, len(ranked_features) + 1),
    'Mean Accuracy with optimal features': scores_list
})

# Display the DataFrame
print(IG_scores)


# In[83]:


# Create a new dataset with the selected optimal features

CIC2017_IG_T = CIC2017_IG[optimal_features + [' Label']]


# In[84]:


print ('CIC2017 Shape after dropping missing values, duplicated rows & constant value rows',CIC2017.shape)
print ('CIC2017 Shape after dropping missing values, duplicated rows & constant value rows & after IG Feature Selection',CIC2017_IG_T.shape)


# #### Lasso (Least Absolute Shrinkage and Selection Operator) 

# In[85]:


CIC2017_LO = CIC2017.copy()

LOx = CIC2017_LO.drop(columns=[' Label'])  #independent columns
LOy = CIC2017_LO[' Label']   #target column


# In[86]:


# Step 1: Scale the features (Lasso regularization is sensitive to the scale of the features)

scaler = StandardScaler()
LOx_s = scaler.fit_transform(LOx)


# In[87]:


# Lasso Logistic Regression
lasso_logistic = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000)  # Using 'saga' as it's suitable for large datasets and supports L1 penalty.
lasso_logistic.fit(LOx_s, LOy)


# In[88]:


# Select important features based on non-zero coefficients

LO_features = LOx.columns[lasso_logistic.coef_[0] != 0]
LO_features


# In[89]:


num_selected_features = len(LO_features)
num_selected_features


# In[90]:


# Create new dataset with only selected features
CIC2017_LO_T = CIC2017_LO[LO_features.tolist() + [' Label']]


# In[91]:


print ('CIC2017 Shape after dropping missing values, duplicated rows & constant value rows',CIC2017.shape)
print ('CIC2017 Shape after dropping missing values, duplicated rows & constant value rows & after LO Feature Selection',CIC2017_LO_T.shape)


# #### Recursive Feature Elimination (RFE)

# In[92]:


# Feature Selection via Recursive Feature Elimination (RFE)

CIC2017_RF = CIC2017.copy()

RFx = CIC2017_RF.drop(columns=[' Label'])  #independent columns
RFy = CIC2017_RF[' Label']   #target column


# In[93]:


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


# In[94]:


# Get the optimal number of features
optimal_features_count = selector.n_features_

# Score at optimal number of features
#optimal_score = selector.grid_scores_[optimal_features_count - 1]

print(f"Optimal number of features: {optimal_features_count}")
#print(f"Score at optimal number of features: {optimal_score:.4f}")


# In[95]:


TOP_features = RFx.columns[selector.support_]
TOP_features


# In[96]:


# Plotting the cross-validation scores

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.title('Cross-validation Score as Number of Features Increases')
plt.xlabel('Number of Features Selected')
plt.ylabel('Cross-validation Score (Accuracy)')
plt.axvline(x=optimal_features_count, color='r', linestyle='--', label=f"Optimal number: {optimal_features_count}")
plt.legend()
plt.show()


# In[98]:


# Create new dataset with the selected features
CIC2017_RF_T = CIC2017_RF[TOP_features.to_list() + [' Label']]
CIC2017_RF_T


# In[99]:


print ('CIC2017 Shape after dropping missing values, duplicated rows & constant value rows',CIC2017.shape)
print ('CIC2017 Shape after dropping missing values, duplicated rows & constant value rows & after RFE Feature Selection',CIC2017_RF_T.shape)


# ### Effect of Feature Selection on Different ML Model Performance

# #### Performance Evaluation

# In[101]:


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
    'Lasso': (CIC2017_LO_T.drop(columns=[' Label']), CIC2017_LO_T[' Label']),
    'RFE': (CIC2017_RF_T.drop(columns=[' Label']), CIC2017_RF_T[' Label']),
}


# In[102]:


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


# In[103]:


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
    'Information Gain': (CIC2017_IG_T.drop(columns=[' Label']), CIC2017_IG_T[' Label']),
}


# In[104]:


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


# In[105]:


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
    'ANOVA F-value': (CIC2017_AF_T.drop(columns=[' Label']), CIC2017_AF_T[' Label']),
}


# In[106]:


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


# # Review Label Data Split

# In[ ]:


datasets = {
    'Without Feature Selection': (CIC2017.drop(columns=[' Label']), CIC2017[' Label']),
    'Correlation': (CIC2017_CR_T.drop(columns=[' Label']), CIC2017_CR_T[' Label']),
    'ANOVA F-value': (CIC2017_AF_T.drop(columns=[' Label']), CIC2017_AF_T[' Label']),
    'Information Gain': (CIC2017_IG_T.drop(columns=[' Label']), CIC2017_IG_T[' Label']),
    'Lasso': (CIC2017_LO_T.drop(columns=[' Label']), CIC2017_LO_T[' Label']),
    'RFE': (CIC2017_RF_T.drop(columns=[' Label']), CIC2017_RF_T[' Label']),
}


# In[ ]:


# Set the color palette
sns.set_palette("coolwarm")

# Visualising the Class Imbalance

CIC2017.rename(columns={' Label': 'Label'}, inplace=True)

CIC2017.Label.value_counts()
labels = [ 'Malicious', 'Benign',]
sizes = [dict(CIC2017.Label.value_counts())[0], dict(CIC2017.Label.value_counts())[1]]
plt.figure()
plt.pie(sizes, labels=labels, autopct='%1.1f%%',# Modelling - KNN Classifier
        shadow=True, startangle=80,)
plt.legend([ 'Malicious', 'Benign',])
plt.title('% of Benign and Maliciuos requests in the CIC2017 Dataset')
plt.show()


# # Modelling - LR Classifier

# ## Modelling with AF Feature Selection & ST Scaling

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
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
    data[' Label'] = le.fit_transform(data[' Label'])
    categorical_cols = data.select_dtypes(['object']).columns.to_list()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    X = data.drop(' Label', axis=1)
    y = data[' Label']
    return X, y

# Load and preprocess the dataset
data = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
X, y = preprocess_data(data)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use SelectKBest with ANOVA F-statistic to select features
selector = SelectKBest(f_classif, k='all')  # Use k='all' to calculate scores for all features
selector.fit(X_train, y_train)
X_train_anova = selector.transform(X_train)
X_test_anova = selector.transform(X_test)

# Define the hyperparameters and their possible values
param_grid = {
    'C': np.logspace(-4, 4, 10),
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'penalty': ['l1', 'l2', 'elasticnet', 'none']
}

# Define a StratifiedKFold instance with 10 splits
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Function to train and evaluate the model
def train_and_evaluate(X_train, X_test, y_train, y_test, param_search):
    # Hyperparameter optimization using random search
    random_search = RandomizedSearchCV(LogisticRegression(max_iter=1000), param_distributions=param_grid, n_iter=10, cv=skf, n_jobs=-1)
    random_search.fit(X_train, y_train)
    best_random_params = random_search.best_params_

    # Hyperparameter optimization using grid search
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid=param_grid, cv=skf, n_jobs=-1)
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

# Results for ANOVA-based feature selection
results_anova = train_and_evaluate(X_train_anova, X_test_anova, y_train, y_test, param_search=param_grid)
print("Results for ANOVA-based feature selection:", results_anova)


# # XGB : Modelling with Correlation Feature Selection and STD Scaler

# In[116]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
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

data = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
data.drop_duplicates(inplace=True)
data.fillna(data.mean(numeric_only=True), inplace=True)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
le = LabelEncoder()
data['Label'] = le.fit_transform(data[' Label'])
categorical_cols = data.select_dtypes(['object']).columns.to_list()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])
X = data.drop('Label', axis=1)
y = data['Label']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Select features with correlation less than 0.7 threshold
corr_matrix = pd.DataFrame(X_train).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
selected_features = [column for column in upper.columns if any(upper[column] < 0.7)]
X_train = X_train[:, selected_features]
X_val = X_val[:, selected_features]
X_test = X_test[:, selected_features]

param_grid = {
    'n_estimators': [10, 50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_child_weight': [1, 2, 4],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0]
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

start_time_train = time()
random_search = RandomizedSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_distributions=param_grid, n_iter=20, cv=skf, n_jobs=-1)
random_search.fit(X_train, y_train)
end_time_train = time() - start_time_train

best_random_params = random_search.best_params_

refined_param_grid = {f'{param}': [value-1, value, value+1] for param, value in best_random_params.items() if isinstance(value, int)}
for param, value in best_random_params.items():
    if f'{param}' not in refined_param_grid:
        refined_param_grid[f'{param}'] = [value]

start_time_val = time()
grid_search = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid=refined_param_grid, cv=skf, n_jobs=-1)
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
    'Model': 'XGBoost',
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
plt.bar(np.array(X.columns[selected_features])[sorted_indices], feature_importances[sorted_indices])
plt.xticks(rotation=90)
plt.show()

print("Selected Features:", np.array(X.columns[selected_features])[sorted_indices[:10]])


# # XGB : Modelling with STD Scaler

# In[115]:


# XGB : Modelling with Correlation Feature Selection and STD Scalerimport pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
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

data = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
data.drop_duplicates(inplace=True)
data.fillna(data.mean(numeric_only=True), inplace=True)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
le = LabelEncoder()
data[' Label'] = le.fit_transform(data[' Label'])
categorical_cols = data.select_dtypes(['object']).columns.to_list()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])
X = data.drop(' Label', axis=1)
y = data[' Label']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

param_grid = {
    'n_estimators': [10, 50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_child_weight': [1, 2, 4],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0]
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

start_time_train = time()
random_search = RandomizedSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_distributions=param_grid, n_iter=20, cv=skf, n_jobs=-1)
random_search.fit(X_train, y_train)
end_time_train = time() - start_time_train

best_random_params = random_search.best_params_

refined_param_grid = {f'{param}': [value-1, value, value+1] for param, value in best_random_params.items() if isinstance(value, int)}
for param, value in best_random_params.items():
    if f'{param}' not in refined_param_grid:
        refined_param_grid[f'{param}'] = [value]

start_time_val = time()
grid_search = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid=refined_param_grid, cv=skf, n_jobs=-1)
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
    'Model': 'XGBoost',
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


# # XGB : Modelling without Feature Selection or  STD Scaler

# In[114]:


# XGB : Modelling with STD Scalerimport pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
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

data = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
data.drop_duplicates(inplace=True)
data.fillna(data.mean(numeric_only=True), inplace=True)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
le = LabelEncoder()
data[' Label'] = le.fit_transform(data[' Label'])
categorical_cols = data.select_dtypes(['object']).columns.to_list()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])
X = data.drop(' Label', axis=1)
y = data[' Label']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)

param_grid = {
    'n_estimators': [10, 50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_child_weight': [1, 2, 4],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0]
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

start_time_train = time()
random_search = RandomizedSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_distributions=param_grid, n_iter=20, cv=skf, n_jobs=-1)
random_search.fit(X_train, y_train)
end_time_train = time() - start_time_train

best_random_params = random_search.best_params_

refined_param_grid = {f'{param}': [value-1, value, value+1] for param, value in best_random_params.items() if isinstance(value, int)}
for param, value in best_random_params.items():
    if f'{param}' not in refined_param_grid:
        refined_param_grid[f'{param}'] = [value]

start_time_val = time()
grid_search = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid=refined_param_grid, cv=skf, n_jobs=-1)
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
    'Model': 'XGBoost',
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


Selection 


# # DT : Modelling with IG Feature Selection and STD Scaler

# In[117]:


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

data = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
data.drop_duplicates(inplace=True)
data.fillna(data.mean(numeric_only=True), inplace=True)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
le = LabelEncoder()
data[' Label'] = le.fit_transform(data[' Label'])
categorical_cols = data.select_dtypes(['object']).columns.to_list()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])
X = data.drop(' Label', axis=1)
y = data[' Label']

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


# In[118]:


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

data = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
data.drop_duplicates(inplace=True)
data.fillna(data.mean(numeric_only=True), inplace=True)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
le = LabelEncoder()
data[' Label'] = le.fit_transform(data[' Label'])
categorical_cols = data.select_dtypes(['object']).columns.to_list()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])
X = data.drop(' Label', axis=1)
y = data[' Label']

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


# In[119]:


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

data = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
data.drop_duplicates(inplace=True)
data.fillna(data.mean(numeric_only=True), inplace=True)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
le = LabelEncoder()
data[' Label'] = le.fit_transform(data[' Label'])
categorical_cols = data.select_dtypes(['object']).columns.to_list()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])
X = data.drop(' Label', axis=1)
y = data[' Label']

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




