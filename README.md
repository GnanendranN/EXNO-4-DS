# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
  
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/user-attachments/assets/98da6724-0b66-449f-a09c-c43bf24850a0)

```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/3d5ecc2b-ee1d-438f-b48b-43f2ce0f1840)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/d7069ac9-c7c9-4d30-ba6d-ed5a24056770)

```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/1f5ddb34-1dbb-4fce-b7e1-130b41e32210)

```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/977ad856-0d91-4633-bdc2-a6b6a1a0cc97)

```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/c4a5b36a-258b-442c-a432-2add666bc69e)

```
data2
```
![image](https://github.com/user-attachments/assets/c85d99a0-cab7-45c2-aeca-e221978d8855)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/0eac7cea-1493-4a5f-83c3-6a0ef258483b)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/c7e6db5a-8181-4f3e-9a56-718c8d46c5c3)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/f9c4f8d7-426b-4c6d-ad3e-ccca73c8cd08)

```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/5eecf7f7-9eda-4810-bf2b-47b28c3b84aa)

```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/eca7f8e7-f22f-4e70-89cc-0898122bd68e)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
     
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
     
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/a84d0ed5-18ca-4270-b0ea-efa01acb069b)

```
prediction=KNN_classifier.predict(test_x)
    
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/d4b8c3e2-e21c-480d-86c8-155fb9c99def)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/fa6cbdd2-68d3-44d8-bf17-1568649a6706)

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/user-attachments/assets/01f37d8c-7d05-4ef2-ae5c-1ba9209f6bf5)

```
data.shape
```
![image](https://github.com/user-attachments/assets/7ac142da-38a8-4f76-8dd3-bb664857c780)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
  
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/9f0fca4f-cc25-40f5-8503-7ce88570df00)

```
tips.time.unique()
```
![image](https://github.com/user-attachments/assets/4c51f3be-bace-45e0-9bc5-c8b6160a3704)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/a67732ec-1198-44bb-920e-58fa181d58ba)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/user-attachments/assets/8a2c80e6-e287-48c3-a001-2b6a9d93bfce)

# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
