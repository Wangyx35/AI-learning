# -*- coding: utf-8 -*-

# In[]:
import pandas as pd   # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np    # linear algebra
from sklearn.preprocessing import StandardScaler

# import libraries for plotting

import matplotlib.pyplot as plt
import seaborn as sns
#  %matplotlib inline  直接在你的python console里面生成图像

#添加属性名称 读取train 和 test 并删除test第一行
columns = ['Age','Workclass','fnlgwt','Education','Education num','Marital Status',
           'Occupation','Relationship','Race','Sex','Capital Gain','Capital Loss',
           'Hours/Week','Native country','Income']

train = pd.read_csv("adult_train.csv",names=columns)

test = pd.read_csv("adult_test.csv",names=columns,skiprows=1)

# In[]:显示数据的类型
print(train.shape)

train.info()

test.info()

# In[]:Cleaning data Some cells contain ' ?', we convert them to NaN
train.replace(' ?', np.nan, inplace=True)
test.replace(' ?', np.nan, inplace=True)

# In[]：数据预处理 - Income
train['Income'] = train['Income'].apply(lambda x: 1 if x==' >50K' else 0)
test['Income'] = test['Income'].apply(lambda x: 1 if x==' >50K.' else 0)

# In[]：数据预处理 - Age
plt.hist(train['Age']);

# In[]：数据预处理 - Workclass
# replace empty with 0 and check how data plot looks like.
train['Workclass'].fillna(' 0', inplace=True)
test['Workclass'].fillna(' 0', inplace=True)
# In[]：
train['Workclass'].value_counts()
# In[]：
sns.catplot(x="Workclass", y="Income", data=train, kind="bar", height = 6, 
palette = "muted")
plt.xticks(rotation=45);
# In[]：

#合并
train['Workclass'].replace(' Without-pay', ' Never-worked', inplace=True)
test['Workclass'].replace(' Without-pay', ' Never-worked', inplace=True)
sns.catplot(x="Workclass", y="Income", data=train, kind="bar", height = 6, 
palette = "muted")
plt.xticks(rotation=45);


# In[]：数据预处理 -fnlgwt  无意义 删掉
train['fnlgwt'].value_counts()
train.drop('fnlgwt',axis=1,inplace=True)
test.drop('fnlgwt',axis=1,inplace=True)


# In[]：数据预处理 -Education
sns.factorplot(x="Education",y="Income",data=train,kind="bar", size = 7, 
palette = "muted")
plt.xticks(rotation=45);

def primary(x):
    if x in [' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' 10th', ' 11th', ' 12th']:
        return ' Primary'
    else:
        return x
    
train['Education'] = train['Education'].apply(primary)
test['Education'] = test['Education'].apply(primary)
sns.factorplot(x="Education",y="Income",data=train,kind="bar", size = 6, 
palette = "muted")
plt.xticks(rotation=60);


# In[]：数据预处理 Education num 删除
sns.factorplot(x="Education num",y="Income",data=train,kind="bar", size = 6, 
palette = "muted")
plt.xticks(rotation=60);

train.drop('Education num',axis=1,inplace=True)
test.drop('Education num',axis=1,inplace=True)

# In[]：数据预处理  Marital Status
sns.factorplot(x="Marital Status",y="Income",data=train,kind="bar", size = 5, 
palette = "muted")
plt.xticks(rotation=60);

#已婚夫妇的特征很少。它们类似于已婚公民的配偶，所以我们可以合并它们。
train['Marital Status'].replace(' Married-AF-spouse', ' Married-civ-spouse', inplace=True)
test['Marital Status'].replace(' Married-AF-spouse', ' Married-civ-spouse', inplace=True)
sns.factorplot(x="Marital Status",y="Income",data=train,kind="bar", size = 6, 
palette = "muted")
plt.xticks(rotation=60);

train['Marital Status'].value_counts()

# In[]：数据预处理  Occupation
train['Occupation'].fillna(' 0', inplace=True)
test['Occupation'].fillna(' 0', inplace=True)
sns.factorplot(x="Occupation",y="Income",data=train,kind="bar", size = 8, 
palette = "muted")
plt.xticks(rotation=60);

# Armed-Forces这个类异常，我的处理方式是全部用0代替。

train['Occupation'].replace(' Armed-Forces', ' 0', inplace=True)
test['Occupation'].replace(' Armed-Forces', ' 0', inplace=True)
sns.factorplot(x="Occupation",y="Income",data=train,kind="bar", size = 8, 
palette = "muted")
plt.xticks(rotation=60);

# In[]：数据预处理  Relationship
sns.factorplot(x="Relationship",y="Income",data=train,kind="bar", size = 6, 
palette = "muted")
plt.xticks(rotation=60);

train['Relationship'].value_counts()



# In[]：数据预处理 Race
sns.factorplot(x="Race",y="Income",data=train,kind="bar", size = 6, 
palette = "muted")
plt.xticks(rotation=45);
train['Race'].value_counts()


# In[]：#  One-hot encoding
joint = pd.concat([train, test], axis=0)
joint.dtypes
categorical_features = joint.select_dtypes(include=['object']).axes[1]

for col in categorical_features:
    print (col, joint[col].nunique())
#one-hot encode
for col in categorical_features:
    joint = pd.concat([joint, pd.get_dummies(joint[col], prefix=col, prefix_sep=':')], axis=1)
    joint.drop(col, axis=1, inplace=True)
    
joint.head()

train = joint.head(train.shape[0])
test = joint.tail(test.shape[0])

# In[]：

Xtrain = train.drop('Income', axis=1)
Ttrain = train['Income']

Xtest = test.drop('Income', axis=1)
Ttest = test['Income']

scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)


# In[]：
from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score



clf = XGBClassifier(max_depth =8,min_child_weight=3,subsample=0.8)
clf.fit(Xtrain,Ttrain)
fig,ax = plt.subplots(figsize=(15,15))
plot_importance(clf,height=0.5,ax=ax,max_num_features=64)
plt.show()
result = clf.predict(Xtest)
print(accuracy_score(Ttest, result))
# In[]：绘制混淆矩阵
import itertools
from sklearn.metrics import confusion_matrix

def plt_confusion_matirx(cm, classes, title = "Confusion Matrix",cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation="nearest", cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max()
    for i , j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j ,i , cm[i, j], horizontalalignment="center", 
                 color= "white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.xlabel("Predicted Classification")
    plt.ylabel("True Classification")

cm = confusion_matrix(Ttest, result)
class_names=[0, 1]
plt.figure()
plt_confusion_matirx(cm, classes=class_names, title="Confusion Matrix",cmap = plt.cm.Blues)
plt.show()


