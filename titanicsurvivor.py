# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #statistical data plotting
import re
from matplotlib import pyplot as plt
from matplotlib import style

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
train_dataset = pd.read_csv("../input/train.csv")

test_dataset = pd.read_csv("../input/test.csv")

# total = train_dataset.isnull().sum().sort_values(ascending=False)
# percent_1 = (train_dataset.isnull().sum()/train_dataset.isnull().count())*100
# percent_2 = round(percent_1,1).sort_values(ascending=False)
# train_dataset.head(8)
# train_dataset.columns.values
train_dataset.columns.values
survived = 'survived'
not_survived = 'not survived'

# fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(15,6))
women = train_dataset[train_dataset['Sex']=='female']
men = train_dataset[train_dataset['Sex']=='male']

# ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
# ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
# ax.legend()
# ax.set_title('Female')

# ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde =False)
# ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde =False)
# ax.legend()
# ax.set_title('Male')
# bins=14 for women

# Any results you write to the current directory are saved as output.
# FacetGrid = sns.FacetGrid(train_dataset, row='Embarked', size=4.5, aspect=1.5)
# FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
# FacetGrid.add_legend()

# sns.barplot(x='Pclass', y='Survived', data=train_dataset)
data = [train_dataset,test_dataset]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
train_dataset['not_alone'].value_counts()
# axes = sns.factorplot('relatives','Survived', data=train_dataset, aspect = 2.5, )

train_dataset = train_dataset.drop(['PassengerId'],axis=1)
train_dataset.columns.values
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train_dataset, test_dataset]

for dataset in data:
    dataset['Cabin'] = 