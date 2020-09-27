# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 22:55:05 2020

@author: Nikolai
"""

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups


groups = fetch_20newsgroups()
print(groups.keys())
sns.distplot(groups.target)

print(groups['target_names'])

plt.show()