# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 21:29:08 2019

@author: mehta
"""

# importing the required modules
import pandas as pd # for dataframes
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs
import datetime as dt
import numpy as np
import time

data = pd.read_excel("Online_Retail.xlsx")
data.head()

filtered_data=data[['Country','CustomerID']].drop_duplicates()

#Top ten country's customer
filtered_data.Country.value_counts()[:10].plot(kind='bar')

uk_data=data[data.Country=='United Kingdom']
uk_data.info()

uk_data.describe()

uk_data = uk_data[(uk_data['Quantity']>0)]
uk_data.info()

uk_data=uk_data[['CustomerID','InvoiceDate','InvoiceNo','Quantity','UnitPrice']]

#Calulate total purchase
uk_data['TotalPurchase'] = uk_data['Quantity'] * uk_data['UnitPrice']


uk_data_group=uk_data.groupby('CustomerID').agg({'InvoiceDate': lambda date: (date.max() - date.min()).days,
                                        'InvoiceNo': lambda num: len(num),
                                        'Quantity': lambda quant: quant.sum(),
                                        'TotalPurchase': lambda price: price.sum()})

uk_data_group.head()


