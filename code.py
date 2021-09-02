#!/usr/bin/env python
# coding: utf-8




# data manipulate 
import numpy as np
import pandas as pd 
# data visualization
import matplotlib.pyplot as plt 
import seaborn as sas 
import klib #==> new model for visualization
# for interactivity
from ipywidgets import interact




# pandas (all lowercase) is a popular Python-based data analysis toolkit which can be imported using import pandas as pd . It presents a diverse range of utilities, ranging from parsing multiple file formats to converting an entire data table into a NumPy matrix array
#lets read the dataset
data = pd.read_csv("data.csv")




# lets cheak teh shape of the dataset
print("shape of the datset:" ,   data.shape)





data.head() #head 5 

data.tail() #last 5 


# #Pandas isnull() and notnull() methods are used to check and manage NULL values in a data frame.


# let find missing value 
data.isnull().sum()


# - klib.describe - functions for visualizing datasets
# - klib.cat_plot(df) # returns a visualization of the number and frequency of categorical features
# - klib.corr_mat(df) # returns a color-encoded correlation matrix
# - klib.corr_plot(df) # returns a color-encoded heatmap, ideal for correlations
# - klib.dist_plot(df) # returns a distribution plot for every numeric feature
# - klib.missingval_plot(df) # returns a figure containing information about missing values


klib.cat_plot(data) # returns a visualization of the number and frequency of categorical features


klib.corr_mat(data) # returns a color-encoded correlation matrix


klib.corr_plot(data) # returns a color-encoded heatmap, ideal for correlations


klib.dist_plot(data) # returns a distribution plot for every numeric feature


klib.missingval_plot(data)# returns a figure containing information about missing values


#let cheak the crops present in this dataset 
data["label"].value_counts()


# The format() method formats the specified value(s) and insert them inside the string's placeholder.
# 
# The placeholder is defined using curly brackets: {}. Read more about the placeholders in the Placeholder section below.
# 
# The format() method returns the formatted string


# lets check the summary for all the crops 
#	Nitrogen	phosphorous	---potassium	temperature	humidity	ph	rainfall
print('==> Avg Ratio of Nitrogen in the soil :{0:.2f}'.format(data["N"].mean()))
print('==> Avg Ratio of phosphorous in the soil :{0:.2f}'.format(data["P"].mean()))
print('==> Avg Ratio of potassium in the soil :{0:.2f}'.format(data["K"].mean()))
print('==> Avg Tempature in celsius :{0:.2f}'.format(data["temperature"].mean()))
print('==> Avg Relative humidity :{0:.2f}'.format(data["humidity"].mean()))
print('==> Avg ph value of the soil :{0:.2f}'.format(data["ph"].mean()))
print('==> Avg RailFall in nm :{0:.2f}'.format(data["rainfall"].mean()))


# @interact  automatically creates user interface (UI) controls for exploring code and data interactively. It is the easiest way to get started using IPython's widgets.

# Mean =  in the data set in have numerical value(1,2,5,9) then replace the missing value helping through mean
# Median = The mean value of numerical data is without a doubt the most commonly used statistical measure.
#          Outlier Analysis is a data mining task which is referred to as an “outlier mining”
# mode  = in the data set in have categorical value(age , time etc) then replace the missing value helping through mean


#let cheak the summary statistics of the crops 
#	Nitrogen	phosphorous	---potassium	temperature	humidity	ph	rainfall

@interact
def summary (crops = list(data["label"].value_counts().index)):
    x = data[data["label"] == crops ]
    print("-----------------------------------------------")
    print("=============statistics for Nitrogen===========")
    print("Minimun Nitrogen reqired:", x["N"].min())
    print("avg Nitrogen reqired:", x["N"].mean())
    print("Maximun Nitrogen reqired:", x["N"].max())
    print("-----------------------------------------------")
    print("=============statistics for phosphorous  ===========")
    print("Minimun phosphorous  reqired:", x["P"].min())
    print("avg phosphorous  reqired:", x["P"].mean())
    print("Maximun phosphorous  reqired:", x["P"].max())
    print("-----------------------------------------------")
    print("=============statistics for potassium  ===========")
    print("Minimun potassium  reqired:", x["K"].min())
    print("avg potassium  reqired:", x["K"].mean())
    print("Maximun potassium  reqired:", x["K"].max())
    print("-----------------------------------------------")
    print("=============statistics for temperature  ===========")
    print('==> Minimun temperature reqired :{0:.2f}'.format(x["temperature"].min()))
    print('==> Avg temperature reqired :{0:.2f}'.format(x["temperature"].mean()))
    print('==> Maximun temperature reqired :{0:.2f}'.format(x["temperature"].max()))
    print("-----------------------------------------------")
    print("=============statistics for humidity  ===========")
    print('==> Minimun  humidity reqired :{0:.2f}'.format(x["humidity"].min()))
    print('==> Avg humidity reqired :{0:.2f}'.format(x["humidity"].mean()))
    print('==> Maximun humidity reqired :{0:.2f}'.format(x["humidity"].max()))
    print("-----------------------------------------------")
    print("=============statistics for ph  ===========")
    print('==> Minimun ph reqired :{0:.2f}'.format(x["ph"].min()))
    print('==> Avg ph reqired :{0:.2f}'.format(x["ph"].mean()))
    print('==> Maximun ph reqired :{0:.2f}'.format(x["ph"].max()))
    print("-----------------------------------------------")
    print("=============statistics for rainfall  ===========")
    print('==> Minimun rainfall reqired :{0:.2f}'.format(x["rainfall"].min()))
    print('==> Avg rainfall reqired :{0:.2f}'.format(x["rainfall"].mean()))
    print('==> Maximun rainfull reqired :{0:.2f}'.format(x["rainfall"].max()))


#lets compere the avg requirement for each each avgrage conditions
#	Nitrogen	phosphorous	---potassium	temperature	humidity	ph	rainfall

@interact
def compare (conditions = ["N" , "P" , "K" , "temperature" , "humidity" , "ph" , "rainfall"]):
    print('avg value for ' , conditions , 'is {0:2f}'.format(data[conditions].mean()))
    print("------------------------------------------------------------------")
    print("grapes : {0:2f}".format(data[(data["label"] == "grapes")][conditions].mean()))
    print("muskmelon : {0:2f}".format(data[(data["label"] == "muskmelon")][conditions].mean()))
    print("kidneybeans : {0:2f}".format(data[(data["label"] == "kidneybeans")][conditions].mean()))
    print("apple : {0:2f}".format(data[(data["label"] == "apple")][conditions].mean()))
    print("lentil : {0:2f}".format(data[(data["label"] == "lentil")][conditions].mean()))
    print("maize : {0:2f}".format(data[(data["label"] == "maize")][conditions].mean()))
    print("jute : {0:2f}".format(data[(data["label"] == "jute")][conditions].mean()))
    print("Rice : {0:2f}".format(data[(data["label"] == "rice")][conditions].mean()))
    print("coffee : {0:2f}".format(data[(data["label"] == "coffee")][conditions].mean()))
    print("mothbeans : {0:2f}".format(data[(data["label"] == "mothbeans")][conditions].mean()))
    print("mango : {0:2f}".format(data[(data["label"] == "mango")][conditions].mean()))
    print("pomegranate : {0:2f}".format(data[(data["label"] == "pomegranate")][conditions].mean()))
    print("chickpea : {0:2f}".format(data[(data["label"] == "chickpea")][conditions].mean()))
    print("coconut : {0:2f}".format(data[(data["label"] == "coconut")][conditions].mean()))
    print("banana : {0:2f}".format(data[(data["label"] == "rice")][conditions].mean()))
    print("pigeonpeas : {0:2f}".format(data[(data["label"] == "pigeonpeas")][conditions].mean()))
    print("orange : {0:2f}".format(data[(data["label"] == "orange")][conditions].mean()))
    print("blackgram : {0:2f}".format(data[(data["label"] == "blackgram")][conditions].mean()))
    print("cotton : {0:2f}".format(data[(data["label"] == "cotton")][conditions].mean()))
    print("mungbean : {0:2f}".format(data[(data["label"] == "mungbean")][conditions].mean()))
    print("watermelon : {0:2f}".format(data[(data["label"] == "watermelon")][conditions].mean()))


#let make this function more intvitive :==>
# in below avg and above agv conditions :==>
@interact
def compare (conditions = ["N" , "P" , "K" , "temperature" , "humidity" , "ph" , "rainfall"]):
    
    print("Crops which require greater than avg", conditions,"\n") 
    print(data[data[conditions]>data[conditions].mean()]["label"].unique())
    print("--------------------------------------------------------------")
    print("Crops which require less than avg", conditions,"\n") 
    print(data[data[conditions]<=data[conditions].mean()]["label"].unique())


#seabron  #displot distribution plot function ?
##	Nitrogen	phosphorous	---potassium	temperature	humidity	ph	rainfall


plt.subplot(2, 4, 1)
sas.distplot(data["N"],color = "darkblue")
plt.xlabel("Ratio of Nitrogen" , fontsize = 12)
plt.grid()

plt.subplot(2, 4, 2)
sas.distplot(data["P"],color = "black")
plt.xlabel("Ratio of phosphorous" , fontsize = 12)
plt.grid()

plt.subplot(2, 4, 3)
sas.distplot(data["K"],color = "grey")
plt.xlabel("Ratio of potassium" , fontsize = 12)
plt.grid()


plt.subplot(2, 4, 4)
sas.distplot(data["temperature"],color = "lightgreen")
plt.xlabel("Ratio of temperature" , fontsize = 12)
plt.grid()

plt.subplot(2, 4, 5)
sas.distplot(data["humidity"],color = "darkgreen")
plt.xlabel("Ratio of humidity" , fontsize = 12)
plt.grid()


plt.subplot(2, 4, 6)
sas.distplot(data["ph"],color = "pink")
plt.xlabel("Ratio of ph" , fontsize = 12)
plt.grid()


plt.subplot(2, 4, 7)
sas.distplot(data["rainfall"],color = "lightgrey")
plt.xlabel("Ratio of rainfall" , fontsize = 12)
plt.grid()


plt.suptitle("distribution Agricultural conditions" , fontsize=20)
plt.show()


#lets find out some interestings facts
##	Nitrogen	phosphorous	---potassium	temperature	humidity	ph	rainfall

print("some Intersting patterns")
print("==========================")
print("crops which requires very high Ratio of Nitrogen content is soil:", data[data["N"]>120]["label"].unique())
print("crops which requires very high Ratio of phosphorous content is soil:", data[data["P"]>100]["label"].unique())
print("crops which requires very high Ratio of potassium content is soil:", data[data["K"]>200]["label"].unique())
print("crops which requires very high rainfall:", data[data["rainfall"]>200]["label"].unique())
print("crops which requires very low temperature :", data[data["temperature"]<10]["label"].unique())
print("crops which requires very high  temperature: ", data[data["temperature"]>40]["label"].unique())
print("crops which requires very low humidity :", data[data["humidity"]<20]["label"].unique())
print("crops which requires very low ph:", data[data["ph"]<4]["label"].unique())
print("crops which requires very high ph:", data[data["ph"]>9]["label"].unique())



##  lets understand which crops can only be grown in summer  , winter , Rainy 
print("====================summer crops======================")

print(data[(data["temperature"]>30) & (data["humidity"]>50)]["label"].unique())

print("=====================winter crops=====================")
print(data[(data["temperature"]<20) & (data["humidity"]>30)]["label"].unique())

print("======================Rainy crops======================")
print(data[(data["rainfall"]>200) & (data["humidity"]>30)]["label"].unique())



#clustering analysis +> used to classication      data point into realvite groups 
#that mean we assing samely data aasing one group


from sklearn.cluster import KMeans

# in unsupervised learning we do not have need labels
#removing the labels colunm
x = data.drop(["label"],axis=1)

#selecting all the value of the data
x = x.values

print(x.shape)


# What is elbow method in K-means?
# 
# Image result for elbow method k means
# The elbow method runs k-means clustering on the dataset for a range of values for k (say from 1-10) and then for each value of k computes an average score for all clusters. By default, the distortion score is computed, the sum of square distances from each point to its assigned center.



#In clustering analysis 1th we knows how many cluster we have
#so knowing cluster help elbow method



# lets determine the optimun number of clusters within dataset 
plt.rcParams["figure.figsize"]=(10,4)
wcss = []
for i in range(1,11):
    km = KMeans(n_clusters = i , init = "k-means++" , max_iter = 300 , n_init = 10 , random_state = 0)
    km.fit(x)
    wcss.append(km.inertia_)
#lets plot the results 
plt.plot(range(1,11),wcss)
plt.title("The Elbow method" , fontsize = 20)
plt.xlabel("No of clusters")
plt.ylabel("wcss")
plt.show()


# lets implement the k means algorithm to perfron clustering analysis
km = KMeans(n_clusters = 4 , init = "k-means++" , max_iter = 300 , n_init = 10 , random_state = 0)
y_means = km.fit_predict(x)

#lets find out the results
a = data["label"]
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means,a],axis= 1)
z = z.rename(columns = {0:"cluster"})


#lets cheak the clusters of each crops
print("lets check results the k means algorithm to perfron clustering analysis \n ")

print("Crops in frist cluster:", z[z["cluster"]==0]["label"].unique())
print("=============================================================")
print("Crops in frist cluster:" ,z[z["cluster"]==1]["label"].unique())
print("=============================================================")

print("Crops in frist cluster:" ,z[z["cluster"]==2]["label"].unique())
print("=============================================================")

print("Crops in frist cluster:" ,z[z["cluster"]==3]["label"].unique())


# #prediction ==> uesd machine learning mobel 
# #predictive modeline is part of data set used dataset and make a partten of model 
# #finaly model train so used  model to predicition of unseen data 
# #machine learning mobel used dataset and trend and partten to  make dision rules to predicition finaly rejeltt


#machine learning mobel used  ==> logical reasoning


# lets split tge adtaset for predictive modeling 
y = data["label"]
x = data.drop(["label"],axis = 1)

print("shape of x:",x.shape)
print("shape of y:",y.shape)

from sklearn.model_selection import train_test_split
x_train , x_test ,  y_train , y_test =  train_test_split(x,y, test_size = 0.2 , random_state = 0)
print("the shape x train:",x_train.shape)
print("the shape x train:",x_test.shape)
print("the shape y train:",y_train.shape)
print("the shape y train:",y_test.shape)


# lets created a predicitive model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train , y_train)

y_pred = model.predict(x_test)


# letts evalute tge model performace 
from sklearn.metrics import confusion_matrix

#lets prit yhe confusion motrix frist
plt.rcParams["figure.figsize"]  =  (10,10)
cm = confusion_matrix(y_test ,y_pred)
sas.heatmap(cm , annot = True , cmap = "Wistia" )
plt.title("confusion maxrtix for logistic regression" , fontsize = 15)


# letts print the classifixcation report also
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)


data.head()


predicition = model.predict((np.array([[90,40,40,20,80,7,200]])))
print("the suggested crop for given climatic condition id :", predicition)


############################################################################### "THANK YOU FOR VISITING" ##############################################################################
