
# PythonProjects

## Bezant-Assignment
### Problem Statement


**Objective:** 
To train, validate and test a Machine Learning model that can predict the direction of Bitcoin’s price. 

**Data:** 
labels.csv: this dataset contains the labels having values in [0, 1], and Bitcoin open, close, high, low prices and volumes using 1 hour intervals. Note that we transformed the continuous price data using 300 min time horizons into two different classes:
Class 0: Negative price action. The price dropped compared to the price at the start of the window.
Class 1: Positive price action. The price increased compared to the price at the start of the window.
features.csv: this dataset contains the features that you will use to predict the labels. The data contains 72 columns.

### Solution 

**Prediction Using Convolutional Neural Network**

The idea is fairly simple: Calculate imp features  with 8 different period lengths for each hour in your trading data. Then convert the 64 (8*8) new features into 15x15 images. Label the data as class0/class1 based the algorithm provided in the [paper](https://www.researchgate.net/publication/324802031_Algorithmic_Financial_Trading_with_Deep_Convolutional_Neural_Networks_Time_Series_to_Image_Conversion_Approach). Then train a Convolutional Neural Network like any other image classification problem.



**Computational Performance Evaluation:**  
Authors have provided two types of model evaluations in the paper, computational and financial evaluation. Computational evaluation includes confusion matrix, F1 score, class wise precision etc. 

**Implementation**

I have not followed the research paper strictly because it didn’t produce expected results. I will mention the differences as and when they come up. But with the changes I made the result was at par with the paper or better in some cases.



**Normalization:** 
I used MinMaxScaler from Sklearn to normalize the data in the range of [0, 1], although the paper used [-1, 1] range (second deviation). This is just a personal preference.

![alt text](https://github.com/nishantsikarwar/PythonProjects/blob/main/images/Screenshot%202022-11-22%20at%2000.48.42.png)

**Feature Selection:**  
After calculating these indicators, grouping them in the image based on their types (momentum, oscillator, etc), and training many CNN architectures, I realized the model just isn’t learning enough. Maybe the features weren’t good enough. So I decided to go with many other indicators without strictly following the rule of calculating them with different periods. Then I used feature selection technique to chose 64 high-quality features. In fact, I used two feature selection methods f_classif and mutual_info_classif and chose the common features from both of their results. 

![alt text](https://github.com/nishantsikarwar/PythonProjects/blob/main/images/Screenshot%202022-11-22%20at%2000.56.09.png)


In the end, I am sorting the indices list and found the intersection of f_classif and mutual_info_classif. This is to ensure that related features are in close proximity in the image since I had appended similar types of indicators closely. Feature selection significantly improved the performance of the model.

**Reshaping the data as an image:** 
 As of now, we have tabular data with 64 features. We need to convert it as images like this:

![alt text](https://github.com/nishantsikarwar/PythonProjects/blob/main/images/Screenshot%202022-11-22%20at%2000.58.50.png))
 
**Handling Class Imbalance:** 
The labelling algorithm presented in the paper produces a somewhat generous number of buy/sell instances as analogous to class 0 | class 1. Any other real-world strategy would produce much fewer instances.

![alt text](https://github.com/nishantsikarwar/PythonProjects/blob/main/images/Screenshot%202022-11-22%20at%2001.00.32.png)


This is less for the model to learn anything significant. The paper mentions only “resampling” as a way of tackling this problem. I tried oversampling and synthetic data generation (SMOTE, ADASYN) but none of them gave any satisfactory results. Finally, I settled for “sample weights”, wherein you tell the model to pay more attention to some samples (fourth deviation). This comes in handy while dealing with class imbalance. Here is how you can calculate sample weight:

This array of sample weights is then passed to Keras ‘fit’ function. You can also look into the ‘class_weights’ parameter.

![alt text](https://github.com/nishantsikarwar/PythonProjects/blob/main/images/Screenshot%202022-11-22%20at%2001.02.13.png)

**Training:**  
The model architecture mentioned in the paper had some missing points. For example, they didn’t mention the strides they had used.  I had no luck with sliding window training, no matter how small a network I used. So I trained with full training data with cross-validation (fifth deviation). But I have included the code for sliding/rolling window training in the project. So, I used a very similar model with small differences like dropouts etc. This is the model I trained with (I have not tried extensive hyper-parameter tuning):

Keras model training was done with EarlyStopping and ReduceLROnPlateau callbacks like this:

![alt text](https://github.com/nishantsikarwar/PythonProjects/blob/main/images/Screenshot%202022-11-22%20at%2001.05.07.png)


As you can see above, I have used F1 score as a metric. For test data evaluation, I have also used the confusion matrix, Sklearn’s weighted F1 score and Kappa (which I got to know about recently and have to dig deeper).


This result somewhat varies every time I run it, possibly due to Keras weight initialisation. This is actually a known behaviour, with a long thread of discussions  [here](https://github.com/keras-team/keras/issues/2743). In short, you have to set a random seed for both NumPy and TensorFlow. I have set a random seed for NumPy only. So I am not sure if it will fix this issue. I will update here once I try it out. But most of the time, and for most other CNN architectures I have tried, the precision of class 0 and class 1 


###  Further Improvements

-   There is much room for better network architecture and hyper-parameter tuning.
-   Using CNN with the same architecture on other datasets didn’t give as impressive precision for 0 and 1. But by playing around with hyper-parameters, 
- Using RNN in the [sample file](https://github.com/nishantsikarwar/PythonProjects/blob/main/Bezant_Assignment.ipynb) we could use RNN to predict labels.  **TODO**

-   Although these results aren't good enough, there is no guarantee that they would give you profits on real-world trading because it would be limited by the strategy you choose to label your data. For example, I backtested the above trading strategy (with original labels and not model predictions!),  But that depends on the labelling of the data. If someone uses a better strategy to label the training data, it may perform better.
-   Exploring other features may further improve the result.

### Conclusion

I started working on this project with a very sceptical mind. I was unsure if the images would have enough information/patterns for ConvNet to find. But since the results seem to be much better than a random prediction, this approach seems promising. I especially loved the way they converted the time series problem to image classification.


**References**: 
Sezer, Omer & Ozbayoglu, Murat. (2018). Algorithmic Financial Trading with Deep Convolutional Neural Networks: Time Series to Image Conversion Approach. Applied Soft Computing. 70. 10.1016/j.asoc.2018.04.024.

