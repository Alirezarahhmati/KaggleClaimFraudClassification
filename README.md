# Fraud Detection

## Abstract :
This project focuses on vehicle insurance claim fraud detection using a dataset with 32 features. We conducted exploratory data analysis to understand the dataset and identify patterns. However,the dataset suffered from class imbalance, particularly with a limited number of fraudulent claims. To address this, we employed oversampling techniques, specifically SMOTE. 
We implemented several classification models, including Logistic Regression, SVM, Decision Trees, and Random Forest, using Scikit-Learn. The models were evaluated using stratified cross-validation, and we further enhanced the SVMand Random Forest models through hyperparameter tuning, feature engineering, and preprocessing

## Introduction:
This project focuses on detecting insurance claim fraud using a comprehensive dataset with 32 features. Our main challenge is the dataset's imbalance, with fewer fraudulent claims compared to legitimate ones. To address this, we willexplore different methods to rebalance the dataset and enhance our fraud detection models.
Our goal is to develop accurate models that can predict fraudulent claims based on the provided features. We willemploy various classification models and evaluate their performance on the dataset to determine the most effective approach for fraud detection.

To tackle the class imbalance issue, we willuse techniques such as oversampling, undersampling, etc approaches. These methods aim to rectify the skewed distribution of fraudulent and legitimate claims, improving the model's ability to identify fraud cases. 
Using the dataset and selected models, we willtrain the models on the available data and evaluate their performance. We willuse evaluation metrics such as accuracy, precision, and F1-score to assess the models' effectiveness in identifying fraudulent claims. Additionally,we willplot the Receiver Operating Characteristic (ROC) curve to analyze the models' performance across different thresholds.

## Methods:

### Data Preprocessing
In the initial stage, we meticulously analyze our data set, extracting valuable insights. Here's an overview of our data set:

![image](https://github.com/Alirezarahhmati/KaggleClaimFraudClassification/assets/124795821/b04d4460-4f9a-4018-98a1-0eb09a4d0919)

After a thorough examination, we have confirmed the absence of missing values in our dataset. To gain insights into the distribution of our target variable, we utilize a box plot visualization :

![image](https://github.com/Alirezarahhmati/KaggleClaimFraudClassification/assets/124795821/b3e8d02f-cbed-4395-9b8c-6b7063a1450d)

The plot clearly shows a significant class imbalance in our dataset, indicating an uneven distribution of samples across different classes. To tackle this issue, we plan to implement appropriate methods in the future.

In the subsequent plots, we observe the box plots representing the numerical features:

![image](https://github.com/Alirezarahhmati/KaggleClaimFraudClassification/assets/124795821/69338cf9-9d63-48d9-80ed-fd06523a05bd)

We notice that the age feature contains certain zero values that need to be removed. Other features do not provide explicit details.

Now,let's examine the categorical features and their corresponding histogram plots.

![image](https://github.com/Alirezarahhmati/KaggleClaimFraudClassification/assets/124795821/28ca3211-f48f-4cfb-be49-d13725bb38a7)

Byobserving the distribution of features, we can gain insights into the relationship between the input features and our target variable.

Upon careful examination, it becomes evident that there are zero values present in the "DayOfWeekClaimed" and "MonthClaimed" features. To address this, we proceed to remove the zero values from these two features, as well as the "age"feature.

Exploring Various Models for Data Training :

In this section,we will train our dataset using a range of models,including Logistic Regression, SVM,Decision Trees,and Random Forest. To address the class imbalance issue,we will employ various techniques such as oversampling,undersampling,and SMOTE. Additionally,in order to enhance the performance of our models,we will utilize 10-fold and 5-fold cross-validation methods. These approaches aim to improve the accuracy and robustness of our models during the training process.

### **Logistic Regression**

Applying Robust Cross-Validation with 10-fold and implementing logistic regression with max\_iter=1000 Parameter.

#### **Over sampling :**

In order to address the class imbalance issue, we willemploy the RandomOverSampler technique from the imblearn library. This approach involves generating synthetic samples for the class with fewer instances, effectively increasing its representation in the dataset. Byintroducing random values, we aim to create a more balanced dataset for training purposes.

This is the result :

Result after over sampling the data : \
Average Accuracy for Logistic Regression : 0.6496108949416342 \
Average Precision for Logistic Regression : 0.12991179583959292 \
Average Recall for Logistic Regression : 0.8504441327723236 \
Average F1\_score for Logistic Regression : 0.22535229075865099 

The current results do not provide a clear indication, necessitating a closer examination through the ROCplot.

![image](https://github.com/Alirezarahhmati/KaggleClaimFraudClassification/assets/124795821/d3cdd722-61cb-4299-b448-933605434cc8)

As the area under the graph approaches 1, the performance improves significantly. Ideally,the graph should shift towards the upper left corner,indicating better results.

The Logistic Regression method with over sampling exhibits lower accuracy but demonstrates a favorable AUC(Area Under the Curve) performance.

#### **Under Sampling :**

To address the class imbalance issue, we utilize the undersampling technique. Specifically,we employ the RandomUnderSampler from the imblearn library. This method involves reducing the size of the class with a higher number of samples by randomly removing instances. Byimplementing undersampling, we aim to create a more balanced dataset for training our models.

This is the result :

Result after under sampling the data : \
Average Accuracy for Logistic Regression : 0.6543450064850844 \
Average Precision for Logistic Regression : 0.1292138266335648 \
Average Recall for Logistic Regression : 0.8310074801309023 \
Average F1\_score for Logistic Regression : 0.22361107777329803

And the ROCresult :

![image](https://github.com/Alirezarahhmati/KaggleClaimFraudClassification/assets/124795821/5832c1e4-5737-47a6-aad3-d2a533641ff0)

Under-sampling exhibits inferior results in accuracy and AUCcompared to Over-sampling.

#### **SMOTE:**

SMOTE,which stands for Synthetic Minority Over-sampling Technique,is a popular algorithm used to address class imbalance in machine learning. It generates synthetic samples for the minority class by interpolating between neighboring instances. SMOTEhelps to balance the dataset by creating new synthetic data points,thereby increasing the representation of the minority class. This technique effectively tackles the class imbalance problem and can improve the performance of models when dealing with imbalanced datasets.

This is the result :

Result after use SMOTE for oversampling :\
Average Accuracy for Logistic Regression : 0.6668612191958496 \
Average Precision for Logistic Regression : 0.13308114824018746 \
Average Recall for Logistic Regression : 0.8266012155212715 \
Average F1\_score for Logistic Regression : 0.2291933349973335

And the ROCplot :

![image](https://github.com/Alirezarahhmati/KaggleClaimFraudClassification/assets/124795821/1346f3b8-eedb-4ebe-a85c-01a71f66ab2d)

Based on the obtained results, it appears that Logistic Regression is not a suitable model for our dataset.

We can now visualize the results of all the sampling methods for Logistic Regression in a single plot.

![image](https://github.com/Alirezarahhmati/KaggleClaimFraudClassification/assets/124795821/d885b669-071e-4e12-821c-8406b346fa1f)

The plotted results clearly demonstrate that in this particular model, oversampling outperforms other sampling methods.

### **SVM** (RBF kernel)

*Oversampling :*In this section, oversampling was employed to balance the data. The C parameter for this model was set to zero due to the significant time complexity associated with finding optimal values for large datasets. It appears that this model, with its substantial time complexity,may not be well-suited for handling large datasets like the one we have. Iuse 5-fold cross validation in this section.

Result :

Result after under sampling the data : \
Average Accuracy for SVM : 0.8666666666666668 \
Average Precision for SVM : 0.1792530323561242 \
Average Recall for SVM : 0.34571092831962397 \
Average F1\_score for SVM : 0.23585195764715222

While achieving an accuracy of 86%may initially appear promising,it would be prudent to further assess the performance by examining the ROC plot.

![image](https://github.com/Alirezarahhmati/KaggleClaimFraudClassification/assets/124795821/7d511619-5598-4dd9-b39f-c823786b7853)

Obtaining both a high accuracy and a good AUC (Area Under the Curve) is indeed encouraging. It would be worthwhile to explore the performance of other sampling methods as well.

*UnderSampling :*Let's proceed and examine the results.

Result after under sampling the data :\
Average Accuracy for SVM : 0.5968871595330739 \
Average Precision for SVM : 0.12252848939523386 \
Average Recall for SVM : 0.9306521739130436 \
Average F1\_score for SVM : 0.21653400134446593

![image](https://github.com/Alirezarahhmati/KaggleClaimFraudClassification/assets/124795821/fa8d4d52-aa11-4e79-876c-29ace0c38f6e)

The application of undersampling in SVMyielded unfavorable results, as it led to low accuracy.

*SMOTE :*We anticipate that the SMOTEmethod willyield superior results compared to the two methods used previously.

Result :

Result after under sampling the data : \
Average Accuracy for SVM : 0.9011024643320363 \
Average Precision for SVM : 0.20302490050462194 \
Average Recall for SVM : 0.22432432432432434 \
Average F1\_score for SVM : 0.21276519304441335

As we expected,the result was better. And the ROC plot :

![image](https://github.com/Alirezarahhmati/KaggleClaimFraudClassification/assets/124795821/9dd7ed3d-7455-44c2-8f63-0b05fcbda017)

*Compared Methods :*We can visualize the results of all the methods in a single plot, allowing us to compare and analyze them collectively.

![image](https://github.com/Alirezarahhmati/KaggleClaimFraudClassification/assets/124795821/129097c7-db70-4c6d-85e0-0f36d1de4162)

Given the document's size,Ihave only included the SMOTEresults for Decision Trees and Naive Bayes. However,for a comprehensive understanding of all the methods and their details,I recommend referring to the code file.

### Decision Tree

In this section,the Decision Tree model was employed to make predictions on unseen data. The model was configured with the criterion set to 'gini',max\_depth set to 15,and random\_stat set to 42. Due to the time complexity involved,Iexplored a few values for max\_depth and determined that 15 is an appropriate depth for this dataset. It is crucial to strike a balance between bias and variance,as increasing the max\_depth can lead to overfitting,while decreasing it can result in underfitting. Thus,selecting the optimal value for max\_depth is of utmost importance.

#### SMOTE:

Now we can evaluate the performance of the Decision model on our dataset by employing the SMOTEmethod to balance the data.

Result :

Result after over sampling the data : \
Average Accuracy for Decision Tree : 0.9059014267185473 \
Average Precision for Decision Tree : 0.21571216710016627 \
Average Recall for Decision Tree : 0.21452781673679291 \
Average F1\_score for Decision Tree : 0.21387718699939046

Achieving an accuracy of 90% is highly commendable and indicates strong performance for our dataset. Now,let's take a look at the ROCplot to further evaluate the results.

![image](https://github.com/Alirezarahhmati/KaggleClaimFraudClassification/assets/124795821/7db5c36c-c5e2-4a31-82d3-f37914c3b250)

Unfortunately,the AUC(Area Under the Curve) value is significantly low,indicating that the model's performance is not satisfactory. One potential approach to address this issue could be to experiment with different values for the max\_depth parameter. However,based on the current observations, it appears that the Decision Tree model is not well-suited for this particular dataset.

Below,you can observe the results of various methods for Decision Tree displayed in a plot:

![image](https://github.com/Alirezarahhmati/KaggleClaimFraudClassification/assets/124795821/44c8796f-0082-4ede-b226-3ff4d661dddd)

### Random Forest

Random Forest is an ensemble learning method that combines multiple decision trees to make accurate predictions.

Let's take a quick look at the results

#### Over Sampling :

Result : 

Result after over sampling the data : \
Average Accuracy : 0.9410505836575875 \
Average Precision : 0.5817857142857144 \
Average Recall : 0.04443665264142122 \
Average F1\_score : 0.08222797608840972

ROCplot :

![image](https://github.com/Alirezarahhmati/KaggleClaimFraudClassification/assets/124795821/6f993a0e-50b2-4352-9925-9b79673853fa)

#### Under Sampling :

Result :

Result after over sampling the data : \
Average Accuracy : 0.6492866407263294 \
Average Precision : 0.1371571562890382 \
Average Recall : 0.9165614773258532\
Average F1\_score : 0.23856089124099925

ROCplot :

![image](https://github.com/Alirezarahhmati/KaggleClaimFraudClassification/assets/124795821/dc655fa6-fe4a-4965-a19f-b71b066992ff)

#### SMOTE:

Result :

Result after over sampling the data : \
Average Accuracy : 0.9401426718547341 \
Average Precision : 0.5733333333333334 \
Average Recall : 0.01625759700794764 \
Average F1\_score : 0.03126140864542294

ROC plot :

![image](https://github.com/Alirezarahhmati/KaggleClaimFraudClassification/assets/124795821/3d2ec2e3-b376-4365-904d-219bac24fec7)

Comparing Methods :

![image](https://github.com/Alirezarahhmati/KaggleClaimFraudClassification/assets/124795821/4f8ffe83-d3b2-468c-bcc9-5f8bc565fa29)

We observe a remarkably high accuracy accompanied by an excellent AUC value,indicating that the results are exceptionally impressive.

### Naive Bayes

Now,let's proceed with trying the Naive Bayes algorithm and evaluate the obtained results.

#### SMOTE:

Result :

Result after SMOTE : \
Average Accuracy : 0.12315175097276265 \
Average Precision : 0.06141498988346748 \
Average Recall : 0.9555750350631136 \
Average F1\_score : 0.11541194046968373

And the ROCplot :

![image](https://github.com/Alirezarahhmati/KaggleClaimFraudClassification/assets/124795821/443bb772-abad-4368-8601-aec302147fa6)

It is evident from our findings that the Naive Bayes model performs poorly on this dataset.

### Compare Models

![image](https://github.com/Alirezarahhmati/KaggleClaimFraudClassification/assets/124795821/bc318994-93c0-4863-97ba-e8581cb33b06)

## Conclusion :

In conclusion, one of the primary challenges addressed in this project was the imbalance in the dataset. To tackle this issue, various sampling methods including Oversampling, Undersampling, and SMOTEwere employed. Among these methods, SMOTEconsistently outperformed the others in balancing the data, as evident from the results.
It was observed that using Oversampling alone led to model overfitting, while Undersampling resulted in underfitting. However,combining both approaches through SMOTEyielded favorable outcomes by effectively addressing the data imbalance.

Furthermore, the performance of different classification models was evaluated. The Decision Tree model performed poorly,exhibiting signs of overfitting on the data. Logistic Regression, on the other hand, showed limited accuracy and underfitting. SVM demonstrated good results, but its time complexity was relatively high.
In the end,Random Forest emerged as the best-performing model. It achieved the highest accuracy while maintaining a reasonable time complexity. Hence,Random Forest can be considered the most suitable model for this particular dataset.

Overall,this project highlights the significance of addressing data imbalance and showcases the effectiveness of various sampling methods. Additionally,it emphasizes the importance of selecting the appropriate model based on performance metrics and considerations such as time complexity. The findings and methodologies presented here provide valuable insights for developing fraud detection systems in the vehicle insurance domain.

Resource:

[https://scikit-learn.org/ ](https://scikit-learn.org/)\
[https://www.analyticsvidhya.com/ ](https://www.analyticsvidhya.com/)\
[https://www.geeksforgeeks.org/ ](https://www.geeksforgeeks.org/)\
[https://pythonprogramming.net/ ](https://pythonprogramming.net/)\
[https://www.section.io/ ](https://www.section.io/)\
[https://www.simplilearn.com/ ](https://www.simplilearn.com/)<https://medium.com/>
