## Project Proposal
### Infographic
![infographic](infographic.png)

### Motivation & Background
Today, heart disease is a leading cause of death around the globe. In the United States alone, heart disease claims the life of one person every 36 seconds. Annually, it is responsible for the deaths of roughly 655,000 Americans which equates to 1 in every 4 deaths. To make matters worse, it costs our country about $219 billion per year due to healthcare service costs, medical costs, and lost productivity costs. Our group understands the impact this disease can have on so many individuals. Because of this, we decided to carry out our machine learning project on the prediction of heart disease.
#### Our Data Set
The data set we are using is a combination of data from four different health clinics around the globe, Cleveland Clinic Foundation, Hungarian Institute of Cardiology, V.A. Medical Center in Long Beach, CA, and University Hospital in Zurich, Switzerland. Each sample in the data set represents a person that was tested as a part of a heart disease study conducted by these facilities. The data set includes a discrete multiclass predictor of the current likelihood the patient has of having heart disease as well. We believe this predictor is based on how much specific blood vessels around each patient’s heart have narrowed.

### Methods
For this project, we have found a data set with 75 attributes to train our models with. For the unsupervised portion of our project, we plan to use the k-means algorithm to cluster our patients together under whether they have or do not have heart disease. We want to find the most important attributes that indicate the presence of heart disease for the patients in our data. For the supervised portion of our project, we plan to use a variety of machine learning techniques. In the past, the methods that have shown the most promise have been support vector machines (SVM), neural networks, decision trees, regression, and naive Bayes’ classifiers. We will be using classification for both of these. The UCI dataset we have been provided with has a goal attribute, which is a discrete number from zero (no presence) to 4 (almost certain presence). The goal for this model is to find trends within our data, and we hope these trends will allow us to find the groups most at risk of developing heart disease.

### Results & Discussion
Our measure of success on this project will be the final accuracy and recall. The final outcome of our project will be a program that predicts the likelihood that a person has heart disease. Therefore, recall will be an extremely important metric for us, as false negatives could prove to be deadly if not caught. Similarly, we aim to achieve high accuracy so that our results can be successfully applied to a large population. As a group, we have decided that our goal is to achieve a prediction accuracy and recall of greater than 60% to 65%, and a recall. Previous studies have reported approximately 75% accuracy and greater. Our overarching goal for this project is to identify the most important, contributing factors to heart disease for the patients in our dataset, and to then apply those findings in a model that can be used on a much larger scale.

### References
* Detrano R, Janosi A, Steinbrunn W, Pfisterer M, Schmid JJ, Sandhu S, Guppy KH, Lee S, Froelicher V. International application of a new probability algorithm for the diagnosis of coronary artery disease. Am J Cardiol. 1989 Aug 1;64(5):304-10. doi: 10.1016/0002-9149(89)90524-9. PMID: 2756873.
* https://www.cdc.gov/heartdisease/facts.htm
* https://www.sciencedirect.com/science/article/pii/S235291481830217X#:~:text=However%2C%20machine%20learning%20techniques%20can,
Regression%20and%20Na%C3%AFve%20Bayes%20classifiers.
* https://archive.ics.uci.edu/ml/datasets/heart+disease
* https://towardsdatascience.com/heart-disease-prediction-73468d630cfc
