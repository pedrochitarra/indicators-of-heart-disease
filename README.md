<p align="center">
  <a href="" rel="noopener">
 <img width=300px height=150px src="https://storage.googleapis.com/kaggle-datasets-images/1936563/3189208/5abb971a28726c55b73b835df915118d/dataset-cover.jpg" alt="Heart Disease"></a>
</p>

<h3 align="center">Indicators of Heart Disease</h3>

<p align="center"> Project aimed to learn MLOps concepts and apply them to a real-world dataset. <br> 
</p>

## 🧐 About <a name = "about"></a>
Welcome to the Indicators of Heart Disease repository, an educational project
that aims to predict the presence of heart disease in patients based on telephonic interviews.

This dataset will be used to create the final project of the MLOps Zoomcamp course,
ministrated by [DataTalks.Club](https://datatalks.club/).

The original repository can be found
[here](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main).

The dataset used in this project can be found on Kaggle, and it can be accessed
[here](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease).

As per the dataset description:

"What subject does the dataset cover?

According to the CDC, heart disease is a leading cause of death for people of most races in the U.S.. About half of all Americans (47%) have at least 1 of 3 major risk factors for heart disease: high blood pressure, high cholesterol, and smoking. 

Other key indicators include diabetes status, obesity (high BMI), not getting enough physical activity, or drinking too much alcohol. Identifying and preventing the factors that have the greatest impact on heart disease is very important in healthcare. In turn, developments in computing allow the application of machine learning methods to detect "patterns" in the data that can predict a patient's condition."

"Where did the data set come from and what treatments has it undergone?

The dataset originally comes from the CDC and is a major part of the Behavioral Risk Factor Surveillance System (BRFSS), which conducts annual telephone surveys to collect data on the health status of U.S. residents. In this dataset, I noticed many factors (questions) that directly or indirectly influence heart disease, so I decided to select the most relevant variables from it. I also decided to share with you two versions of the most recent dataset: with NaNs and without it."

"What can you do with this data set?

As described above, the original dataset of nearly 300 variables was reduced to 40 variables. In addition to classical EDA, this dataset can be used to apply a number of machine learning methods, especially classifier models (logistic regression, SVM, random forest, etc.).

You should treat the variable "HadHeartAttack" as binary ("Yes" - respondent had heart disease; "No" - respondent did not have heart disease). Note, however, that the classes are unbalanced".

So, given the full credits to the dataset creator for describing the problem and giving us the opportunity to work with this dataset, the project can be started. The dataset selected is the one with NaNs, as it is the most realistic scenario.

## 🔎 EDA <a name = "eda"></a>

Before any model is created, it is important to understand the dataset and its features. This is done through Exploratory Data Analysis (EDA), which is a process of analyzing data sets to summarize their main characteristics, often with visual methods.
