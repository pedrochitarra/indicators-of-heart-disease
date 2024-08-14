<p align="center">
  <a href="" rel="noopener">
 <img width=300px height=150px src="https://www.tbecker.com.br/uploads/images/2017/04/adote-12-medidas-para-proteger-a-saude-do-coracao-1493058672.jpg" alt="Heart Disease"></a>
</p>

<h3 align="center">Indicators of Heart Disease</h3>

<p align="center"> Project aimed to learn MLOps concepts and apply them to a real-world dataset. <br> 
</p>

# 🧐 Problem description <a name = "about"></a>
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

The folder notebooks contains the EDA notebook, where the dataset is analyzed and the features are understood.

# 🧪 Experiment tracking and model registry <a name = "experiment"></a>

The project uses MLflow to track experiments and register models. MLflow is an open-source platform to manage the end-to-end machine learning lifecycle. It helps with experiment tracking, reproducibility, and deployment.

MLFlow documentation can be found [here](https://www.mlflow.org/docs/latest/index.html).

The MLFlow server is hosted at DagsHub, and the experiments can be accessed [here](https://dagshub.com/pedrochitarra/indicators-of-heart-disease).

In Dagshub, the experiments are tracked and the models are registered. The models can be downloaded and used in other applications. Also, it integrates DVC, MLFlow and the Git repo, making it easier to track the experiments, models and code.

# 🔄 Workflow orchestration <a name = "workflow"></a>

The project uses DVC to orchestrate the workflow. DVC is an open-source version control system for machine learning projects. It is designed to handle large files, data sets, machine learning models, and metrics as well as code. Also, it is designed to work with Git and associate each Git commit with a unique DVC commit, in a way that the data, and code are all versioned together.

DVC documentation can be found [here](https://dvc.org/doc).

# ⚙️ Model deployment <a name = "deployment"></a>

The model is deployed using FastAPI, as seen at app/main.py script. The image
is built and can be used to generate predictions as well, available at
[here](https://hub.docker.com/repository/docker/pedrochitarra/indicators-of-heart-disease).

# 🔬 Model monitoring <a name = "monitoring"></a>

# 🖥️ Reproducibility <a name = "reproducibility"></a>

The model can be created running the pipeline defined at the dvc.txt file by
executing the command `dvc repro`. It will check the stages that are already
computed and will run the stages that are not computed yet. At the end, the
model will be created and the metrics will be saved at the MLFlow server.

With a model created, it can be downloaded using the `src/gather_mlflow_model.py`
script. By the model_family set at `params.yaml`, the model will be downloaded and
saved at the root folder as a .pkl file.

Then, the image can be created by running the command
`docker build -t indicators-of-heart-disease .` at the root folder.
The image will be created and can be used to generate predictions.

# 🪖 Best practices <a name = "best_practices"></a>
For every commit, the CI/CD pipeline is triggered. It checks the code quality
using flake8 and if there are any errors, the pipeline fails. The `.pre-commit-config.yaml`
file has the rules that are checked before every commit. It can be installed
locally also to avoid waiting for the CI/CD pipeline to check the code quality.
It can be installed by running the command `pre-commit install` and then check
the code quality by running `pre-commit run --all-files`.
