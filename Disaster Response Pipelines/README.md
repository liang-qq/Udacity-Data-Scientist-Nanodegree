**Project Motivation**

This is an Udacity Data Science Nanodegree course project: Disaster Response Pipeline. In this project I utilize the message and category datasets to build a ETL pipeline that
loaded and preprocessed the dataset into a clearning dataframe. Then a machine learning pipeline was developed to build a machine learning model and tuning parameters to
make accurate predictions

**File Descriptions**

-ETL Pipeline Preparation.ipynb: The code in this Jupyter notebook was used as the template to develop process_data.py.

-ML Pipeline Preparation.ipynb: The code in this Jupyter notebook was used as the template to develop train_classifier.py. 

-process_data.py: This code load message and categories datasets, and creates an SQLite database after preprocessing the data.

-train_classifier.py: This code reads the developed SQLite database produced by process_data.py and build a machine learning model. It also has functions to
evaluate model performance and tune parameter.

-data: This folder contains all of the files necessary to loan and pre-process the raw datasets.

-app: This folder contains all of the files necessary to run the web app.

-model: This folder contains all of the files necessary to build up the machine learning model.


**Running Instruction**

1. Run the following commands in the project's root directory to set up your database and model.

- To run process_data.py which is the ETL pipeline that cleans data and stores in database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
     
- To run train_classifier.py which is a ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

Results

The main findings of the code are shown below.

The website should be succefully running here https://052b8c06e71c40e99045311a7e35ff96-3000.udacity-student-workspaces.com/:
A list of prarameters:
 * Running on http://0.0.0.0:3000/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 262-748-047
172.18.0.1 - - [22/May/2022 02:34:53] "GET / HTTP/1.1" 200 

A screenshot of the webpage:
![image](https://user-images.githubusercontent.com/42193031/169675836-7e66d7d2-ece6-4571-a786-07ff6ede8872.png)



**Technical Requirements**

The coding is written in Jupyter Notebook. Please be sure the libraries of Pandas, Numpy, nltk, json, plotly, flask, sys, re, pickle, sklearn 
and sqlalchemy are installed before running the code.


**Licensing, Authors, Acknowledgements, etc.**

Data was provided from Udacity in partnership with Figure Eight.
