# Disaster Response Pipeline Project

# Table of Contents
- Project Motivation
- File Descriptions
- Components
- Instructions of How to Interact With Project
- Licensing, Authors, Acknowledgements, etc.

# Project Motivation
In this project, I applied my data engineering skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. I have created a machine learning pipeline to categorize real messages that were sent during disaster events so that the messages could be sent to an appropriate disaster relief agency. The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

# File Descriptions

## app Folder

- template foledr
  - master.html (Web application code)
  - go.html
- run.py (Flask code to run the application)

## data Folder

- disaster_categories.csv (Data)
- disaster_messages.csv (Data)
- process_data.py (data preperation pipeline)
- DisasterResponce.db (results of clean data saved in DB)

## models Folder

- train_classifier.py (ML pipeline)
- classifier.pkl (Pkl file to save the model)

## jupyter notebooks Folder
- categories.csv (Data)
- messages.csv (Data)
- disaster_messages.db (results of clean data saved in DB)
- ETL Pipeline Preperation.ipynb (data preperation pipeline)
- ML Pipeline Preperation.ipynb (ML pipeline)
  

## README.md File

___

# Project Components
There are three components you'll need to complete for this project.

### 1- ETL Pipeline
In a Python script, `process_data.py`, write a data cleaning pipeline that:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database
  
### 2- ML Pipeline
In a Python script, `train_classifier.py`, write a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

### 3- Flask Web App
The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. The outputs are shown below:


___


# Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
