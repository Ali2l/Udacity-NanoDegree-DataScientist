# Disaster Response Pipeline Project

# Table of Contents
- Project Motivation
- File Descriptions
- Components
- Instructions of How to Interact With Project
- Licensing, Authors, Acknowledgements, etc.

# Project Motivation
In this project, I applied my data engineering skills to analyze disaster data to build a model for an API that classifies disaster messages.

# File Descriptions

### app Folder

- template foledr
  - master.html (Web application code)
  - go.html
- run.py (Flask code to run the application)

### data Folder

- disaster_categories.csv (Data)
- disaster_messages.csv (Data)
- process_data.py (data preperation pipeline)
- DisasterResponce.db (results of clean data saved in DB)

### models Folder

- train_classifier.py (ML pipeline)
- classifier.pkl (Pkl file to save the model)

### jupyter notebooks Folder
- categories.csv (Data)
- messages.csv (Data)
- disaster_messages.db (results of clean data saved in DB)
- ETL Pipeline Preperation.ipynb (data preperation pipeline)
- ML Pipeline Preperation.ipynb (ML pipeline)
  

### README.md File

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

![Screenshot 2022-04-30 225827](https://user-images.githubusercontent.com/37556393/166120867-2b907f3a-14b0-4ccd-9487-ff31e7003eb8.png)
![Screenshot 2022-04-30 225747](https://user-images.githubusercontent.com/37556393/166120873-c28dadec-8564-43db-9fc9-1c6cc9546f9c.png)
![Screenshot 2022-04-30 225808](https://user-images.githubusercontent.com/37556393/166120875-81e7c73d-7af7-4d00-adb7-4ef8cd08769d.png)

___


# Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ (if facing problems try http://localhost:3001 in a browser)
