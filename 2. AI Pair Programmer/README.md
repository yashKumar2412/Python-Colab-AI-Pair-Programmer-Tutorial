# AI Pair Programmer Tutorial

For this section, I experimented with the AI Pair Programmer tool, Cursor. I used it to create a basic front-end project with a ML model at 
the backend to predict a passenger's chances of survival on the Titanic.

The artifacts uploaded include:
1. titanic.csv - The [Kaggle Titanic Survivor dataset](https://www.kaggle.com/competitions/titanic/data).
2. train.py - Basic Python code with Logistic Regression implemented on the Titanic Survivor Dataset to train and save a model that classifies
   passengers on their survival.
3. titanic_model.pkl - Logistic Regression Classification Model trained on the Titanic Survivor Dataset.
4. predict.py - Basic Python code with Flask implemented to listen for requests with the passenger data and use the trained model to predict survival.
5. index.html - HTML code for the front-end webpage with clean CSS and a simple form with the required inputs to send to the model.

All these were made through the help of code generation and autocompletion of the Cursor tool on VSCode. For a detailed walkthrough, 
watch the [walkthrough video](https://youtu.be/bXqcCyEY-ns) I uploaded for the same.
