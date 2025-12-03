Absenteeism Prediction Using Machine Learning

This project predicts employee absenteeism hours based on various workplace and personal factors. The goal is to help organizations identify factors contributing to high absenteeism and make data-driven decisions to improve productivity.

📁 Project Structure
├── Absenteeism Predictions-checkpoint.ipynb   # Jupyter Notebook with full workflow
├── model.pkl                                  # Saved model using pickle
├── scaler.pkl                                 # Saved scaler using pickle (if applicable)
├── data/                                      # (Optional) Raw and cleaned datasets
├── README.md                                  # Project documentation

 Project Overview

This project uses machine learning to:

Clean and preprocess employee absenteeism data

Encode categorical variables

Scale numerical features

Build a predictive model

Evaluate performance

Save the trained model and preprocessing objects using pickle

Generate predictions on new datasets with the same structure

 Data Preprocessing

The preprocessing steps performed include:

Handling missing values

Creating dummy variables

Feature engineering (e.g., categorizing reasons for absence)

Scaling numerical features with StandardScaler

Splitting data into training and testing sets

All steps are detailed in the notebook Absenteeism Predictions-checkpoint.ipynb.

 Model Building & Saving with Pickle

The machine learning model (e.g., Logistic Regression) was trained and then saved using Python’s pickle library:

import pickle

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)


The same approach was used to save preprocessors such as scalers or encoders:

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)


You can load them later for prediction:

with open("model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    loaded_scaler = pickle.load(file)

 How to Use the Model for New Predictions

Prepare your new dataset in the same column order as the training dataset.

Load the scaler and model saved with pickle.

Apply the scaler:

new_scaled = loaded_scaler.transform(new_data)


Predict:

predictions = loaded_model.predict(new_scaled)


Export or analyze the prediction results.

Technologies Used

Python

Pandas

NumPy

Scikit-learn

Pickle

Jupyter Notebook

 Future Enhancements

Save the entire preprocessing + model pipeline in one pickle file

Build a user-friendly interface for prediction

Deploy the model via Flask/FastAPI for real-time usage

Add more features to improve model accuracy

 Contact

If you have questions or want help extending the project, feel free to reach out.
