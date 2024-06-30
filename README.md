# Well Prediction Project

This project focuses on predicting well outcomes using machine learning techniques, particularly the Random Forest Classifier. The main goal is to predict a target variable related to well data based on various input features.

## Project Structure

- **Data Preprocessing**: Data is loaded and preprocessed to prepare for model training.
- **Model Training**: A Random Forest Classifier is used to train the model.
- **Model Evaluation**: The model's accuracy is evaluated on a test dataset.
- **Model Saving**: The trained model is saved for future use.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- pickle

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/PrajwalKorade0212/Well_Predction_CGWB.git
    cd Well_Predction_CGWB
    ```

2. Install the required packages:

    ```bash
    pip install pandas scikit-learn
    ```

## Usage

1. Load and preprocess the data:

    ```python
    import pandas as pd

    # Load your dataset
    df = pd.read_csv('path_to_your_dataset.csv')

    # Preprocess the data as done in the notebook
    df_x = df.drop(columns=['TARGET_VARIABLE'])
    df_y = df['TARGET_VARIABLE']
    ```

2. Train the model:

    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42)

    # Initialize Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    rf_classifier.fit(X_train, y_train)

    # Make predictions
    predictions = rf_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)
    ```

3. Save the trained model:

    ```python
    import pickle

    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(rf_classifier, f)
    ```

4. Make predictions with the saved model:

    ```python
    with open('rf_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    inp = [11.545, 92.74583, 7.74, 349, 0, 183, 25, 1, 1, 0, 175, 34, 22, 10, 0, 0.84, 0]
    prediction = loaded_model.predict([inp])
    print("Prediction:", prediction)
    ```

## Notebook Overview

- **Data Loading**: The data is loaded and inspected.
- **Preprocessing**: Necessary preprocessing steps like dropping columns and converting data types are performed.
- **Model Training and Evaluation**: A Random Forest model is trained and its accuracy is evaluated.
- **Prediction**: The model is used to make predictions on new data.
- **Model Saving**: The trained model is saved using pickle.

## Contributing

Feel free to open issues or submit pull requests if you have any improvements or suggestions.

## License

This project is licensed under the MIT License.
