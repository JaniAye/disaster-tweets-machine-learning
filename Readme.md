# NLP with Disaster Tweets

This project uses Natural Language Processing (NLP) to classify tweets about disasters.

## Requirements

- Python 3.10
- Flask
- scikit-learn
- pandas
- numpy

## Installation

1. Clone this repository:
    ```
    git clone https://github.com/yourusername/yourrepository.git
    ```
2. Navigate to the project directory:
    ```
    cd yourrepository
    ```
3. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Usage

1. Start the Flask server:
    ```
    You can run the application using either of the following commands, depending on your system configuration:

     ```bash
     python app.py
        or 
     python3 app.py
     ```
2. Open a web browser and navigate to `http://127.0.0.1:5000/`.
3. Enter a tweet in the text area and click the "Predict" button to classify it.

## Model Training

The model was trained on a dataset of tweets about disasters. The dataset was split into training and test sets, and the model was trained using the training set. The model's performance was evaluated using the test set.

The model uses a bag-of-words approach to convert the tweets into numerical features that can be used for machine learning. The model itself is a logistic regression model.

The trained model is saved to a pickle file (`disaster-tweets.pkl`), which is loaded by the Flask app to make predictions.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
