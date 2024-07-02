
# E-Commerce Chatbot README

This repository contains code for a simple chatbot implemented in Python using TensorFlow and NLTK.

## Overview

The chatbot is trained to understand and respond to user queries based on predefined intents. It uses a bag-of-words approach to encode input sentences and a neural network for classification.

## Requirements

- Python 3.9.7
- TensorFlow
- NLTK (Natural Language Toolkit)

## Environment Activation 

1. Create a virtual environment:

```bash
python -m venv myenv
```
2. Activate the virtual environment:

```bash
.\myenv\Scripts\activate

```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
    but you don't have to train model is already trained and saved in chatbot_model.h5 file.

```bash
python ecomChatbot.py
```

2. Run with the chatbot:
run the comm

```bash
streamlit run chatbotApp.py  
```

## Files

- `intents.json`: Contains predefined intents and responses.
- `ecomChatbot.py`: Script to train the chatbot model.
- `chatbotApp.py`: Script to interact with the trained chatbot. with streamlit GUI.
- `chatbot_model.h5`: Trained model file.
- `words.pkl` and `classes.pkl`: Pickled files containing words and classes used for training.

## How it Works

1. **Data Preprocessing**: Intents are loaded from `intents.json`, tokenized, lemmatized, and converted into a bag-of-words format.
2. **Model Training**: A neural network model is defined and trained on the processed data.
3. **Prediction**: Input sentences are tokenized and converted into bag-of-words vectors, and the model predicts the intent.
4. **Response Generation**: Based on the predicted intent, a response is randomly chosen from the predefined responses in `intents.json`.

## Customization

- Modify `intents.json` to add or edit intents and responses.
- Tweak model architecture and parameters in `train_chatbot.py` for better performance.
- Extend functionalities by adding more functions in `chatbot.py`.


Feel free to adjust or expand it as needed!