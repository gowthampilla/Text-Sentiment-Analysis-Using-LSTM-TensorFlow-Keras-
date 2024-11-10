#Text Sentiment Analysis Using LSTM (TensorFlow & Keras)
This project showcases a Text Sentiment Analysis system, using LSTM (Long Short-Term Memory) networks to classify text as positive or negative. Built with TensorFlow and Keras, it demonstrates the power of deep learning for Natural Language Processing (NLP), enabling users to analyze sentiments in text data, such as customer reviews or social media posts.

Project Highlights
LSTM Model Architecture:

Embedding Layer: Transforms text into dense word vectors.
LSTM Layer: Captures the sequence patterns in text.
Dense Layer: Classifies text sentiment (positive/negative).
Data Preprocessing:

Tokenizes and pads sequences for consistent input.
Splits data into training and test sets.
Training and Evaluation:

Trains with binary cross-entropy loss for binary sentiment classification.
Evaluates accuracy, precision, and recall on test data.
Real-Time Predictions:

Takes custom text input for live sentiment predictions.
Dataset
We use a labeled dataset, such as the IMDB movie reviews, containing 25,000 labeled samples each for training and testing. The model is trained to identify the sentiment polarity based on the context and phrasing in the reviews.

Requirements
Install dependencies with:

bash
Copy code
pip install tensorflow numpy matplotlib
Usage
Run the Model: Train, validate, and test on labeled text data.
Predict Sentiment: Input custom text for real-time sentiment analysis.
To Run:
Clone the repo, install dependencies, and run:

python
Copy code
python sentiment_analysis.py
Project Structure
Data Preprocessing: Loads, tokenizes, and pads text data.
Model Building: Constructs an LSTM-based sentiment classifier.
Evaluation: Visualizes accuracy and loss trends.
Prediction: Provides a real-time interface for sentiment prediction.
Example Prediction
python
Copy code
new_review = "This movie was amazing!"
new_review_seq = tokenizer.texts_to_sequences([new_review])
new_review_padded = pad_sequences(new_review_seq, maxlen=MAX_SEQUENCE_LENGTH)
prediction = model.predict(new_review_padded)
sentiment = "Positive" if prediction >= 0.5 else "Negative"
print(f"Sentiment: {sentiment}")
Results
With an accuracy rate around 85-90%, this LSTM model effectively identifies sentiment in text. It offers flexibility for various NLP applications like review analysis, feedback processing, and social media sentiment tracking.

Future Improvements
Hyperparameter Tuning: Optimize model performance with varying LSTM units, dropout rates, or learning rates.
Transfer Learning: Consider integrating pre-trained models (e.g., BERT) for enhanced performance on larger datasets.
