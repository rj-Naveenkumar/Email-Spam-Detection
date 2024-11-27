# **Spam Email Detection**

A machine learning project designed to classify emails as spam or not spam using Natural Language Processing (NLP) techniques and an LSTM (Long Short-Term Memory) neural network model. This project demonstrates data preprocessing, text vectorization, and model training on email datasets.

---

## **Features**
- **Text Preprocessing**: Cleaning and tokenizing email content.
- **Model Architecture**: LSTM-based neural network for sequence modeling.
- **Spam Detection**: Classifies emails as spam or non-spam with high accuracy.
- **Visualizations**: Includes class distribution plots and word clouds for data insights.

---

## **Dataset**
The dataset used in this project contains two columns:
- `Category`: Labels the email as "spam" or "ham" (not spam).
- `Message`: The email text content.

---

## Technologies used : 
- pandas
- seaborn
- matplotlib
- nltk
- scikit-learn
- wordcloud

---

## **Methodology**
1. **Data Cleaning**:
   - Removed punctuation, stopwords, and any unnecessary text, including email headers like "Subject:".
   - Applied text preprocessing techniques to ensure clean and standardized input.

2. **Balancing the Dataset**:
   - Handled the imbalance between spam and non-spam emails using downsampling to equalize the number of examples for both classes.
   - Ensured the model isn’t biased toward the majority class.

3. **Text Vectorization**:
   - Converted email text data into numerical sequences using the `Tokenizer` from TensorFlow.
   - Padded sequences to ensure uniform input length for the LSTM model.

4. **Model Training**:
   - Implemented an LSTM (Long Short-Term Memory) neural network.
   - The model consists of:
     - An Embedding layer for word embeddings.
     - An LSTM layer for sequence learning.
     - Dense layers for binary classification.
   - Trained the model on 80% of the balanced dataset and validated it on the remaining 20%.

5. **Evaluation**:
   - Assessed model performance using metrics like accuracy and loss.
   - Monitored training and validation accuracy over epochs to prevent overfitting using callbacks like `EarlyStopping` and `ReduceLROnPlateau`.

---

## **Visualizations**
1. **Word Clouds**:
   - Created word clouds to visualize the most frequent words in spam and non-spam emails.
   - Helps in understanding the distinct vocabulary used in each category.<br>

    <img width="371" alt="non-spam" src="https://github.com/user-attachments/assets/ad884ed1-4a13-49f2-9853-26f521c571eb"><br>  
    *Word Cloud for Non-Spam Emails*  

   <img width="367" alt="spam" src="https://github.com/user-attachments/assets/c99b01bd-aa98-4320-a487-e9f2e02d8bbe"><br>
   *Word Cloud for Spam Emails*  

2. **Training Performance**:
   - Plotted training and validation accuracy over epochs to monitor model performance.
   - Provides insights into the learning progress and potential overfitting. 

   *Training and Validation Accuracy Over Epochs* <br>
   <img width="388" alt="Training performance" src="https://github.com/user-attachments/assets/2152749f-e403-431a-8804-92dc052139e1">
---

## **Results**
- **Training Accuracy**: Achieved an accuracy of **98.7%** on the training dataset.
- **Validation Accuracy**: Maintained a validation accuracy of **97.3%**.
- **Test Accuracy**: Evaluated the model on a separate test set with an accuracy of **96.5%**.
