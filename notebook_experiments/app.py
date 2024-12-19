import streamlit as st
import torch
from transformers import AutoTokenizer
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
import numpy as np
import finnhub


from transformers import AutoTokenizer, AutoModelForCausalLM





##LSTM STUff
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model_config = {
    'vocab_size': tokenizer.vocab_size,
    'embedding_dim': 512,
    'hidden_size': 128,
    'num_layers': 1,
    'num_classes': 3,
    'dropout_rate': 0.5
}
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_size=128, num_layers=2, num_classes=3, dropout_rate=0.3):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.layer_norm = nn.LayerNorm(hidden_size)
    def attention_net(self, lstm_output, attention_mask):
        attention_weights = self.attention(lstm_output)
        attention_weights = attention_weights.squeeze(-1)
        attention_weights = attention_weights.masked_fill(~attention_mask, float('-inf'))
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output)
        return context.squeeze(1)
    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        lstm_output, _ = self.lstm(embedded)
        context = self.attention_net(lstm_output, attention_mask.bool())
        x = self.dropout(context)
        x = F.relu(self.fc1(x))
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(**model_config).to(device)
model.load_state_dict(torch.load('best_model.pt'))
model.eval()
def predict_sentiment(model, tokenizer, sentence, device):
    # Set model to evaluation mode
    model.eval()

    # Tokenize the sentence
    inputs = tokenizer(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Move inputs to device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, 1)

        # Get probability scores
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = torch.max(probabilities).item()

    # Convert prediction to sentiment
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    predicted_sentiment = sentiment_map[predicted.item()]

    return predicted_sentiment, confidence

FINNHUB_API_KEY = "xxxxxxxx"


finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)




st.title('Sentiment Analysis with LSTM or LLM')

st.write("""
    This app allows you to classify the sentiment of text input using either an LSTM or LLM model. 
    Select a model to use from the dropdown below and enter a sentence to predict its sentiment.
""")

model_choice = st.selectbox("Select Model", ["LSTM", "LLM"])

user_input = st.text_area("Enter a sentence:")

if st.button("Predict Sentiment"):
    if user_input:
        if model_choice == "LSTM":
            ticker = user_input
            response = finnhub_client.company_news(ticker, _from='2024-12-04', to="2024-12-09")
            length = min(25, len(response))
            neutral_count = 0
            positive_count = 0
            negative_count = 0
            for x, y in enumerate(response[0:length]):
                #st.write(f"{x+1}. {y['summary']}")
                sentence = y["summary"]
                sentiment, confidence = predict_sentiment(model, tokenizer, sentence, device)

                if sentiment == "Negative":
                    negative_count += 1
                elif sentiment == "Neutral":
                    neutral_count += 1
                else:
                    positive_count += 1

                #st.write(f"Sentiment: {sentiment}")

            total_count = negative_count + neutral_count + positive_count

            score = (negative_count * 0 + neutral_count * 0.5 + positive_count * 1) / total_count
            st.write(f"Sentiment Score: {score:.2f}")
        
        elif model_choice == "LLM":
            # Placeholder for LLM functionality (does nothing for now)
            st.write("LLM model selected. This functionality is not implemented yet.")

    else:
        st.write("Please enter a sentence to analyze.")
