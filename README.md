This project was intitally made on a different git account and copied to my personal hence the single commit with all the code. Project timeline: Nov 24 - Dec 24


# Stock Foresight App 
Gather sentiment across articles for a certain company to help influence stock decisions. 

We used stock market article sentiment to train both an LSTM model and LLM (llama3.1 8B) to create a sentiment score for a certain ticker symbol in any given time period. Reading articles for every single stock decision takes too long, using our app we can get a quick sentiment to understand the market standpoint on a certain company. Utilizes FinHub api to get stock article information.


## Files: 
**LSTM_Training_pytorch.ipynb**: File to download financial phrasebank dataset and train / test LSTM model on the data \
**app.py**: Run the stock sentiment predictor UI. Input ticker symbol and target date to return a sentiment on the articles from that given time period \
**finetune.ipynb** : Fine-tune the llama model on our dataset. persist to huggingface \
**eval.ipynb** : Use fine-tuned model and test on the test set \
**graph.ipynb** : Generate graphs for slides presentation \
**stock_pred.ipynb** : Run fine-tuned model on the articles to gather data for results \
**custom_models.ipynb** : Additional testing on the newsmtsc dataset 


## How to Run:

Main files of interest are the LSTM_Training_pytorch, finetune, and app.py. The two notebooks can be run from top to bottom and have comments throughout to help. To run the UI, run "streamlit app.y " in terminal. pip install streamlit might be neccessary. 

