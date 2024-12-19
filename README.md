# Stock Foresight App 
Gather sentiment across articles for a certain company to help influence stock decisions. 

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

