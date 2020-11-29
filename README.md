# DL_Final_Project

Introduction:
The paper we want to implement builds three different neural networks using LSTM, in order to perform stock market prediction with S&P 500 index (SPY). We decided to work on the paper because we were not only interested in solving problems in the stock market, but also interested in comparing different uses of LSTM.

Data:
The data used in the paper are from S&P 500 (SPY) index. The researchers have merged two datasets together for the analysis. Among the merged two datasets, the first one has adjusted closing price and stock volumes and the other dataset has accounting and corporate statistics. We tried to email the researchers about the dataset but the email didn’t go through. We are planning to scrape data for the project from yahoo finance website.

Methodology:
We are implementing a research paper.Thus,the hardest part in the project could be pre-processing of the data.LSTMs are hard to train and we might face some computational problems during that time.Also,we might have to work a lot to improve the model to achieve the required accuracy threshold.
There are three different methods of neural networks using LSTM.

1) Time Series Model

2) RNN with Single/Stacked-LSTM Model

3) Attentions LSTM Model

We will use mean squared error to measure the performance of our model.We would compare the predicted stock price with the actual price the next time and calculate the error values and try to optimize the model accordingly.

1) Time Series Model(ARIMA); Hakan will be doing the Time Series Model.
2) RNN with LSTM Model (LSTM);
Chun will be working on a LSTM model.
3) RNN with Stacked-LSTM (Stacked-LSTM)
Chun will be working on a stacked-LSTM.
4.RNN with LSTM + Attention (Attention-LSTM).
Uttam will be doing RNN with LSTM with Attention.