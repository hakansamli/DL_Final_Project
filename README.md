# DL_Final_Project

Introduction:
The paper we want to implement builds three different neural networks using LSTM, in order to perform stock market prediction with S&P 500 index (SPY). We decided to work on the paper because we were not only interested in solving problems in the stock market, but also interested in comparing different uses of LSTM.

Data:
The data used in the paper are from S&P 500 (SPY) index. The researchers have merged two datasets together for the analysis. Among the merged two datasets, the first one has adjusted closing price and stock volumes and the other dataset has accounting and corporate statistics. We tried to email the researchers about the dataset but the email didn’t go through. We are planning to scrape data for the project from yahoo finance website.

Methodology:
We are implementing a research paper.Thus,the hardest part in the project could be pre-processing of the data.LSTMs are hard to train and we might face some computational problems during that time.Also,we might have to work a lot to improve the model to achieve the required accuracy threshold.
There are three different methods of neural networks using LSTM.
1) Time Series Model
AutoRegressive Integrated Moving Average (ARIMA) model is a widely used statistical method for time series forecasting (equation 1). In this work, we followed the Box-Jenkins Methodology to build an ARIMA model as a baseline to compare with Deep Learning models. [4] For the ARIMA model, only “adjusted close price” was used to fit the model. We used summary statistics and functions such as moving average and autocorrelation function to identify data trends and the parameters (p, d, and q) of ARIMA model.
δY​t​(p,d,q)=μ+􏰀​p=1​(φ​p ​×δY​t−p​)−􏰀​pq=1(​ θ​q ​×e​t−q​) where δY​t ​=Y​t ​−Y​t−d ​(1) 2) RNN with Single/Stacked-LSTM Model
The main idea of RNN is to apply the sequential observations learned from the earlier stages to forecast future trends. Long-Short Term Memory (LSTM) model is an updated version of RNN. It can overcome the drawback of RNN in capturing long term influences.
LSTM introduces the memory cell that enables long-term dependency between time lags. The memory cells replace the hidden layer neurons in the RNN and filter the information through the gate structure to maintain and update the state of memory cells. The gate structure includes input gate, forget gate and output gate.
The forget gate in the LSTM determines which cell state information is discarded from the model, it accepts the output from the previous time step h​t−1 ​and the new input of the current time step. Its main function is to record the amount of information reserved from the previous cell state to the current cell state. It will output a value between 0 and 1 where 0 means complete reservation and 1 means complete abandonment.
3) Attentions LSTM Model
Machine learning algorithms are inspired by biological phenomena and human perception. For instance, we do not treat all information with equal importance, instead human perception focuses on the important parts first for the newly received information. This phenomenon is analogous to the financial market as well, as the prices of securities assign different levels of importance into the market information, and it prompts us to use the Attention Mechanism to add this feature into our RNN LSTM. In our model, we apply the soft attention, where we update the input of the model by assigning weights to input information based on the learning results and obtaining results in a more logical order. Mathematically, it is formulated as:​exp(e​t ​)​e​t​=tanh(W​a​[x​1​,x​2​,...,x​T​]+b) α​t=​ ​􏰀​Tk=1​exp(e​k​)
