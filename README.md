# Predict profitability of trades based on indicator buy / sell signals <br>
Trade profitability analysis for trades based on various indicators signals:
<li> MACD </li>
<li> Simple Moving Average </li>
<li> Exponential Moving Average </li>
<br>
Trading assumptions:
<ol>
  <li>Trade is profitable if, profit >0</li>
  <li>Buy / sell happen the following day of the signal</li>
  <li>Buy / sell are taken 10% from the open price towards close price</li>
</ol>
<br>
Machine learning assumptions:
<ul>
  <li>Binary classification: 1 - profit, 0 - loss</li>
  <li>A separate model for each company / ticker</li>
  <li>Model is trained vs optimal precision</li>
</ul>
<br>
Machine learning models used:
<ol>
  <li>Linear Support Vector Classifier</li>
  <li>Decision Tree Classifier</li>
  <li>Random Forest Classifier</li>
  <li>Gradient Boosting Classifier</li>
  <li>XGBoost Classifier</li>
  <li>Keras classifier</li>
</ol>
<br>
<u>Trade analysis intermediate results:</u><br>
30-40% of trades based on indicator signals are profitable<br>
In general trades on SMA signals are more often profitable than the ones based on EMA signals<br>
<br>
<u>Trade profitability predictions intermediate results (based on test data)</u>/<br>
The precision of the predictions is oscilating around 70%, which is pretty good, considering that the analysts estimate other signals accuracy as 30 to 50% (double top, shoulder & arms, etc). This means, there is ~70% chance that predicted trade will be profitable (Reminder: profitable -> profit > 0)<br>
However, the recall is only around 15%, which means that very the model pick-up very few of the actually profitable trades.<br>

#Detailed analysis tbc

