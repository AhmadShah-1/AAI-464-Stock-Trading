Now that we have a process to train the model, and features that we can discern as important

We need to consider approaches to training the model on different stocks

Reasoning:

Take TSLA for example, its stock price and volume were minimal (compared to today) prior to 2020, then it boomed in 2021, and steadily increased

SMP500 on the other hand, does not have a high variance, and volume is consistent

If training on stocks such as Tesla, no matter how long of a period the model is trained on certain features (from feature engineering) will dominate, such as volume, variance, SMA5, etc. However, applying models that weigh these features highly to more stable stocks will lead to incorrect predictions

Consideration: SMP500 is known to be very stable due to its culmination of holdings, and Tesla is just a singular privately owned company. However the point still holds, certain features will be held highly for Tesla, but when used on another company, the features that had a high weight for Tesla might not be entirely compatable of sensible for a different stock pattern. There are several different techniques (used by traders) to evaulate different stocks, to help solve this. However applying several rulesets will be cumbersome (in verifying and testing) to implement and code. Instead a different approach is needed

Approach 1: Take several different tickers, concatenate the data into a single dataframe, order by date, and train the model

Approach 2: Take several different tickers, concatenate, normalize price ranges, order by date, and train the model

Approach 3: Seperate training per stock and combine predictions (Ensemble)

Approach 4: Seperate training per feature and combine predictions
