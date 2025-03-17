# Kaggle's Titanic Challenge

This challenge and its datasets can be found in https://www.kaggle.com/competitions/titanic/ .

This is the *legendary Titanic ML competition* – the best, first challenge for you to dive into ML competitions and familiarize yourself with how the Kaggle platform works. This competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck. Based on the full range of information provided in the training and testing datasets, your challenge is to formalize a machine learning classification model that indicates just that: whether the individual in question survived or perished in the accident.

I won't go into much details here about all the content of the datasets, their features, dimensions, etc. You can find this information on the challenge website. But I will comment on my own approach, which ranked me among the top 2.5% worldwide (score 0.79904). You can find the relevant code above, in the `titanic.py` file.

Since I studied a lot before actually getting my hands into it, I also assume that you have studied this challenge from other sources and this is not the first one you see. I will not show graphs comparing sex, class, age and survival probability here, there are plenty of them on YouTube. It might be interesting to take a look at these subjects if this is your first time here, it helps to visually stimulate some reasoning that we may not have at first glance. But, as I said, I will skip this entire part, which is very basic and does not help at all in my explanation.

**IMPORTANT:** for my code to work properly (but of course you can change this) the databases and the reference file for submitting the results must all be inside `./dados/` directory. The files generated as results will be located in `./resultados/` (but the system will inform you of this, don't worry).

## Data Wrangling

The beginning here is quite straightforward, this part does not require any unusual reasoning, as I will show you. First of all, we will deal with some of the NaNs observed in the dataset, more specifically the 'Embarked' feature, then we will test another feature that has several NaNs, 'Age'. Since this feature will depend on feature engineering in my approach, its NaNs will be dealt with later.

Next, I will start my procedures working with the names of the passengers. My goal here is twofold: to deal with the titles of each person (Mr., Mme., Master, whatever) and also to estimate the maiden name of the women (where this is possible). The titles are important because: a) they indicate age; b) they may indicate nobility status (and, consequently, a lower chance of perishing in an accident); c) they indicate group travel (groups may have a greater chance of survival given the mutual help, IMO).

Once the names are done, I proceed to conduct an inference study about people traveling together, based on the passengers tickets. There is no perfect relationship, but it helps with the final purpose. With this same line of reasoning, I engeneer a new feature accounting for the size of each group traveling together.

Next, I try to estimate the location of each passenger in relation to the cabins, after all, the members of a cabin closer to the lifeboats would have a greater chance of survival, right?

The variables that are still in categories at this point go through one hot encoding, and then the missing values from Age feature are estimated.

To fill in the Age feature, the use of statistical variables (mean, median, whether globally or by any class stratum, cabins, etc.) did not prove to be very useful, so I chose to forcibly estimate all 263 missing ages. That is why it is last, after all the other NaNs have already been treated.

For this imputation, I use the `IterativeImputer` algorithm, from the `scikit-learn` package, which proved to be quite satisfactory for the proposed problem. The ages of the 'Master' titles, for example, cannot be greater than 16 years, with values of 14 or less being preferable. "Why?" you ask yourself. Base the reasoning on https://en.wikipedia.org/wiki/Master_(form_of_address) and deepen your research, you will discover where I got this reason from.

Once all the data is filled in within the dataset, we can proceed to visualize any outliers. After all this data wrangling, we only have two variables that contain outliers, 'Fare' and 'Age'. I wrote a piece of code to filter this information in case it is of interest to you, in my models I did not remove the outliers.

## Modeling

Finally, I proceed with the coding and configuration of the algorithms to be used to create the machine learning models. In this challenge, I present four different models (XGBoost, LightGBM, CatBoost and Random Forest), plus an additional simple arithmetic mean ensemble between two of these algorithms.

This modeling part has no secret; it is simply the conclusion of what was done previously. The training group is separated 80/20, the models are trained and evaluated using the Coefficient of Determination (R^2) and Cross-validation score. In each of the models, my code establishes the hyperparameters, fit the models, processes the cross-validation, returns the evaluation indicators and presents a Confusion Matrix for user analysis. If everything is correct and if the user wishes, my code already generates the prediction upon the test dataset and generates a file with a personalized name for each model. In all possible algorithms I use seed, so in theory your result should be identical to mine if you run exactly my code without changing anything.

The best model up until the writing of this README was the one based on CatBoost, with a final score of 0.79904 on Kaggle, R^2 0.144094 and CV 0.828660. This score puts me in the top 2.44% worldwide (disregarding the 100% accuracy submissions, which according to Kaggle itself in the vast majority of cases are just cheating).


## Final words

I don't mind if you use my code, please feel free to do so. I just ask you that you please let me know what you changed in my code if your score is better than mine, so I can improve too ;)
