# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import warnings
#import optuna
from datetime import datetime
from sklearn.metrics import r2_score, log_loss
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

%autoindent OFF  # just for NeoVIM, you can safely comment this if you'll run the code somewhere else

# Loading the data (CSV - unprocessed data)
test = pd.read_csv('./dados/test.csv')
train = pd.read_csv('./dados/train.csv')

# Unifying train/test sets to perform data wrangling
train['origem'] = 'treino'
test['origem'] = 'teste'
dados = pd.concat([train, test], axis = 0)

####
# DEALING WITH NaN VALUES
####

# Check for NaN values
print(dados.isnull().sum())


# 'Cabin' feature: filling NaNs with 'U' ('unknown')
dados['Cabin'] = dados['Cabin'].fillna('U')


# 'Embarked' feature:
print("Passenger boarding ratios regarding the to boarding cities:\n")
print(dados['Embarked'].value_counts(normalize = True)) # based on the proportions alone, the passengers probably boearded in Southampton
print('\nDescriptive statistics of the amounts paid for the ticket ("Fare") in relation to each of the departure cities:\n')
print(dados.groupby(['origem', 'Embarked'])['Fare'].describe()) # based on the quartiles of the fare value, it's more likely that the passengers embarked in Cherbourg. I'm assuming this is the correct approach.
# Therefore...
dados['Embarked'] = dados['Embarked'].fillna('C')


# 'Fare' NaN (it's in test dataset)
dados[(dados['origem'] == 'teste') & dados['Fare'].isnull()]
# Passenger embarked in Southampton. Maybe 'Fare' has some correlation with 'Age'? Let's see.
dados[['Age', 'Fare']].corr()
# Since both features have a very weak correlation, I'll use the median of Southampton fares to impute this value
dados.loc[dados['Fare'].isnull(), 'Fare'] = dados[dados['Embarked'] == 'S'].Fare.median()


# 'Age' NaNs
dados['Age'].isnull().sum()
# There are 263 observations without Age records. I'll treat them with algorithmic imputation, but for better assertiveness in this I'll first process the names of the passengers.


####
# DATA WRANGLING
####

##### Dealing with names and titles

# Fetching last names
sobrenomes = dados['Name'].str.split(',').str.get(0)
sobrenomes = sobrenomes.str.strip()

# Fetching titles (Mr., Mrs., Miss., etc.)
resto_parcial = dados['Name'].str.split(',').str.get(1)
titulo = resto_parcial.str.split('.').str.get(0)
titulo = titulo.str.strip()

# Fetching first names
nomes = resto_parcial.str.split('.').str.get(1)
nomes_atuais = nomes.str.split('(').str.get(0)
nomes_atuais = nomes_atuais.str.strip()

# Fetching maiden names (when applicable)
solteira = nomes.str.split('(').str.get(1)
nome_solteira = solteira.str.split(')').str.get(0)
nome_solteira = nome_solteira.str.strip()

# Making a temp dataset with these manipulated names
temp = pd.concat([titulo, nomes_atuais, sobrenomes, nome_solteira], axis = 1)
temp.columns = ('Title', 'Name', 'Surname', 'Single')
temp[["PassengerId", "Ticket"]] = dados[["PassengerId", "Ticket"]]
temp.reset_index(col_level = 'PassengerId', inplace = True)
temp.drop('index', axis = 1, inplace = True)

# Some names do not follow the same record pattern, returning an empty value, we'll adjust it here: 
for i in temp.index:
    if temp.loc[i, 'Name'] == '':
        temp.loc[i, 'Name'] = temp.loc[i, 'Single']
    else: pass

# Drops maiden name from the record
temp.drop('Single', axis = 1, inplace = True)


###### Creating a new feature, a noblesse indicator, and checking all the existing titles
temp['Noblesse'] = np.nan
temp.Title.value_counts()

# ]Here some explanation is needed for your understanding.
# There are several titles of nobility presented in the list of passengers. The premise is that if the passenger is considered nobility, and therefore of an 'elite' class, he or she would have a greater chance of survival. So:
# - Dr: doctor, physician or similar. Most likely he or she was part of the elite (as is often the case today);
#- Sir: title granted by the monarch of England to some prominent individuals in the British community in general;
#- Mme: Madame, a woman belonging to the nobility in France;
#- Don/Dona: titles of nobility in Spain (male/female, respectively);
#- Countess: Degree of nobility, wife of a Count;
#- Jonkheer: lowest rank of nobility, but still noble.

# For this challenge purposes, some titles will be tested as both nobility and 'non-nobility':
#- Col: Colonel, high military rank. He is probably a nobleman, but due to the military issue he may have fought to save as many passengers as possible on the ship;
#- Major: Same reasoning as the Colonel;
#- Rev: Cleric;
#- Capt: Captain, same reasoning as the other military men.

# Explanations done, let's deal with the data:
militar = ('Col', 'Major', 'Capt')
noblesse = ('Dr', 'Col', 'Major', 'Sir', 'Mme', 'Don', 'Capt', 'the Countess', 'Dona', 'Jonkheer', 'Rev')
mask = temp.Title.isin(noblesse)
temp.loc[mask, 'Noblesse'] = 1
temp.Noblesse = temp.Noblesse.fillna(0)

# Creating groups based on the ticket number, they can be families friends or have any other affinity. 
tickets = pd.DataFrame(temp.Ticket.value_counts())
temp['Grouped'] = 1
tickets_individuais = tickets[tickets['count'] == 1].index
mask = temp['Ticket'].isin(tickets_individuais)
temp.loc[mask, 'Grouped'] = 0

# Drop 'Ticket' feature, already used
temp.drop('Ticket', axis = 1, inplace = True)


##### Finally let's add the temp dataset to the original df
dados.reset_index(inplace = True)
dados.drop(['index', 'Name'], axis = 1, inplace = True)
dados = pd.concat([dados, temp.drop('PassengerId', axis = 1)], axis = 1)


##### Changing 'Sex' feature to dummy variable (male = 1, fem = 0)
mask = dados['Sex'] == 'male'
dados.loc[mask, 'isMale'] = 1
dados.isMale = dados.isMale.fillna(0)
dados.drop('Sex', axis = 1, inplace = True)


##### Creating a variable about family size. The total size of the embarked family group is given by the number of siblings/spouses (SibSp) added to the number of parents/children (Parch) plus 1 (the individual himself)
dados['FamilyTotal'] = dados['SibSp'] + dados['Parch'] + 1


##### Filling cabins info according to tickets. I assumed that the same ticket number gives access to the same cabin (or at least the same class of cabins)
# Checks which tickets have over one record
mask = dados.Ticket.duplicated(keep = False)
mask = dados.Cabin.loc[mask].value_counts() > 1

# Identify the 'real' cabins (disregards the 'unknown' ones)
cabines = mask[mask == True].index
cabines = cabines.drop('U')

# From the right cabins, checks the corresponding tickets as to do the other way around
tickets = dados[dados.Cabin.isin(cabines)]['Ticket']
mask = dados.Ticket.isin(tickets)

# With the tickeds already indentified, I change the 'unknown' class to NaN, so it'll be easily filled with ffill function
mask_cabines = dados.Cabin == 'U'
dados.loc[mask_cabines, 'Cabin'] = np.nan

# Separates and orders the observations by 'Ticket' and 'Cabin', so that ffill function can be used, then returns the 'U' values to the cabins that really cannot be identified.
dados.loc[mask, ['Ticket', 'Cabin']] = dados.loc[mask, ['Ticket', 'Cabin']].sort_values(['Ticket', 'Cabin']).ffill()
dados.Cabin = dados.Cabin.fillna('U')

# Since the cabins are already well adjusted with reference to the Tickets, I'm keeping only the cabin classes (A, B, C, etc.) and eliminating the need for identification numbers
dados.Cabin = dados.Cabin.str.slice(stop = 1)


##### Partial cleaning and One Hot Encoding (OHE)

# Makes a copy of 'Survived' and 'origem' features, they'll be needed after OHE
dados_hist = dados[['Survived', 'origem']].copy()

# Removing 'Embarket', 'Ticket', 'Name' and 'Surname' for none have any more impact over classification
# Removing 'SibSp' and 'Parch' due to high correlation with 'FamilyTotal' -- DE REPENTE deixar Parch, pq a ligação entre pai e filho é maior do que entre irmãos/cônjuges
#dados.drop(['Embarked', 'Ticket', 'Name', 'Surname'], axis = 1, inplace = True)
dados.drop(['Embarked', 'Ticket', 'Name', 'Surname', 'SibSp', 'Parch'], axis = 1, inplace = True)

# Defines the features to be one hot encoded and copies them in a new df
colunas_ohe = ['Pclass', 'Cabin', 'Title']
temp_ohe = dados[colunas_ohe].copy()

# Calls OHE and processes it, dropping the first column from every original feature
ohe = OneHotEncoder(categories = 'auto', drop = "first", sparse_output = False).set_output(transform = 'pandas')
categ_ohe = ohe.fit_transform(temp_ohe)

# Eliminates original columns from dataset
dados.drop(colunas_ohe, axis = 1, inplace = True)

# Joins both datasets
dados = pd.concat([dados, categ_ohe], axis = 1)


##### Imputing Ages
# Just to remember, there are 263 observations that need to be estimated/imputed:
dados.Age.isnull().sum()


# Dropping target feature (Survived) and origin. If target feature remains in the dataset then the algorythm will try to impute its values too.
dados.drop(['Survived', 'origem'], axis = 1, inplace = True)

# Imputer parameters
imputer = IterativeImputer(max_iter=100,
                           tol=0.001,
                           initial_strategy='mean',
                           skip_complete=False,
                           min_value=0.33,
                           verbose=1,
                           random_state=1,
                           add_indicator=False,
                           keep_empty_features=False)

# The imputation itself generates an array, that I'll transform in a dataframe on spot
dados_imputados = pd.DataFrame(imputer.fit_transform(dados))

# Imputation clears the columns names, recovering.
dados_imputados.columns = dados.columns

# Passing on the imputed ages to our work dataset and recovering features 'Survived' and 'origem'
dados.Age = dados_imputados.Age
dados[['Survived', 'origem']] = dados_hist[['Survived', 'origem']]


##### Checking outliers through graph analysis (just 'Fare' and 'Age' have outlies)
# IMPORTANT: check outliers ONLY in train sets, never in test sets. One cannot drop any observation in test sets, ever.
# 'Fare' boxplot, there are some outliers
plt.figure(figsize =(10, 8))
sns.boxplot(data = dados[dados['origem'] == 'treino'].Fare,
            saturation = 0.8,
            fill = False,
            width = 0.3,
            gap = 0.3,
            whis = 1.5, # IQR
            linecolor = 'auto',
            linewidth = 1,
            fliersize = None,
            native_scale = False)
plt.title("Boxplot da variável 'Fare'", loc="center", fontsize=14)
plt.xlabel("Fare")
plt.ylabel("Valores")
plt.show()

# 'Age' boxplot, there are some outliers
plt.figure(figsize =(10, 8))
sns.boxplot(data = dados[dados['origem'] == 'treino'].Age,
            saturation = 0.8,
            fill = False,
            width = 0.3,
            gap = 0.3,
            whis = 1.5, # IQR
            linecolor = 'auto',
            linewidth = 1,
            fliersize = None,
            native_scale = False)
plt.title("Boxplot da variável 'Age'", loc="center", fontsize=14)
plt.xlabel("Idade")
plt.ylabel("Anos")
plt.show()

# Outliers cutoff through interquartile ranges (IQR)
# Estimating interquartile ranges (IQR)
q1_fare = np.percentile(dados[dados['origem'] == 'treino'].Fare, 25)
q3_fare = np.percentile(dados[dados['origem'] == 'treino'].Fare, 75)
iqr_fare = q3_fare - q1_fare

q1_age = np.percentile(dados[dados['origem'] == 'treino'].Age, 25)
q3_age = np.percentile(dados[dados['origem'] == 'treino'].Age, 75)
iqr_age = q3_age - q1_age

# Estimating upper limit to both ('Fare'/'Age')
outliers_fare_sup = q3_fare + 1.5 * iqr_fare
outliers_age_sup = q3_age + 1.5 * iqr_age

# Creates a mask (idx_outliers) so I can easily drop the outliers, if it is needed
mask = dados[dados['origem'] == 'treino'].index
mask = (dados.loc[mask, 'Fare'] >= outliers_fare_sup) | (dados.loc[mask, 'Age'] >= outliers_age_sup)
idx_outliers = dados[dados['origem'] == 'treino'].loc[mask].index



####
# MODELLING
####

# Adjusting datasets
train = dados[dados['origem'] == 'treino'].copy()
train.drop(['origem', 'PassengerId'], inplace = True, axis = 1)
test = dados[dados['origem'] == 'teste'].copy()
test.drop(['origem', 'Survived', 'PassengerId'], inplace = True, axis = 1)

# Dependent and independent variables
target = ['Survived']
train_data = train.drop(target, axis = 1)
test_data = train[target]
tamanho_treino = 0.8

# Train/test split
treino_x, teste_x, treino_y, teste_y = train_test_split(train_data, test_data, train_size = tamanho_treino, random_state = 1)


#### TEST MODEL 1 - XGBoost (XGBClassifier) - score 0.77751 (r2 0.236624, cv 0.821657)
# Model parameters
nome_modelo = datetime.now().strftime("%Y%m%d-%H%M")  # I like to record the inicial time for the model's name at the end
classif_xgb = xgb.XGBClassifier(booster = "gbtree", # gbtree, dart
                                tree_method = "approx", # hist, approx
                                n_estimators = 1000,
                                early_stopping_rounds = 300,
                                device="cuda",
                                nthread = 12,
                                eta = 0.01, #learning rate (0.0-1.0)
                                max_depth = 3, # default 1
                                max_leaves = 5,
                                objective = 'binary:logistic', # binary:logistic (returns probability), binary:logitraw (retorns score befor log transform), binary:hinge is default
                                eval_metric = 'error', # used in binary classification
                                seed = 1)

# Model fitting
classif_xgb.fit(treino_x, treino_y,
            eval_set = [(treino_x, treino_y), (teste_x, teste_y)],
            verbose = 0)

# Model cross-validation (assessing the model's robustness)
cv_scores_xgb_treino = cross_val_score(classif_xgb, treino_x, y = treino_y,
                                       cv = 7, # None: default 5-fold cross validation
                                       n_jobs = 12,
                                       verbose = 0,
                                       params = {'eval_set':[(treino_x, treino_y), (teste_x, teste_y)], 'verbose':0}, 
                                       error_score = 'raise')

# Assessment of the prediction
ypred_xgb = classif_xgb.predict(teste_x)
score_xgb_r2 = r2_score(teste_y, ypred_xgb)
print(f'\nCoefficient of Determination (R2): {score_xgb_r2:.6f}')
print(f'Cross-validation score: {cv_scores_xgb_treino.mean():.6f}')

# Generating a confusion matrix
conf_matrix = metrics.confusion_matrix(teste_y.Survived.values.astype('int'), ypred_xgb, labels=[1, 0])
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix,
            annot = True,
            cmap = 'Oranges',
            xticklabels = ['Survived', 'Perished'],
            yticklabels = ['Survived', 'Perished'])
plt.title(f'Confusion Matrix - Model XGBoost {nome_modelo}', fontsize=12)
plt.xlabel('Test data', fontsize=12)
plt.ylabel('Predicted data', fontsize=12)
plt.show()

# SUBMISSION
# Predicts the target on test data
ypred_xgb_final = classif_xgb.predict(test)

# Generates the name to save file
print(f'Prediction in use has record #{nome_modelo}.')
print("Compiling filename to submission...")
ext_final = (".csv")
save_final = ("XGBClassifier_predicted_"+(nome_modelo + ext_final))
print("Generating submission file named " + save_final)
# Reads submission file, records the obtained data and rewrites the file with it
submissao = pd.read_csv("./dados/gender_submission.csv")
submissao["Survived"] = ypred_xgb_final
submissao.to_csv("./resultados/"+save_final, index=False)
print("\nSuccess. File is in directory ('./resultados/').")


#### TEST MODEL 2 - CatBoost (CatBoostClassifier) - score 0.79904 (r2 0.144094, cv 0.828660)
# Model parameters
nome_modelo = datetime.now().strftime("%Y%m%d-%H%M")  # I like to record the inicial time for the model's name at the end
classif_cat = CatBoostClassifier(loss_function='Logloss', # https://catboost.ai/en/docs/concepts/loss-functions-classification#usage-information
                                 eval_metric = 'Logloss', # Logloss, AUC, MAPE, Poisson, Precision, Accuracy, R2, MedianAbsoluteError, PairAccuracy, PrecisionAt https://catboost.ai/en/docs/references/custom-metric__supported-metrics
                                 iterations = 1000,
                                 learning_rate = 0.01,
                                 random_seed = 1,
                                 bootstrap_type = 'MVS', #Bayesian (log), Bernoulli (stochastic), MVS (variance), Poisson (Poisson distribution)
                                 bagging_temperature = 7, 
                                 depth = 10,
                                 early_stopping_rounds = 500,
                                 thread_count = 12,
                                 task_type = 'CPU', 
                                 gpu_ram_part = 0.2,
                                 target_border = 0.5, 
                                 grow_policy = 'Lossguide', # Lossguide, Depthwise, SymmetricTree
                                 min_child_samples = 15, # default 1
                                 max_leaves = 20, # default 31
                                 boosting_type = 'Plain', # https://catboost.ai/en/docs/references/training-parameters/common#boosting_type
                                 score_function = 'L2' # L2, NewtonL2
                                 )

# Model fitting
classif_cat.fit(treino_x, treino_y,
            eval_set = (teste_x, teste_y),
            verbose = 100)

# Model cross-validation (assessing the model's robustness)
cv_scores_cat = cross_val_score(classif_cat, treino_x, y = treino_y,
                                       cv = 7, # None: default 5-fold cross validation
                                       n_jobs = 12,
                                       verbose = 0,
                                       params = {'eval_set':(teste_x, teste_y), 'verbose':0}, 
                                       error_score = 'raise')

# Assessment of the prediction
ypred_cat = classif_cat.predict(teste_x)
score_cat_r2 = r2_score(teste_y, ypred_cat)
print(f'\nCoefficient of Determination (R2): {score_cat_r2:.6f}')
print(f'Cross-validation score: {cv_scores_cat.mean():.6f}')


# Confusion Matrix
conf_matrix = metrics.confusion_matrix(teste_y.Survived.values.astype('int'), ypred_cat, labels=[1, 0])
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix,
            annot = True,
            cmap = 'Oranges',
            xticklabels = ['Survived', 'Perished'],
            yticklabels = ['Survived', 'Perished'])
plt.title(f'Confusion Matrix - model CatBoost {nome_modelo}', fontsize=12)
plt.xlabel('Test data', fontsize=12)
plt.ylabel('Predicted data', fontsize=12)
plt.show()


# SUBMISSION
# Predicts the target on test data
ypred_cat_final = classif_cat.predict(test)
# Generates the name to save file
print(f'Prediction in use has record #{nome_modelo}.')
print("Compiling filename to submission...")
ext_final = (".csv")
save_final = ("CatBoostClassifier_predicted_"+(nome_modelo + ext_final))
print("Generating submission file named " + save_final)
# Reads submission file, records the obtained data and rewrites the file with it
submissao = pd.read_csv("./dados/gender_submission.csv")
submissao["Survived"] = ypred_cat_final
submissao.to_csv("./resultados/"+save_final, index=False)
print("\nSuccess. File is in directory ('./resultados/').")


#### TEST MODEL 3 - LightGBMt (LGBMClassifier) - score 0.78468 (r2 0.028431, cv 0.839978)
# Model parameters
nome_modelo = datetime.now().strftime("%Y%m%d-%H%M")  # I like to record the inicial time for the model's name at the end, as you already now it.
classif_lgbm = LGBMClassifier(boosting_type = 'gbdt', # 'gbdt' default, 'rf'
                              num_leaves = 16, # default 31
                              max_depth = 3,
                              learning_rate = 0.01,
                              n_estimators = 800, # default 100
                              objective = 'binary',
                              min_split_gain=0.02, 
                              min_child_weight=0.022,
                              min_child_samples=20, # default
                              subsample=0.6, # default 1
                              random_state=1,
                              n_jobs=12,
                              importance_type='split',
                              metric = 'binary_logloss' # 'map', 'auc', 'average_precision', 'binary_logloss', 'binary_error, 'auc_mu', 'logloss'
                              )

# Model fitting
classif_lgbm.fit(treino_x, treino_y,
            eval_set = (teste_x, teste_y))

# Model cross-validation (assessing the model's robustness)
cv_scores_lgbm = cross_val_score(classif_lgbm, treino_x, y = treino_y,
                                       cv = 5, # None: default 5-fold cross validation
                                       n_jobs = 12,
                                       verbose = 0,
                                       params = {'eval_set':(teste_x, teste_y)}, 
                                       error_score = 'raise')

# Assessment of the prediction
ypred_lgbm = classif_lgbm.predict(teste_x)
score_lgbm_r2 = r2_score(teste_y, ypred_lgbm)
print(f'\nCoeficiente de determinação: {score_lgbm_r2:.6f}')
print(f'Score de validação cruzada: {cv_scores_lgbm.mean():.6f}')


# Confusion Matrix
conf_matrix = metrics.confusion_matrix(teste_y.Survived.values.astype('int'), ypred_lgbm, labels=[1, 0])
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix,
            annot = True,
            cmap = 'Oranges',
            xticklabels = ['Survived', 'Perished'],
            yticklabels = ['Survived', 'Perished'])
plt.title(f'Confusion Matrix - model LightGBM {nome_modelo}', fontsize=12)
plt.xlabel('Test data', fontsize=12)
plt.ylabel('Predicted data', fontsize=12)
plt.show()


# SUBMISSION
# Predicts the target on test data
ypred_lgbm_final = classif_lgbm.predict(test).astype(int)
# Generates the name to save file
print(f'Prediction in use has record #{nome_modelo}.')
print("Compiling filename to submission...")
ext_final = (".csv")
save_final = ("LGBMClassifier_predicted_"+(nome_modelo + ext_final))
print("Generating submission file named " + save_final)
# Reads submission file, records the obtained data and rewrites the file with it
submissao = pd.read_csv("./dados/gender_submission.csv")
submissao["Survived"] = ypred_lgbm_final
submissao.to_csv("./resultados/"+save_final, index=False)
print("\nSuccess. File is in directory ('./resultados/').")


#### TEST MODEL 4 - Random Forest (RandomForestClassifier) - score 0.79186 (r2 0.144094, cv 0.841367)
# Model parameters
nome_modelo = datetime.now().strftime("%Y%m%d-%H%M")  # Again.
classif_skl_rf = RandomForestClassifier(n_estimators = 1600,
                                        criterion = 'entropy', # 'gini' default, 'entropy', 'log_loss'
                                        max_depth = 21, # default None
                                        min_samples_split = 6, # default 2
                                        min_samples_leaf = 1, # default 1
                                        max_features='sqrt', # 'sqrt' default, 'log2, None
                                        bootstrap = True, # se 'False' usa o dataset inteiro para montar cada árvore
                                        oob_score = False, # Utilizado se bootstrep = True, usa uma métrica para o score geral (algo tipo r2score(y_true, y_pred))
                                        max_samples = 0.4, # usa com bootstrap = True, percentual de utilização amostral de X para treinar cada estimamdor
                                        n_jobs = 12,
                                        random_state = 1,
                                        verbose = 2,
                                        warm_start = False, # default, se True reutiliza a solução do último fit e adiciona mais estimadores ao agrupado
                                        )

# Model fitting
classif_skl_rf.fit(treino_x, treino_y)

# Model cross-validation (assessing the model's robustness)
cv_scores_skl_rf = cross_val_score(classif_skl_rf, treino_x, y = treino_y, # estimador (usado no fit), X, y (se existente)
                                       cv = 5, # None: default 5-fold cross validation
                                       n_jobs = 12,
                                       verbose = 0,
                                       error_score = 'raise')

# Assessment of the prediction
ypred_skl_rf = classif_skl_rf.predict(teste_x)
score_skl_rf = r2_score(teste_y, ypred_skl_rf)
print(f'\nCoeficiente de determinação: {score_skl_rf:.6f}')
print(f'Score de validação cruzada: {cv_scores_skl_rf.mean():.6f}')


# Confusion Matrix
conf_matrix = metrics.confusion_matrix(teste_y.Survived.values.astype('int'), ypred_skl_rf, labels=[1, 0])
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix,
            annot = True,
            cmap = 'Oranges',
            xticklabels = ['Survived', 'Perished'],
            yticklabels = ['Survived', 'Perished'])
plt.title(f'Confusion Matrix - model Random Forest (sklearn) {nome_modelo}', fontsize=12)
plt.xlabel('Test data', fontsize=12)
plt.ylabel('Predicted data', fontsize=12)
plt.show()


# SUBMISSION
# Predicts the target on test data
ypred_skl_rf_final = classif_skl_rf.predict(test).astype(int)
# Generates the name to save file
print(f'Prediction in use has record #{nome_modelo}.')
print("Compiling filename to submission...")
ext_final = (".csv")
save_final = ("RFClassifier_predicted_"+(nome_modelo + ext_final))
print("Generating submission file named " + save_final)
# Reads submission file, records the obtained data and rewrites the file with it
submissao = pd.read_csv("./dados/gender_submission.csv")
submissao["Survived"] = ypred_skl_rf_final
submissao.to_csv("./resultados/"+save_final, index=False)
print("\nSuccess. File is in directory ('./resultados/').")



#### TEST MODEL 5 - Arithmetic ensamble from models Random Forest and CatBoost), just for example
nome_modelo = datetime.now().strftime("%Y%m%d-%H%M")  # You know it.
best = (ypred_skl_rf_final + ypred_cat_final)/2
mask = best == 0.5
best[mask] = 1
best = best.astype(int)
print("Compiling filename to submission...")
ext_final = (".csv")
save_final = ("Ensemble_"+(nome_modelo + ext_final))
print("Generating submission file named " + save_final)
# Reads submission file, records the obtained data and rewrites the file with it
submissao = pd.read_csv("./dados/gender_submission.csv")
submissao["Survived"] = best
submissao.to_csv("./resultados/"+save_final, index=False)
print("\nSuccess. File is in directory ('./resultados/').")

