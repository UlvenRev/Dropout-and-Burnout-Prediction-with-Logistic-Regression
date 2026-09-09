
# What this project is about 

This was my take at taking a simple `csv` dataset and training a machine learning model on it from scratch, so that it could predict the target values given a set of new data.

The dataset was taken from [Kaggle](https://www.kaggle.com/datasets/harpartapsingh13/student-burnout-and-dropout-risk-dataset)

Categories:
- Demographics - `Age`, `Gender`, `Year_of_Study`, `Department`, `Residence_Type`
- Academic - `Attendance_Percent`, `Study_Hours_Per_Day`, `Previous_GPA`, `Backlogs`
- Lifestyle - `Sleep_Hours`, `Screen_Time_Hours`, `Exercise_Freq_Per_Week`, `Social_Activity_Score`, `Part_Time_Job`
- Financial - `Family_Income_Bracket`, `Financial_Stress_Score`, `Family_Support_Score`  
- Mental Health - `Stress_Level`, `Anxiety_Score`, `Motivation_Score`, `Peer_Pressure_Score`, `Counseling_Access`
- Targets - `Dropout_Risk` (Yes/No), `Burnout_Level` (Low/Medium/High)


<br/>
<br/>

# How to navigate the files

I have 3 files in here. If you want to see:

- **A DETAILED DESCRIPTION** of all processes and decisions (a.k.a. beginner guide) -`exploration.ipynb`
- **HOW I FOUND BEST FEATURES** for both targets - `finding_best_features.ipynb`
- **A CLEAN FINAL SOLUTION** with code cells for **making your own predictions** - `Trained_Model.ipynb`


<br/>
<br/>

# Setup

`requirements.txt` contains all the Python packages you need to install in order to run the `.ipynb` files.

Run:
```

pip install -r requirements.txt

```

**or**

Run these files in Google Collab to use them straight away:

| Notebook | Open |
|---|---|
| `exploration.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UlvenRev/Dropout-and-Burnout-Prediction-with-Logistic-Regression/blob/main/Burnout%20%26%20Dropout%20Predictions/exploration.ipynb) |
| `finding_best_features.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UlvenRev/Dropout-and-Burnout-Prediction-with-Logistic-Regression/blob/main/Burnout%20%26%20Dropout%20Predictions/finding_best_features.ipynb) |
| `Trained_Model.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UlvenRev/Dropout-and-Burnout-Prediction-with-Logistic-Regression/blob/main/Burnout%20%26%20Dropout%20Predictions/Trained_Model.ipynb) |


<br/>
<br/>

# How to get predictions from your custom data in `Trained_Model.ipynb`

I will show this first, and if you want a detailed description of what was done in this project and why, look at the sections below.

1. Open `Trained_Model.ipynb`
2. Run all cells in order and wait for it to complete
3. **Scroll down** to the section "Test the model with your own data entry!"

### If you want to predict `Dropout_Risk`:
1. Change the values inside `[]` to your desired ones, following the commented range on the right
2. RUN this cell with `data_for_DR` **again** - this is important if you want the data to update
3. Run the code cell below with `predict_DR`

<img width="1041" height="687" alt="Screenshot 2026-09-09 124453" src="https://github.com/user-attachments/assets/3e4b3c21-c4b1-461d-98e4-d13d4765bbd2" />


### If you want to predict `Burnout_Level`:
Do the same thing as above:
1. Change the values inside `[]` to your desired ones, following the commented range on the right
2. RUN this cell with `data_for_BL` **again** - this is important if you want the data to update
3. Run the code cell below with `predict_BL`

<img width="867" height="693" alt="Screenshot 2026-09-09 124506" src="https://github.com/user-attachments/assets/c9232e57-b1bd-44f7-a67d-cb66c5cadb17" />


<br/>
<br/>

# Results of the project

- Trained the model to predict `Dropout_Risk` (binary classification) and `Burnout_Level` (multiclass classification)
- Selected low impact features using SHAP values and dropped these features for the final model
- Compared Logistic Regression against Random Forest model and XGBoost using ROC AUC score
- Logistic Regression performed best across 3 models, suggesting that the relationship between the features in the dataset is linear
- Logistic Regression predicts:
	- `Dropout_Risk` with AUC of 0.76 (Cross-Validation) and 0.80 (single validation)
	- `Burnout_Level` with AUC of 0.872 (Cross-Validation) and 0.875 (single validation)


<br/>
<br/>

# Key findings

From overall model-training process:
- It's better to deal with missing values **after** splitting the dataset into training and validation sets
- Cross-Validation will give a more accurate ROC AUC score than a single validation run
- Multiclass classification takes a slightly different approach than binary classification (meaning, some methods should be changed in order to work)

From using SHAP values:
- For both targets, `Department` and `Residence_Type` played almost no role in making predictions. They are the categorical variables, so it took some effort to encode them correctly with One-Hot Encoding, however in the end I just dropped them entirely.
- Both targets have almost the exact same redundant features. This could mean that:
	-  demographic/administrative features, like `Gender`, `Age`, `Department`, `Residence_Type`, `Year_of_Study`,  aren't strongly tied to psychological/behavioral outcomes here. 
	- `Dropout_Risk` and `Burnout_Level` targets have correlation between themselves, which is why they have the same features for predictions


<br/>
<br/>

# Limitations

- This Kaggle dataset is synthetic, meaning it doesn't represent actual world data, so predictions we get are also based on synthetic values and may not be real-world accurate.
- Small dataset (800 rows) creates natural variance in validation metrics. Cross-Validation is helpful in a way that it trains the model on more times and takes the average, but for a better-trained model a bigger dataset would be preferrable 
- `Burnout_Level` is treated as unordered multiclass (High, Low, Medium) rather than ordinal, which is a simplification from my side

<br/>
<br/>

# Deeper dive into the files - what I learned

## `exploration.ipynb`

This file is purely me exploring the dataset and going step-by-step through training the model for predicting `Dropout_Risk`.

The file contains a **detailed explanation** for each code cell, but I want to mention the biggest differences of this file with other two.
### Interesting stuff I learned:

<br/>

#### 1) Encoded vs String Targets

Both `Dropout_Risk` and `Burnout_Level` are of type `string`, and what you usually do is:
- **Encode** into numbers and then write as your target `y`
- **Leave it** as a string and write as your target `y`

Both approaches are really just a preference if you don't use these targets anywhere outside `sklearn`, because `sklearn` doesn't care whether `y` is a string or a number. 

So in the `exploration.ipynb` I encoded the targets, but in other two I left them as strings, which improved readability in the end (predictions are returned as "Yes/No" or "Low/Medium/High" and not numbers).

<br/>


#### 2) Choosing features through heatmap

Since this file was all about exploring, I used a **heatmap** from `seaborn` package to see **which features correlated the most.**

High correlation between two features means one of them is redundant and we don't need to take it into account, as we can derive it from the other one. 

But you can see that the heatmap is entirely blue, meaning at a **very low correlation values:**
<img width="682" height="584" alt="Screenshot 2026-09-09 124201" src="https://github.com/user-attachments/assets/9b7753a5-1be0-4ea8-a7c2-d178bb888262" />


So for this first run, I took **all features** into `X` for predicting `Dropout_Risk`.

<br/>


#### 3) Learning about Data Leakage

An important detail I learned was that you **need to deal with missing values AFTER splitting your dataset into training and validation sets.**

For example, if you have a column `Gender` which has values `Male/Female/Other` and some `NaN` values as well that **you want to remove**, you can either:
- Replace `NaN` with `Other` type (say a person didn't want to disclose their gender)
- **Derive** what a `NaN` value could be based on all values from the `Gender` column

With the first approach it **doesn't matter if you do it BEFORE or AFTER splitting your data** - you're just replacing `NaN`s with a constant. The entries will turn out the same either way.

But with the second approach, it's important to do it **AFTER splitting**. Assume you will replace `NaN` with **the most frequent value** (and it happens to be `Female`) - but you **calculated this value INCLUDING what will end up in your VALIDATION set.** And this is **leakage.** You should only rely on your **TRAINING set** in this case.

So it's always better to **split first, then deal with missing values**.

<br/>


#### 4) Dealing with Missing Values

After splitting the dataset into training and validation sets, I checked each column for its type and **analyzed how the missing values can be removed.**

In the code cells in `exploration.ipynb` you will see that I:
1. Take a column with type `string`
2. Check its range of values 
3. Decide whether I'm going to replace `NaN` with a **constant** or **derive a new value based on other values**
4. Check the range again to make sure the `NaN` is gone

It's a repetitive process in this file, as I wanted to make sure it worked for each column.

In the two other files, this process is automated!

<br/>


#### 5) What categorical variables are and how to preprocess them

I learned that **categorical variables** are the ones that take a limited number of values, e.g. like the column `Gender` with its `Male/Female/Other` values.

This variables can't stay like this if I want to train a model on them - they have to be encoded from strings into numbers the same way I did with the targets.

So I learned how to use Ordinal Encoding and One-Hot Encoding:
- `OrdinalEncoding` - encodes categorical variables which have clear order, e.g. Low (0) < Medium (1) < High (2), or for which you can give your own order, e.g. No (0) < Yes (1)
- `OneHotEncoding` - needed when you don't have an order, so instead of giving numbers 0, 1, 2 etc., you make **additional columns** where you mark which value is **present** with a 1 and which is not with 0

<br/>


#### 6) Cross-Validation

Cross-Validation is a reliable method to find out how well your model performs. 

What it does:
- Splits the dataset into 5 equal chunks
- Trains the model on 4 of them
- Compares how well it predicts the 5th chunk (validation step)
- Rotates to choose the next chunk as the one for validation, taking the 3 old chunks + one new chunk to train the model again
- Repeats the process until it covers all chunks 

This process is more reliable than doing just one training run on one big chunk of data and comparing it to the validation set.

<br/>


#### 7) How to compare and choose between different ML models 

Exactly where cross-validation became handy - I used it along with `ROC AUC score` collected across all 5 training runs cross-validation does.

And I used it on 3 different models:
- Logistic Regression - linear model for linear relationships
- Random Forest model - tree model with independent trees
- XGBoost - tree model, where each tree corrects the errors of its ancestor

Logistic Regression showed the **highest AUC score**, so I went with it from then on.

<br/>


#### 8) Grid Search and how it is used

I learned that if you have many parameters for a model you want to test, you can use `GridSearchCV`.

It's job is to take **every combination** of all parameters you pass with the parameter grid and test the model, returning you the best one.

In my example I test only the parameter `C` for `LogisticRegression`, as there are not many other parameters to test, but now I know that I can put as many things as I want into parameter grid without manually testing each one with my model.

<br/>


#### 9) Selecting best features using SHAP values

This is the most important bit I did in this project, since my goal was to find out how to select only the useful features for my model, and not just all of them.

Essentially, SHAP value is a score that tells you how much a specific feature contributed to a model's prediction, and by looking at the graph I simply removed the ones that had low contribution.

This actually raised the model's validation AUC in my case, though I made sure to confirm this improvement held up under cross-validation rather than trusting a single validation split.

<br/>
<br/>

## `finding_best_features.ipynb`

This file is much shorter than `exploration.ipynb` and has all the same code, but combined into separate functions.

<br/>


### How and why did I separate the training process into functions

So I have 3 functions in this file:
- `preprocess` - selects the target, features, splits the dataset into training and validation sets, deals with missing values, encodes categorical variables
- `get_model_predictions` - since I already know that Logistic Regression is the best model, I'm not comparing it with the other two and going straight to grid search to find the best `C` parameter. The function returns the model's AUC and predictions
- `shap_visualization` - draws a `beeswarm` plot to show SHAP values as in the other files

The benefit of such separation is that further on you can see me reuse the same code easily to test the different features quickly.

<br/>
<br/>

## `Trained_Model.ipynb`

This is the clean solution to the whole project!

It has the same neat organization into functions and code cells for training the model for predicting both targets.

<br/>


#### Getting custom predictions 

This was my original goal when I started training the model - in the end I wanted to be able to give it my own entry and get a prediction (something like a web-quiz "are you going to drop out?").

How to try the model with your own predictions is described in the beginning of this README!

