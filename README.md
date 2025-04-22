# What will my recipe be rated on food.com?
analysis on recipes/reviews on food.com

## Checkpoint
1. We will be analyzing the **Recipes and Ratings** dataset as we feel this will apply to a larger audience. 
2. Plotly (see below)
3. We will be trying to predict the `rating` column. This will be a regression problem as the ratings are numerical, ranging from 0 to 5. 

# Introduction
The objective of this project is to analyze past recipes/reviews to predict what a recipe will be rated on food.com. Our first dataset contains information about recipes, `recipes`, and the second is about the interactions with these recipes on food.com, `interactions`. We will discuss our datasets further below and how we used them to solve our problem. 

To approach this problem, we will first clean the data. Then, analyze specific factors that play a role in our `ratings` predictions, and test our models on these factors. 

This question is significant because if you're an aspiring chef, you want to ensure your meals appeal to a larger audience and also build a reputation on review websites like Food.com. Knowing how certain dishes perform would greatly impact the recipe choices chefs make. 

Our first dataset, `recipes`, contains 83782 rows and 12 columns:

| Column Name     | Description                                                                                                                                         |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `'name'`            | Recipe name                                                                                                                                        |
| `'id'`              | Recipe ID                                                                                                                                          |
| `'minutes'`         | Minutes to prepare recipe                                                                                                                          |
| `'contributor_id'`  | User ID who submitted this recipe                                                                                                                  |
| `'submitted'`       | Date recipe was submitted                                                                                                                          |
| `'tags'`            | Food.com tags for recipe                                                                                                                           |
| `'nutrition'`       | Nutrition information in the form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)] |
| `'n_steps'`         | Number of steps in recipe                                                                                                                          |
| `'steps'`           | Text for recipe steps, in order                                                                                                                    |
| `'description'`     | User-provided description                                                                                                                          |


Our second dataset, `interactions`, contains 731927 rows and 5 columns:

| Column Name | Description            |
|-------------|------------------------|
| `'user_id'`     | User ID                |
| `'recipe_id'`   | Recipe ID              |
| `'date'`        | Date of interaction    |
| `'rating'`      | Rating given           |
| `'review'`      | Review text            |

# Data Cleaning and Exploratory Data Analysis
## Cleaning 
1. Left merge the `recipes` and `interactions` datasets.
2. In this merged dataset, `df`, we fill all '0' ratings with `np.nan`.
   - The '0' ratings are probably missing values, since food.com uses a scale of 1-5. Therefore, the '0' should not play a role in our calculations.
3. Next, we calculate the average rating per recipe, and store it as a series `avg_rating`.
4. We add the series, `avg_rating`, back to the merged df, `df`.
5. Dropping rows that do not have a rating, as they will not be relevant to our calculations.
6. Dropping `'description'`, `'review'`, `'tags'`, `'ingredients'`, `'nutrition'` and `'steps'` columsn are they are not relevant to our calculations.
7. Created a new `'month'` column to use in further analysis.

As we can see below, we have 51832 values of '0' to turn to `np.nan`. 

|   rating |   count |
|---------:|--------:|
|        5 |  523620 |
|        4 |  112393 |
|        0 |   51832 |
|        3 |   25054 |
|        1 |    9718 |
|        2 |    9310 |

This is how the first 3 row of the merged df `df` looks like. 

| name                                 |     id |   minutes |   contributor_id | submitted   |   n_steps |   n_ingredients |   user_id |   recipe_id | date                |   rating |   avg_rating |   month |
|:-------------------------------------|-------:|----------:|-----------------:|:------------|----------:|----------------:|----------:|------------:|:--------------------|---------:|-------------:|--------:|
| 1 brownies in the world    best ever | 333281 |        40 |           985201 | 2008-10-27  |        10 |               9 |    386585 |      333281 | 2008-11-19 00:00:00 |        4 |            4 |      11 |
| 1 in canada chocolate chip cookies   | 453467 |        45 |          1848091 | 2011-04-11  |        12 |              11 |    424680 |      453467 | 2012-01-26 00:00:00 |        5 |            5 |       1 |
| 412 broccoli casserole               | 306168 |        40 |            50969 | 2008-05-30  |         6 |               9 |     29782 |      306168 | 2008-12-31 00:00:00 |        5 |            5 |      12 |





## Exploratory Data Analysis
### Univariate Analysis

 <iframe
 src="assets/univariate-ratings.html"
 width="800"
 height="600"
 frameborder="0"
 ></iframe>

Exploring the distribution of the average ratings:

<iframe
 src="assets/univariate-avgratings.html"
 width="800"
 height="600"
 frameborder="0"
 ></iframe>

We see that the majority of ratings are above 4.5. Although one indication may be that all the recipes on the website are that good, it's unlikely. This questions the scale, bias and judgment of those who reviewed these recipes, as most ratings are very high. However, following our objective, this does not affect our calculation as we are interested in how an aspiring chef can achieve high ratings on their dishes, regardless of the scale. 

### Bivariate Analysis

<iframe
 src="assets/bivariate-n_steps-rating.html"
 width="800"
 height="600"
 frameborder="0"
 ></iframe>

 <iframe
 src="assets/bivariate-minutes-rating.html"
 width="800"
 height="600"
 frameborder="0"
 ></iframe>

## Interesting Aggregates

## Imputation (Optional? delete if so)

# Framing a Prediction Problem
Using linear regression, our model will aim to predict a recipe's rating. We are treating the rating as a number between the range of [1, 5]. This number can be in the form of a float, which makes this a **regression** problem. 

- Make sure to state whether you are performing binary classification or multiclass classification.
- Report the response variable (i.e., the variable you are predicting) and why you chose it, the metric you are using to evaluate your model, and why you chose it over other suitable metrics (e.g., accuracy vs. F1-score).
To predict the rating, the factors that will be taken into consideration will be everything except the rating of the dish. Since the purpose of this objective is to learn what leads to a higher rating. (something like this)

# Baseline Model
Our initial phase will include implementing a baseline model. We will do this by using a **Linear Regression** model on the following features: 
1. model0 - `'minutes'` (quantitative)
2. model1 - `'n_steps'` (quantitative)
3. model2 - `'minutes + n_ingredients'` (quantitative + quantitative)

### Comparing models:
`model2` will represent our Baseline model, using features minutes and n_ingredients. 
Below are the results (MSE) of the 3 models. 
```
{'minutes': 0.4977188447490406,
 'n_steps': 0.5055249094062664,
 'minutes + n_ingredients': 0.4937348049211309}
```
Therefore, our baseline model scored a MSE of approximately 0.49373, which is acceptable. Since the range of the `'rating'` is [1-5], $$\sqrt{0.49373} ≈ 0.70$$. This means a true rating of 4.0 can lie in the range of [3.3, 4.7]. However, even though we are below 1, there is still room for improvement. 

We have chosen these features, because (look at graphs)

- Both now and in Step 5: Final Model, make sure to evaluate your model’s ability to generalize to unseen data!
## Performance of Model
# Final Model
