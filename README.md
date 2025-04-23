# What will my recipe be rated on food.com?
Analysis on recipes/reviews on food.com


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
3. Next, we calculate the average rating per recipe and store it as a series `avg_rating`.
4. We add the series, `avg_rating`, back to the merged df, `df`.
5. Dropping rows that do not have a rating, as they will not be relevant to our calculations.
6. Dropping `'description'`, `'review'`, `'ingredients'`, and `'steps'` columsn are they are not relevant to our calculations.
7. Created a new `'month'` column to use in further analysis.
8. Extracting `'calories'` from the nutrituion column
9. Removing rows where `'minutes'` exceeds 25,000

As we can see below, we have 51832 values of '0' to turn to `np.nan`. 

|   rating |   count |
|---------:|--------:|
|        5 |  523620 |
|        4 |  112393 |
|        0 |   51832 |
|        3 |   25054 |
|        1 |    9718 |
|        2 |    9310 |

This is how the first 5 row of the merged df `df` looks like. 

| name                                 |     id |   minutes |   contributor_id | submitted   | tags                                                                                                                                                                                                                        | nutrition                                    |   n_steps |   n_ingredients |          user_id |   recipe_id | date                |   rating |   avg_rating |   month |   calories |
|:-------------------------------------|-------:|----------:|-----------------:|:------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------|----------:|----------------:|-----------------:|------------:|:--------------------|---------:|-------------:|--------:|-----------:|
| 1 brownies in the world    best ever | 333281 |        40 |           985201 | 2008-10-27  | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'for-large-groups', 'desserts', 'lunch', 'snacks', 'cookies-and-brownies', 'chocolate', 'bar-cookies', 'brownies', 'number-of-servings'] | [138.4, 10.0, 50.0, 3.0, 3.0, 19.0, 6.0]     |        10 |               9 | 386585           |      333281 | 2008-11-19 00:00:00 |        4 |            4 |      11 |      138.4 |
| 1 in canada chocolate chip cookies   | 453467 |        45 |          1848091 | 2011-04-11  | ['60-minutes-or-less', 'time-to-make', 'cuisine', 'preparation', 'north-american', 'for-large-groups', 'canadian', 'british-columbian', 'number-of-servings']                                                               | [595.1, 46.0, 211.0, 22.0, 13.0, 51.0, 26.0] |        12 |              11 | 424680           |      453467 | 2012-01-26 00:00:00 |        5 |            5 |       1 |      595.1 |
| 412 broccoli casserole               | 306168 |        40 |            50969 | 2008-05-30  | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                        | [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0]    |         6 |               9 |  29782           |      306168 | 2008-12-31 00:00:00 |        5 |            5 |      12 |      194.8 |
| 412 broccoli casserole               | 306168 |        40 |            50969 | 2008-05-30  | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                        | [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0]    |         6 |               9 |      1.19628e+06 |      306168 | 2009-04-13 00:00:00 |        5 |            5 |       4 |      194.8 |
| 412 broccoli casserole               | 306168 |        40 |            50969 | 2008-05-30  | ['60-minutes-or-less', 'time-to-make', 'course', 'main-ingredient', 'preparation', 'side-dishes', 'vegetables', 'easy', 'beginner-cook', 'broccoli']                                                                        | [194.8, 20.0, 6.0, 32.0, 22.0, 36.0, 3.0]    |         6 |               9 | 768828           |      306168 | 2013-08-02 00:00:00 |        5 |            5 |       8 |      194.8 |





## Exploratory Data Analysis

For our exploratory analysis, we plan to discover patterns in our data that will help guide the factors we select to investigate further. The figures will consist of univariate and bivariate analysis. 
### Univariate Analysis

 <iframe
 src="assets/univariate-ratings.html"
 width="800"
 height="600"
 frameborder="0"
 ></iframe>

From the above figure, we can see that most of the recipes were rated very highly. Majority recieved 5/5 as a rating. To further analyze this distribution, we looked at the same axis but replaced `'rating'` with the `'avg_rating'`. 

<iframe
 src="assets/univariate-avgratings.html"
 width="800"
 height="600"
 frameborder="0"
 ></iframe>

We see that the majority of ratings are above 4.5. Although one indication may be that all the recipes on the website are that good, it's unlikely. This questions the scale, bias, and judgment of those who reviewed these recipes, as most ratings are very high. However, following our objective, this does not affect our calculation as we are interested in how an aspiring chef can achieve high ratings on their dishes, regardless of the scale. 

### Bivariate Analysis

<iframe
 src="assets/bivariate-n_steps-rating.html"
 width="800"
 height="600"
 frameborder="0"
 ></iframe>

As seen in the above graph, the majority of the recipes with fewer steps result in a higher score. This may come from the fact that they are straightforward/easy to follow. However, we also see that recipes with a lot of steps can also result in high scores, showing us that complexity alone is not a sufficient factor. 

 <iframe
 src="assets/avgr-time.html"
 width="800"
 height="600"
 frameborder="0"
 ></iframe>

This graph shows us that the time spent cooking doesn't have a strong impact on the rating. Some dishes that take a very long time (> 5000 minutes) can also result in a high rating, which may come from showing more prep in the recipe, causing the taste/presentation of the food to increase. 

<iframe
 src="assets/avgr-cal.html"
 width="800"
 height="600"
 frameborder="0"
 ></iframe>

This graph shows us the average rating vs calories. Most fall under 2,000 calories. However, there is a sparse tail that exceeds 10,000 calories, and they are still within the [4, 5] range. Having calories alone is a weak indicator with no simple trend, which is why we later engineer cal_per_ingredient. 

 
## Interesting Aggregates

| minutes_bin / ing_quartile  |     Low |   Low-Mid |   Mid‑High |    High |   Overall |
|:--------------|--------:|----------:|-----------:|--------:|----------:|
| G1            | 4.73278 |   4.68674 |    4.73016 | 4.73671 |   4.71879 |
| G2            | 4.6832  |   4.67803 |    4.67388 | 4.69367 |   4.68147 |
| G3            | 4.68461 |   4.65426 |    4.64754 | 4.6874  |   4.66836 |
| G4            | 4.68361 |   4.65087 |    4.68433 | 4.67463 |   4.67072 |
| G5            | 4.64345 |   4.63962 |    4.65403 | 4.66742 |   4.6533  |
| Overall       | 4.70066 |   4.66493 |    4.6717  | 4.68205 |   4.67987 |

There seems to be a negative correlation between the number of minutes it takes to prepare the recipe and the average rating. In other words, users tend to prefer quicker recipes. Furthermore, there seems to be a preference for polarized numbers of ingredients. In other words, recipes with a relatively large or relatively small number of ingredients tend to be rated slightly higher than those with a moderate number of ingredients. However, in aggregating these features, there does also seem to be a slight preference against recipes that involve very few ingredients yet take many minutes to prepare, which partially counteracts the overall preference for a polarized number of ingredients.

## Imputation
Data imputation was not applicable for our project. For rows where there was a missing (i.e., NaN) value in the rating column, we could not impute a value for the rating because the goal of our project is to build a model that predicts rating values. Thus, any imputation in this column would improperly bias our final regression model. This means we have to drop rows with NaN rating. Once we do this, the only remaining NaNs in our dataframe are in the description and review columns. We do not use either of these features in our regression model, so there is no need for imputation here either.

# Framing a Prediction Problem
Using linear regression, our model will aim to predict a recipe's rating. We are treating the rating as a number between the range of [1, 5]. This number can be in the form of a float, which makes this a **regression** problem. 

- Make sure to state whether you are performing binary classification or multiclass classification.
- Report the response variable (i.e., the variable you are predicting) and why you chose it, the metric you are using to evaluate your model, and why you chose it over other suitable metrics (e.g., accuracy vs. F1-score).
To predict the rating, the factors that will be taken into consideration will be everything except the rating of the dish. Since the purpose of this objective is to learn what leads to a higher rating. (something like this)

# Baseline Model
Since this is a regression problem. Our initial phase will include implementing a baseline model. Our response variable is `'rating'` as we want to be able to predict how certain recipes will be viewed by others on food.com. We will do this by using a **Linear Regression** model on the following features: 
1. model0 - `'minutes'` (quantitative)
2. model1 - `'n_steps'` (quantitative)
3. model2 - `'minutes' + 'n_ingredients' + 'calories'` (all quantitative)

### Comparing models:
We will be using MSE, as it is able to penalize large errors more heavily. This was our choice since we predicted a 1-star on a 4-star recipe should be much worse than a 3-star on a 4-star recipe. 
`model2` will represent our Baseline model, using features minutes, n_ingredients and calories. 
Below are the results (MSE) of the three models. 
```
{'minutes': 0.5012293752319682,
 'n_steps': 0.5012364897342074,
 'minutes + n_ingredients'+ calories': 0.5012586066763299}
```
Therefore, our baseline model scored a MSE of approximately `0.5013`, which is acceptable. However, even though MSE is below 1, there is still room for improvement. 

We have chosen these features because we found them to have a correlation to the `avg_rating`, stronger than others (like month, which can be found in the ipynb).

- Both now and in Step 5: Final Model, make sure to evaluate your model’s ability to generalize to unseen data!

# Final Model
For the final model, we plan to improve the accuracy of the baseline model's prediction. The baseline model scored an MSE of 0.5013. For our final model, we plan to engineer 2 features:
1. `'tags_count'`
2. `'cal_per_ingredient'`

The `'tags_count'` is vital to the rating as the rating is a collection of reviews on food.com. This would be since recipes with more tags likely represent dishes that incorporate more styles, which may induce bias in reviewers, increasing their rating of the recipe. Having more information/tags potentially correlates with a chef caring more about their recipe, further leading to higher ratings, with the opposite also being valid. 

The `'cal_per_ingredient'` feature represents the calories per ingredient in the recipe. This could show how efficiently the recipe uses calories in its dish, trying to minimize the number of ingredients used. This also plays a role in the complexity of the dish, since more ingredients used may steer away users who prefer a simple dish, whilst maintaining their calorie intake. 

For this model, we iteratively tested N_estimaters from 50, 100, 150, and eventually narrowed it down as the model returned its most optimal one to 235. We continued this same approach for max_depth and min_samples_split, to which the data is listed below. We used a Random Forest Regressor as it can handle mixed feature types, and outliers (as we see in minutes) dont have as strong of an influence compared to linear models. 

## Results
After running and fine-tuning our hyperparameters, we were able to get our optimal results with the parameters below. 
```{'model__max_depth': 18, 'model__min_samples_split': 22, 'model__n_estimators': 235}```. 
The calculated MSE for our final model was `0.492`, showing a slight improvement from our baseline model. 

# Overall Conclusion 
Through this analysis, we discovered how the time taken, number of ingredients, and calories specifically played a major role in depicting the rating. Through our grid search and EDA, we would predict cal_per_ingredient and minutes to dominate. For future analysis, we think it would be beneficial to categorize the recipe types to perform `rating ` analysis. 
