---
layout: post
title: My first (and only) Kaggle Top 10%
featured-img: first_kaggle_cover
category: [kaggle]
mathjax: true
summary: What I've learned and how I succeeded in a big Kaggle competition. 
---

# My first (and only) Kaggle Top 10%

In this short post, I'll revisit my Top 10% solution to the [Santander Customer Satisfaction](https://www.kaggle.com/c/santander-customer-satisfaction) competition on Kaggle. I'll use the benefit of hindsight to analyze the competition and my performance. You can check the code at my [GitHub](https://github.com/gdmarmerola/santander-satisfaction) (README is written in Portuguese but the code is commented in english).

## Why is this relevant?

Kaggle is a great platform to learn about data science and practice real-world machine learning skills. Although doing ML in industry is not the same thing, Kaggle provides you with a solid base to do ML on your own and with inspiration to solve problems differently later on. I hope to give you a feel of what a Kaggle competition is a some modeling tips if you want to participate in the future.

## Introduction

"From frontline support teams to C-suites, customer satisfaction is a key measure of success. Unhappy customers don't stick around. What's more, unhappy customers rarely voice their dissatisfaction before leaving.

Santander Bank is asking Kagglers to help them identify dissatisfied customers early in their relationship. Doing so would allow Santander to take proactive steps to improve a customer's happiness before it's too late.

In this competition, you'll work with hundreds of anonymized features to predict if a customer is satisfied or dissatisfied with their banking experience."

Santander provided Kaggle with 371 features about clients who were either satisfied or dissatisfied with the bank. The data was not big, with the training set totalling 76,020 rows while the test set had 75,818 rows. This challenge presented itself as the perfect opportunity for me to compete, for the first time, in a *Featured* competition (the ones which count to rankings and have cash prizes), as the dataset was small and the problem was simple. My main goal with the competition was to learn how to use Gradient Boosting through the `xgboost` package, and increase my proficiency with hyperparameter optimization tools, such as `hyperopt`. 

Other people saw the same opportunity: it was the largest competition in terms of competitors at the time, with 5,123 teams. It was also my best result on Kaggle so far, where I got out ranked 470/5123 and with a Top 10% trophy.

## The data

The data was somewhat messy, with very few variables accounting for most of the importance and a heavy imbalance between positive and negative classes, with 3.95% of rows being part of the minority class. Users reported that identical rows had different ground-truths, which made the problem harder.

Santander tried to hide the features' meaning by giving uninformative names like "var15" to the features. However, users reported possible interpretations of the variables, like in this [discussion](https://www.kaggle.com/cast42/exploring-features/comments). People with finance background used this knowledge to their advantage.

Feature engineering did not play a huge part. Thus, results among competitors were very similar, with differences of the order of 0.1 percentage points in AUC.

## The competition

Kaggle has actually two leaderboards, Public and Private, where the Public leaderboard stays open throughout the competition and the Private leaderboard is revealed only at the end. Only the Private leaderboard counts toward points and prizes. As it was fairly easy to test even the most crazy ideas, [simple scripts](https://www.kaggle.com/zfturbo/to-the-top-v3/comments) started to emerge close to the top of the Public leaderboard, with many users reporting difficulties to beat them.

On the other hand, some noted that these scripts could be **overfitting** the public leaderboard, and would greatly suffer when the private leaderboard was revealed, due to possible differences between the public and private LB, caused by the target variable imbalance.

In the end of the competition, it became clear that this was true, and that a robust validation process was paramount. The shake-up (difference between public and private LB rankings) was very high due to the use of scripts, making some competitors fall over a thousand places in the LB. Fortunately, I trusted my CV and gained 470 positions, which put me in the Top 10%. 

![]({{ "assets/img/my_first_kaggle/kaggle-img-1.png" | absolute_url }})

As people mentioned in the forums, the biggest lesson to be taken from this competition is *trust your CV*.

## Modeling

As I wanted to use a very expressive model (`xgboost`) and tune the hell out of it (with the TPE algorithm from `hyperopt`), I needed to devise a robust validation process. One of Kaggle greatest competitors, [Lucas Eustaquio](https://www.kaggle.com/leustagos) (who, unfortunately, lost the battle against cancer during the competition), mentioned at an interview that validation is one of the things [he cared the most about](http://blog.kaggle.com/2016/02/22/profiling-top-kagglers-leustagos-current-7-highest-1/), being the first thing that he builds at the start of a competition.

In order to get stable AUC measurements (0.003 of AUC would mean 1,350 positions in the LB) and achieve my goals, I used two CV strategies to evaluate my models:

1. **1st Round:** I used a computationally cheaper 10-fold stratified CV in order to get a feel for the best feature engineering and hyperparameters of the models I tested.

2. **2nd Round:** with the narrow search space obtained in the previous round, I used an expensive 7-fold CV with 6 repetitions, where I would throw away the results of the best and worst folds.

I generated over 200 `xgboost` models with different parameters. To take advantage of this big model dictionary, I devised a simple greedy ensembling method, which would work as follows:

0. Initialize an empty ensemble and sort the model dictionary from best to worst with respect to CV results
1. Take the best model from the dictionary and calculate CV score if its predictions were added to the ensemble (simple average)
2. If CV score improves, add model to ensemble. If not, do nothing.
3. Remove the model from the dictionary and return to (1) until no model remains in the dictionary

This method is similar to the Ensemble Selection methodology from [Caruana et. al](http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf) but with less variance, as models have equal weight and are added only once to the ensemble. This was a good feature given the risk of overfitting in the competition. Seven models were selected for the ensemble to compose my final submission. The following plot shows the 2nd round of model search yielding over 100 different models:

![]({{ "assets/img/my_first_kaggle/kaggle-img-2.png" | absolute_url }})

In agreement with [previous research on hyperparameter tuning](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf), few hyperparameters really mattered. The TPE algorithm made good choices as most experiments yielded a good result.

## What I could've done better

From a business perspective, the difference between my model and the best model is irrelevant. In a business context, we would improve the model by collecting more variables or maybe help with devising good offers for the dissatisfied clients.

From the competition perspective, I could've spent a little bit more time in feature engineering, even if it was not super important for this competition. Also, and more importantly, other ensemble techniques such as Stacking, could've gone far as well. Using different algorithms in opposition as using one algorithm with slight hyperparameter variations could've contrbuted to adding beneficial variability to the ensemble.

Nevertheless, I'm satisfied with the result and with the learning opportunity. I've heard that in Kaggle competitions you always win, because you always take knowledge and experience home. 

## Conclusions

Kaggle is a great platform for learning cool stuff about machine learning and exercizing software engineering skills. I still haven't returned to the platform as a competitor, as I started to work in industry shortly after this competition. However, I'm always checking for new models and ideas in the community. I greatly recommend giving it a shot, and I hope that this post may help you with some inspiration in the future.
