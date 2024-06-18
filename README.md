# kaggle-housing-price

![Example Image](images/kaggle_5407_media_housesbanner.png)

In this project, I analyse the Ame's housing dataset to predict prices of housing in Boston. The goal is to leverage powerful Machine Learning techniques to improve prediction accuracy.

You can take a look at the full report detailing all the processing steps and modelling to submit my predictions to the Kaggle competition [here](reports/report-kaggle-housing-GV-3-robustscaler-clean.ipynb)

After preprocessing and feature engineering, I train and fine tune 6 models (XGboost, light gradient boost, random forests, Lasso, Ridge Kernel, Support vector machines). I then blend their predictions to optimize perfomances, reaching a rmse of 0.12557 on the test set and the top 15% on the Kaggle competition.

