# NTU DLWeek Hackathon 
- Purely proof of concept to parse backend ML "Pipeline" to final frontend product

- - Run data_ingestion.py > convert xlsx dataset to json dump for model to interpret

- Run train.py > train xgboost model on train set (0.8) and validation set (0.2) 

Training RMSE: The training RMSE of 1.01228 is fairly high but acceptable given such a small(SHAG LA TOUGH) dataset
Evaluation RMSE: The evaluation RMSE of 0.43760 is much lower, which suggests that the model is performing well on the validation data and generalizing effectively. This is a positive sign, indicating that the model is not overfitting and is able to predict future data with a relatively low error.

- Run predict.py to get prediction for each tool's ideal callibration date before the due date, in order not to miss deadlines. the csv file will be then parsed to the front end to be displayed and outputted
