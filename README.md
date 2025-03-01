# Baddie Borsch
Group: baddies in computing
- our name damn fire bro

# NTU DLWeek Hackathon 

- Problem Statement Track 5: Bosch
Ensuring precise and timely tool calibration is critical for operational efficiency and compliance, yet traditional tracking methods often lead to inefficiencies, missed deadlines, and potential risks. We seek innovative approaches to enhance calibration monitoring through data-driven insights, automation, and predictive analytics.
How might we transform tool calibration tracking to improve accuracy, streamline planning, and enable proactive decision-making? Consider intelligent alert systems, interactive dashboards, and cutting-edge technologies that redefine efficiency and reliability.
Can your solution set a new standard for calibration management?
- Proof of concept ML "pipeline" in the form of input, process, output to then parse data to the frontend portion

# Setting Up Dev Environment

- Create a python virtual environment (python -m venv venv)
- Enter virtual environment directory (cd /venv/Scripts) and run (./activate)
- Install pip packages and libraries (pip install -r requirements.txt) from main working directory 

# Running "Pipeline"

- Run data_ingestion.py > convert xlsx dataset to json dump for model to interpret
- Run train.py > train xgboost model on train set (0.8) and validation set (0.2) 
- Training RMSE: The training RMSE of 1.01228 is fairly high but acceptable given such a small(SHAG LA TOUGH) dataset
- Evaluation RMSE: The evaluation RMSE of 0.43760 is much lower, which suggests that the model is performing well on the validation data and generalizing effectively. This is a positive sign, indicating that the model is not overfitting and is able to predict future data with a relatively low error.
- Run predict.py to get prediction for each tool's ideal callibration date before the due date, in order not to miss deadlines. the csv file will be then parsed to the front end to be displayed and outputted
