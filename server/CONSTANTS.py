from pathlib import Path

#TODO: Picking a model should not be hardcoded. Either its passed in as an argument or we should have a list of models to choose from in the actual frontend. That'd be cool.
#Possibly use a config file for the below? 
MODEL_NAME = "dummy.pth"
MODEL_PATH = Path('server/model_files/models')
TEST_FILE_PATH = 'server/model_files/data/test-00000-of-00001.parquet' # Uses server/ since this is used when running ./run.sh
TRAIN_FILE_PATH = 'model_files/data/train-00000-of-00001.parquet' # Used for training the model so user is assumed to be in server folder running train.py