from pathlib import Path

MODEL_NAME = "dummy.pth"
MODEL_PATH = Path('server/model_files/models')
TEST_FILE_PATH = 'server/model_files/data/test-00000-of-00001.parquet' # Uses server/ since this is used when running ./run.sh
TRAIN_FILE_PATH = 'model_files/data/train-00000-of-00001.parquet' # Used for training the model so user is assumed to be in server folder running train.py