# Review_Predictor

## Description

This is a machine learning, web application that allows the user to play against an trained model to predict the review score based off of its text alone. The player and model score a point when they are close to the actual score (within one point range) and gain more points when they are spot on.

## Dependencies and Running the Application

### Installing Dependencies

pytorch, sklearn, spacy, pandas, possibly en_core_web_sm for spacy

npm install for Node Modules

![Link to dataset](https://huggingface.co/datasets/Yelp/yelp_review_full)

### Running the Program

To launch the full application, run the following command at the root folder of the project:

```bash
./run.sh
```

The bash file may need execution privilges in order to run properly. The below command will apply said privileges (You may need to 'sudo' the command):

```bash
chmod +x run.sh
```

For development purposes there are other options to run parts of the application

Running ```npm run build``` to rebuild the frontend files.

Running ```npm start``` to quickly test/dev the frontend

Running ```python3 server.py``` in the server file will activate the server

## Training and Using New Models

### How to Train Your Model

(No affiliation with How to Train your Dragon, unfortunately.)

Use the train.py file in the server folder to train new ML models.

```bash
python3 train.py
```

There are several parameters that can be passed in to fine tune your training. To see all possible arguments, run train.py with the `-h` flag to see all arguments.
Not all arguments need to be supplied a value by the user, and the program will default to selected values if not provided one.
Once done training, you can save the model and use it in the application by following the below section. If you choose _not_ to save the model **it cannot be recovered and will be lost**.

### Using Different Models

Currently the model used is handled i the CONSTANTS.py file. To use a model, update the MODEL_NAME to use the name of the .pth file in the model_files/models folder

A future update will aim to upgrade this to an argument passed on project start or handled in the UI itself

## Notes

This project is functional, however it still is a WIP (as is this [slightly less messy than before] README file).
The UI has received updates, the only other main frontend task is adding an info page
Plus some TODO's in the backend for further QOL before I'd consider this project fully complete
