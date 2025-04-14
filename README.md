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

Use train.py in the server folder

ADD MORE!!!

### Using Different Models

Currently the model used is handled i the CONSTANTS.py file. To use a model, update the MODEL_NAME to use the name of the .pth file in the model_files/models folder

A future update will aim to upgrade this to an argument passed on project start or handled in the UI itself

## Notes

This project is functional, however it still is a WIP (as is this mess of a README file). You can play against the AI to predict review scores.
The UI has received updates, the only other main frontend task is adding an info page
Plus some TODO's in the backend for further QOL
