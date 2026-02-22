# Review Predictor

## Description

This is a machine learning, web application that allows the user to play against an trained model to predict the review score based off of its text alone. The player and model score a point when they are close to the actual score (within one point range) and gain more points when they are spot on.

## Dependencies and Running the Application

### Installing Dependencies

Running the following will install everything needed for both python and npm packages:

``` #bash
make setup
```

### Running the Program

To launch the full application, run the following command at the root folder of the project, along with setting up a venv for the server. Note that this will take a few minutes to complete:

```bash
make run
```

For development purposes there are other options to run parts of the application

Running `npm run build` to rebuild the frontend files.

Running `npm start` to quickly test/dev the frontend

Running `python3 server.py` in the server file will activate the server

## Training and Using New Models

### How to Train Your Model

(No affiliation with How to Train your Dragon)

Use the `train.py` file in the server folder to train new ML models.

```bash
python3 train.py
```

There are several parameters that can be passed in to fine tune your training. To see all possible arguments, run train.py with the `-h` flag to see all arguments.
Not all arguments need to be supplied a value by the user, and the program will default to selected values if not provided one.
Once done training, you can save the model and use it in the application by following the below section. If you choose _not_ to save the model **it cannot be recovered and will be lost**.

### Using Different Models

Currently the model used is handled i the CONSTANTS.py file. To use a model, update the MODEL_NAME to use the name of the .pth file in the model_files/models folder

A future update will aim to upgrade this to an argument passed on project start or handled in the UI itself

### Dataset used

As of writing, I have not included a proper way to get the training set to others, since its too big for github. (I plan on fixing that soon)
For now use the following link, and extract files into the server/model_files/data.
It should be two files:
test-00000-of-00001.parquet
train-00000-of-00001.parquet

![Link to dataset](https://huggingface.co/datasets/Yelp/yelp_review_full)

## Future Plans

* Add a make command to download the dataset used instead of manually
* Fix TechDebt (TODOs) to ensure code quality is up to date
* Include more test files for the frontend. Perhaps the back end as well.
* Allow users to swap what model is being used during run time
* Look into future feature updates
  * Get this project hosted online so people can try it out without having to build the project
  * A database to save scores, and a scoreboard would be nice
