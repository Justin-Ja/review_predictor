# Review_Predictor

## Description

This is a machine learning, web application that allows the user to play against an trained model to predict the review score based off of its text alone. The player and model score a point when they are close to the actual score (within one point range) and gain more points when they are spot on.

## A Quick Note

This project is functional, however it still is a WIP (as is this mess of a README file). You can play against the AI to predict review scores, however the front-end (while functional) needs further UI/UX updates.

### Dependencies and Running

pytorch, sklearn, spacy, pandas, possibly en_core_web_sm for spacy

npm install for Modules

![Link to dataset](https://huggingface.co/datasets/Yelp/yelp_review_full)

npm run build to rebuild the file. python3 server.py to activate the server

npm start to quickly test/dev the frontend

run.sh may need execution privilges:

```bash
chmod +x run.sh
```
