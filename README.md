# TDDE16_Text-Mining_project
Project repository for the TDDE16 Text Mining course project.

Classification of song lyrics by mood with Naive-Bayes, Linear Support Vector Classification, and Multi-layer Perceptron Classifier.

## Prerequisites

### Dependency installation

```bash
python -m pip install -r requirements.txt
```

### Environment variables and API setup

The mood classification tags used in this project were fetched using the last.fm API.
Reproducing that requires a few preliminary setup steps, namely getting an API Key and a Shared Secret. Please refer to last.fm's API documentation [here](https://www.last.fm/api/authentication).

A .env file containing the environment variables is also needed. It needs to be placed at the root of the project's folder and should look like this:

    DATA_FOLDER=data // Location of the folder that contains/will contain the data, relative to the root of the project's folder
    RESULTS_FOLDER=results // Location of the folder that contains/will contain the results, relative to the root of the project's folder
    LAST_FM_KEY=your_lastfm_api_key
    LAST_FM_SECRET=your_lastfm_shared_secret

### Create "results" folder

Add the following hierarchy to the project folder:
    
    results/
        figures/
            best-params/
            default/
        reports/
            best-params/
            default/

This is needed to store the results of our experiments.

### Unzip the data

Unzip "lyrics-gold-labels.zip" into "data/Clean_data/".

> Note: You can skip over to the step "Running the experiments" once you're done with the setup if you're not interested in retrieving and cleaning the data yourself.

If you want to reproduce the whole tag scraping and data cleaning process, unzip the two files contained in AZLyrics.zip into "data/Raw_data".

## Lyrics data cleaning

```bash
python3 lyrics_english
```

## Tag fetching and cleaning

```bash
python3 fetch_tags
python3 clean_tags
```

## Running the experiments

```bash
python3 training.py
```
