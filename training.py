import re

import os
from dotenv import load_dotenv

# Loading environment variables
load_dotenv()
DATA_FOLDER = os.environ.get("DATA_FOLDER")
RESULTS_FOLDER = os.environ.get("RESULTS_FOLDER")

import time
import csv
from nltk.corpus import stopwords
import re
import string
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV

stop_words = set(stopwords.words("english"))

def tokenize_text(text):
    text = str(text)
    tokens = [token for token in text.split() if not token in stop_words]
    return tokens

def lyrics_preprocessing():
    with open(DATA_FOLDER + "/Clean_data/" + "lyrics-gold-labels.csv", "r") as data, \
        open(DATA_FOLDER + "/Clean_data/" + "lyrics-tok-gold-labels.csv", "w") as fout:
        writer = csv.writer(fout, delimiter=',')
        writer.writerow(["artist", "song_name", "lyrics", "mood"])
        for row in csv.reader(data, delimiter=','):
            lyrics = row[2]
            # Remove things like [Chorus] in the lyrics
            lyrics = re.sub(r" ?\[[^)]+\]", "", lyrics)
            # Remove punctuation
            lyrics = re.sub(r"[^\w\d\s\']+", "", lyrics)
            # Remove 'm 's ...
            lyrics = re.sub(r"\'[a-zA-Z]+", "", lyrics)
            lyrics = lyrics.lower()
            # Tokenize
            tokenized_lyrics = tokenize_text(lyrics)
            separator = ' '
            lyrics = separator.join(tokenized_lyrics)
            writer.writerow([row[0], row[1], lyrics, row[3]])

lyrics_preprocessing()

with open(DATA_FOLDER + "/Clean_data/" + "lyrics-tok-gold-labels.csv", "r") as data:
    # Total number of rows: 23755
    # We'll do a 80% / 20% split of the data (80% training data, 20% evaluation data)
    training_data = pd.read_csv(data, sep = ',', nrows = 19004)
    # Add an ID column
    training_data.insert(0, "id", range(1, 1 + len(training_data)))
    data.seek(0)
    test_data = pd.read_csv(
        data, header = 0, names = ["artist", "song_name", "lyrics", "mood"], sep = ',', skiprows = 19004
    ) # should give us 4751 rows
    # Add an ID column
    test_data.insert(0, "id", range(1, 1 + len(test_data)))

labels = ["happy", "angry", "sad", "relaxed"]

XTrain = training_data["lyrics"].values.astype('U')
yTrain = training_data["mood"]
XTest = test_data["lyrics"].values.astype('U') # Used for evaluation
yTest = test_data["mood"] # Used for evaluation


# Plot a visualization of the current data

fig, axes = plt.subplots(nrows=1, ncols=2)

yTrain.value_counts().plot(
    kind="bar",
    colormap="gray",
    ax=axes[0],
    sharey=True,
    sharex=True,
    xlabel="Mood",
    ylabel="Nb. of songs"
)
yTest.value_counts().plot(
    kind="bar",
    ax=axes[1],
    sharey=True,
    sharex=True,
    xlabel="Mood",
    ylabel="Nb. of songs"
)
axes[0].set_title("Nb. of songs in training data")
axes[1].set_title("Nb. of songs in test data")
fig.subplots_adjust(wspace = 0.4)
plt.show()
fig.tight_layout()
fig.savefig(RESULTS_FOLDER + "/figures/" + "data_count_plot.png", dpi = 199)


### TRAIN CLASSIFIERS WITH BAG OF WORDS VECTORS

vectorizers = {
    "CountVectorizer": CountVectorizer(),
    "TfidfVectorizer": TfidfVectorizer()
}

###
# RUN TESTS WITH UNBALANCED SET
###


### SEARCH BEST HYPERPARAMETERS

# ComplementNB

ComplementNB_parameter_space = {
    "complementnb__alpha": [1, 0.1, 0.01, 0.001, 0.0001],
}

# MultinomialNB

MultinomialNB_parameter_space = {
    "multinomialnb__alpha": [1, 0.1, 0.01, 0.001, 0.0001],
}

# MLPClassifier

MLP_parameter_space = {
    "mlpclassifier__activation": ["logistic", "tanh", "relu"],
    "mlpclassifier__alpha": [0.0001, 0.05],
}



filename = RESULTS_FOLDER + "/reports/default/" + "classification_reports-unbalanced.csv"

classifiers = {
    "LinearSVC": LinearSVC(dual=False),
    "ComplementNB": ComplementNB(),
    "MLPClassifier": MLPClassifier(verbose = True)
}

for class_key, classifier in classifiers.items():
    for vect_key, vectorizer in vectorizers.items():
        print("\n### %s with %s - Unbalanced dataset ###\n" % (class_key, vect_key))

        # Build the pipeline, train the model and predict
        clf = make_pipeline(
            vectorizer,
            StandardScaler(with_mean=False),
            classifier
        )
        clf.fit(XTrain, yTrain)
        predictions = clf.predict(XTest)

        # Reports
        report = classification_report(yTest, predictions, target_names = labels, output_dict = True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(filename, header=True, index=True, sep='\t', mode='a')

        plot_confusion_matrix(clf, XTest, yTest)  
        plt.savefig(
            RESULTS_FOLDER + "/figures/default/" + class_key.lower() + "-" + vect_key.lower() + "-conf_matrix.png"
        )


filename = RESULTS_FOLDER + "/reports/best-params/" + "classification_reports-unbalanced.csv"

print("### FINDING BEST PARAMETERS FOR UNBALANCED DATASET... ###")

classifiers = {
    "ComplementNB": ComplementNB(),
    "MLPClassifier": MLPClassifier(verbose = True)
}

for class_key, classifier in classifiers.items():
    if (class_key == "ComplementNB"):
        parameter_space = ComplementNB_parameter_space
    elif (class_key == "MLPClassifier"):
        parameter_space = MLP_parameter_space

    for vect_key, vectorizer in vectorizers.items():
        print("\n### %s with %s - Unbalanced dataset ###\n" % (class_key, vect_key))

        # Build the pipeline, train the model and predict
        pipeline = make_pipeline(
            vectorizer,
            StandardScaler(with_mean=False),
            classifier
        )
        clf = GridSearchCV(pipeline, parameter_space, n_jobs=-1, cv=3)
        clf.fit(XTrain, yTrain)

        print("Best parameters found:\n", clf.best_params_)

        predictions = clf.predict(XTest)

        # Reports
        report = classification_report(yTest, predictions, target_names = labels, output_dict = True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(filename, header=True, index=True, sep='\t', mode='a')

        plot_confusion_matrix(clf, XTest, yTest)  
        plt.savefig(
            RESULTS_FOLDER + "/figures/best-params/" + class_key.lower() + "-" + vect_key.lower() + "-conf_matrix.png"
        )


###
# RUN TESTS WITH BALANCED SETS
###

classifiers = {
    "LinearSVC": LinearSVC(dual=False),
    "MultinomialNB": MultinomialNB(),
    "MLPClassifier": MLPClassifier(verbose = True)
}

### UNDERSAMPLING

filename = RESULTS_FOLDER + "/reports/default/" + "classification_reports-under.csv"

rus = RandomUnderSampler()
XTrain_resampled, yTrain_resampled = rus.fit_sample(training_data, yTrain)
XTrain_resampled = XTrain_resampled["lyrics"].values.astype("U")

for class_key, classifier in classifiers.items():
    for vect_key, vectorizer in vectorizers.items():
        print("\n### %s with %s - Undersampling ###\n" % (class_key, vect_key))

        # Build the pipeline, train the model and predict
        clf = make_pipeline(
            vectorizer,
            StandardScaler(with_mean=False),
            classifier
        )
        clf.fit(XTrain_resampled, yTrain_resampled)
        predictions = clf.predict(XTest)

        # Reports
        report = classification_report(yTest, predictions, target_names = labels, output_dict = True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(filename, header=True, index=True, sep='\t', mode='a')

        plot_confusion_matrix(clf, XTest, yTest)  
        plt.savefig(
            RESULTS_FOLDER + "/figures/default/" + class_key.lower() + "-" + vect_key.lower() + "-under-conf_matrix.png"
        )

"""
filename = RESULTS_FOLDER + "/reports/best-params/" + "classification_reports-under.csv"

print("### FINDING BEST PARAMETERS FOR UNDERSAMPLED DATASET... ###")

classifiers = {
    "MultinomialNB": ComplementNB(),
    "MLPClassifier": MLPClassifier(verbose = True)
}

for class_key, classifier in classifiers.items():
    if (class_key == "MultinomialNB"):
        parameter_space = MultinomialNB_parameter_space
    elif (class_key == "MLPClassifier"):
        parameter_space = MLP_parameter_space

    for vect_key, vectorizer in vectorizers.items():
        print("\n### %s with %s - Undersampling ###\n" % (class_key, vect_key))
        pipeline = make_pipeline(
            vectorizer,
            StandardScaler(with_mean=False),
            classifier
        )
        clf = GridSearchCV(pipeline, parameter_space, n_jobs=-1)
        clf.fit(XTrain, yTrain)

        print("Best parameters found:\n", clf.best_params_)

        predictions = clf.predict(XTest)

        report = classification_report(yTest, predictions, target_names = labels, output_dict = True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(filename, header=True, index=True, sep='\t', mode='a')

        plot_confusion_matrix(clf, XTest, yTest)  
        plt.savefig(
            RESULTS_FOLDER + "/figures/best-params/" + class_key.lower() + "-" + vect_key.lower() + "under-conf_matrix.png"
        )
"""

### OVERSAMPLING

filename = RESULTS_FOLDER + "/reports/default/" + "classification_reports-over.csv"

classifiers = {
    "LinearSVC": LinearSVC(dual=False),
    "MultinomialNB": MultinomialNB(),
    "MLPClassifier": MLPClassifier(verbose = True)
}
   
ros = RandomOverSampler()
XTrain_resampled, yTrain_resampled = ros.fit_sample(training_data, yTrain)
XTrain_resampled = XTrain_resampled["lyrics"].values.astype("U")

for class_key, classifier in classifiers.items():
    for vect_key, vectorizer in vectorizers.items():
        print("\n### %s with %s - Oversampling ###\n" % (class_key, vect_key))

        # Build the pipeline, train the model and predict
        clf = make_pipeline(
            vectorizer,
            StandardScaler(with_mean=False),
            classifier
        )
        clf.fit(XTrain_resampled, yTrain_resampled)
        predictions = clf.predict(XTest)

        # Reports
        report = classification_report(yTest, predictions, target_names = labels, output_dict = True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(filename, header=True, index=True, sep='\t', mode='a')

        plot_confusion_matrix(clf, XTest, yTest)  
        plt.savefig(
            RESULTS_FOLDER + "/figures/default/" + class_key.lower() + "-" + vect_key.lower() + "-over-conf_matrix.png"
        )

"""
filename = RESULTS_FOLDER + "/reports/best-params/" + "classification_reports-over.csv"

print("### FINDING BEST PARAMETERS FOR OVERSAMPLED DATASET... ###")

classifiers = {
    "MultinomialNB": ComplementNB(),
    "MLPClassifier": MLPClassifier(verbose = True)
}

for class_key, classifier in classifiers.items():
    if (class_key == "MultinomialNB"):
        parameter_space = MultinomialNB_parameter_space
    elif (class_key == "MLPClassifier"):
        parameter_space = MLP_parameter_space

    for vect_key, vectorizer in vectorizers.items():
        print("\n### %s with %s - Undersampling ###\n" % (class_key, vect_key))
        pipeline = make_pipeline(
            vectorizer,
            StandardScaler(with_mean=False),
            classifier
        )
        clf = GridSearchCV(pipeline, parameter_space, n_jobs=-1)
        clf.fit(XTrain, yTrain)

        print("Best parameters found:\n", clf.best_params_)

        predictions = clf.predict(XTest)

        report = classification_report(yTest, predictions, target_names = labels, output_dict = True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(filename, header=True, index=True, sep='\t', mode='a')

        plot_confusion_matrix(clf, XTest, yTest)  
        plt.savefig(
            RESULTS_FOLDER + "/figures/best-params/" + class_key.lower() + "-" + vect_key.lower() + "over-conf_matrix.png"
        )
"""