import os
from dotenv import load_dotenv

# Loading environment variables
load_dotenv()
DATA_FOLDER = os.environ.get("DATA_FOLDER")

import csv

nb_workable_rows = 0
# Remove rows without tags
with open(DATA_FOLDER + "/Clean_data/" + "tags-data.csv", "r") as tags_data, \
    open(DATA_FOLDER + "/Clean_data/" + "tags-data-noempty.csv", "w") as tags_data_clean:
    writer = csv.writer(tags_data_clean, delimiter=',')
    for tag_row in csv.reader(tags_data, delimiter=','):
        if tag_row[3] != "":
            writer.writerow(tag_row)
            nb_workable_rows += 1

# Filter out tags that are not relevant
with open(DATA_FOLDER + "/Clean_data/" + "tags-data-noempty.csv", "r") as tags_data, \
    open(DATA_FOLDER + "/Clean_data/" + "tags-data-mood.csv", "w") as tags_data_processed:
    writer = csv.writer(tags_data_processed, delimiter=',')
    emotion_list = [
        "happy", "happiness", "joyous", "bright", "cheerful", "humorous", "fun", "merry", "exciting", "silly",
        "angry", "aggressive", "outrageous", "fierce", "anxious", "rebellious", "tense", "fiery", "hostile", "anger",
        "sad", "bittersweet", "bitter", "tragic", "depressing", "sadness", "gloomy", "miserable", "funeral", "sorrow",
        "relaxed", "tender", "soothing", "peaceful", "gentle", "soft", "quiet", "calm", "mellow", "delicate"
    ]
    nb_workable_rows = 0
    for tag_row in csv.reader(tags_data, delimiter=','):
        tags = tag_row[3].split(", ")
        clean_tags = []
        for tag in tags:
            # Some tags are actually short sentences
            # We'll make the assumption that instances of mood-describing words in tags can be taken individually
            tag = tag.lower()
            sub_tags = tag.split(' ')
            for sub_tag in sub_tags:
                if sub_tag in emotion_list and sub_tag not in clean_tags:
                    clean_tags.append(sub_tag)
        if len(clean_tags) > 0:
            separator = ", "
            writer.writerow([tag_row[0], tag_row[1], tag_row[2], separator.join(clean_tags)])
            nb_workable_rows += 1

# Attribute gold labels to the lyrics
with open(DATA_FOLDER + "/Clean_data/" + "tags-data-mood.csv", "r") as tags_data, \
    open(DATA_FOLDER + "/Clean_data/" + "lyrics-gold-labels.csv", "w") as fout:
    writer = csv.writer(fout, delimiter=',')
    Q1 = [
        "happy", "happiness", "joyous", "bright", "cheerful", "humorous", "fun",
        "merry", "exciting", "silly"
    ]
    Q2 = [
        "angry", "aggressive", "outrageous", "fierce", "anxious", "rebellious",
        "tense", "fiery", "hostile", "anger"
    ]
    Q3 = [
        "sad", "bittersweet", "bitter", "tragic", "depressing", "sadness", "gloomy",
        "miserable", "funeral", "sorrow"
    ]
    Q4 = [
        "relaxed", "tender", "soothing", "peaceful", "gentle", "soft", "quiet", "calm",
        "mellow", "delicate"
    ]
    nb_workable_rows = 0
    for tag_row in csv.reader(tags_data, delimiter=','):
        nb_Q1 = 0
        nb_Q2 = 0
        nb_Q3 = 0
        nb_Q4 = 0
        tags = tag_row[3].split(", ")
        for tag in tags:
            if tag in Q1:
                nb_Q1 += 1
            elif tag in Q2:
                nb_Q2 += 1
            elif tag in Q3:
                nb_Q3 += 1
            else:
                nb_Q4 += 1

        if nb_Q1 > max(nb_Q2, nb_Q3, nb_Q4):
            writer.writerow([tag_row[0], tag_row[1], tag_row[2], "happy"])
            nb_workable_rows +=1
        elif nb_Q2 > max(nb_Q1, nb_Q3, nb_Q4):
            writer.writerow([tag_row[0], tag_row[1], tag_row[2], "angry"])
            nb_workable_rows +=1
        elif nb_Q3 > max(nb_Q1, nb_Q2, nb_Q4):
            writer.writerow([tag_row[0], tag_row[1], tag_row[2], "sad"])
            nb_workable_rows +=1
        elif nb_Q4 > max(nb_Q1, nb_Q2, nb_Q3):
            writer.writerow([tag_row[0], tag_row[1], tag_row[2], "relaxed"])
            nb_workable_rows +=1

print("Number of workable rows: %d" % nb_workable_rows)