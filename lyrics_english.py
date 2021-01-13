import os
from dotenv import load_dotenv

# Loading environment variables
load_dotenv()
DATA_FOLDER = os.environ.get("DATA_FOLDER")

import csv

unique_rows = 0

# Here we simply filter out lyrics that are not in the English language
with open(DATA_FOLDER + "/Raw_data/" + "lyrics-data.csv", "r") as fin, open (DATA_FOLDER + "/Clean_data/" + "lyrics-data.csv", "w") as fout:
    writer = csv.writer(fout, delimiter=',')
    for row in csv.reader(fin, delimiter=','):
        if row[4] == "ENGLISH":
            writer.writerow(row)
            unique_rows += 1

print("Number of rows: %d" % unique_rows)