import os
from dotenv import load_dotenv

# Loading environment variables
load_dotenv()
API_KEY = os.environ.get("LAST_FM_KEY")
API_SECRET = os.environ.get("LAST_FM_SECRET")
DATA_FOLDER = os.environ.get("DATA_FOLDER")

import pylast
import csv

network = pylast.LastFMNetwork(
    api_key=API_KEY,
    api_secret=API_SECRET
)

unique_rows = 0

with open(DATA_FOLDER + "/Clean_data/" + "lyrics-data.csv", "r") as lyrics_data, \
    open(DATA_FOLDER + "/Raw_data/" + "artists-data.csv", "r") as artists_data, \
    open(DATA_FOLDER + "/Clean_data/" + "tags-data.csv", "w") as tags_data:
    writer = csv.writer(tags_data, delimiter=',')
    for lyrics_row in csv.reader(lyrics_data, delimiter=','):
        artist_name = None
        for artist_row in csv.reader(artists_data, delimiter=','):
            if artist_row[3] == lyrics_row[0]:
                artist_name = artist_row[0]
                break
        artists_data.seek(0)
        try:
            track = network.get_track(artist_name, lyrics_row[1])
        except pylast.WSError:
            pass
        if track is not None:
            # Artist name
            # Song title
            # Lyrics
            # Top tags
            try:
                raw_tags = track.get_top_tags()
            except pylast.WSError:
                pass
            tags = ""
            if raw_tags is not None:
                for tag in raw_tags:
                    tags += tag.item.get_name() + ", "
            writer.writerow([artist_name, lyrics_row[1], lyrics_row[3], tags])
            unique_rows += 1

print("Number of rows: %d" % unique_rows)