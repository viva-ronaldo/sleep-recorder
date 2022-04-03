import os, subprocess, time, glob
import sqlite3
from sqlite3 import Error
import pandas as pd
from functions import read_all_clips_pd

db_file_path = './audio_labels.db'

conn = sqlite3.connect(db_file_path)

dates_with_recordings = [p.split('/')[1] for p in glob.glob('data/2022????')]

all_clips = read_all_clips_pd(dates_with_recordings, dir_prefix='data/')

clips_in_table = pd.read_sql('SELECT * FROM staging', conn)
print(f'{len(clips_in_table)} clips in table')

data_for_table = all_clips[~all_clips.filename.isin(clips_in_table.filename)]
print(f'Adding {len(data_for_table)} new clips')

#TODO calculate paa features now

data_for_table.to_sql('staging', conn, if_exists='append', index=False)

#filename, date, duration_s, timestamp, label
#c = conn.cursor()
#c.execute('DROP TABLE prod;')
#c.execute('CREATE TABLE prod AS SELECT * FROM staging;')
conn.close()
