import os, subprocess, time, glob
import sqlite3
from sqlite3 import Error
import pandas as pd
from functions import read_all_clips_pd

db_file_path = './audio_labels.db'

try:
    conn = sqlite3.connect(db_file_path)

    dates_with_recordings = glob.glob('data/2022????')

    data_for_table = read_all_clips_pd(dates_with_recordings)

    data_for_table.to_sql('staging', conn, index=False)

    #filename, date, duration_s, timestamp, label
    c = conn.cursor()
    c.execute('CREATE TABLE prod AS SELECT * FROM staging;')

    #check tables from command line: 'sqlite3 audio_labels.db'; '.tables'

except Error as e:
    print(e)
finally:
    conn.close()
