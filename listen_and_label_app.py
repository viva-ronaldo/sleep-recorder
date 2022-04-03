import sys, glob, subprocess, os, time
import pandas as pd, numpy as np
from scipy.io import wavfile
from scipy.signal import resample, hann
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt, seaborn as sns
import streamlit as st
import sqlite3, joblib

from functions import *

data_dir = 'data'

conn = sqlite3.connect('audio_labels.db', timeout=5)

classes = ['background','me sleep talking','me awake talking','cough/sneeze/laugh/groan',
    'other talking','window/door','vehicle','piano','watching tv',
    'rain',
    'other',
    'UNLABELLED']

dates_with_recordings = [p.split('/')[1] for p in glob.glob('data/2022????')]

model = joblib.load('models/clip_classifier_9classes_latest.pkl')

st.write(f"""
    # Audio clip summary and labeller
    ## Summary
    """)
                
#if 'labelling_radio' in st.session_state:
#    print('can write to db now with value',st.session_state['labelling_radio'],'for')#,clip_to_play.filename)
    

def read_all_clips(conn):
    tmp = pd.read_sql('SELECT * FROM staging WHERE duration_s >= 3;', conn)
    tmp['timestamp'] = pd.to_datetime(tmp.timestamp)
    tmp['time_hour'] = tmp.timestamp.dt.hour
    tmp['time_hour_night'] = tmp.time_hour <= 6
    return tmp

def write_choice_to_db():
    new_label = st.session_state['labelling_radio']
    c = conn.cursor()
    c.execute(f"UPDATE staging SET label='{new_label}' WHERE filename='{clip_to_play.filename}';")
    conn.commit()
    c.close()    
    print(f'wrote value {new_label} for {clip_to_play.filename}')


print('reading in data again')
clips = read_all_clips(conn)
#print('done read; n rows =',len(clips))
#print(clips.label.value_counts())
st.write(f'Found {len(clips)} clips in total, from {clips.date.min()} to {clips.date.max()}')

fig, axs = plt.subplots(2, 2, figsize=(10,10))
axs[0][0].hist(pd.to_datetime(clips.date.tolist()), color='forestgreen')
axs[0][0].set_ylabel('Number of clips')
axs[0][0].set_xlabel('Date')
axs[0][0].set_title('Recording frequency over time')
#axs[0][0].xticks(rotation=45)
axs[0][0].tick_params(axis='x', labelrotation=45)
#axs[0][0].set_xticklabels(labels=axs[0][0].get_xticklabels(), rotation=45)
#print(dir(axs[0][0]))
#print(axs[0][0].get_xmajorticklabels())
#print(axs[0][0].get_xticklabels())
#print(axs[0][0].get_xticks())

axs[0][1].hist(clips.timestamp.dt.hour, bins=24, range=(0,24))
axs[0][1].set_ylabel('Count')
axs[0][1].set_xlabel('Hour')
axs[0][1].set_xticks(range(0,24,6))
axs[0][1].set_title('Recording frequency by time of day')

axs[1][0].hist(clips.duration_s, bins=100, color='coral')
axs[1][0].set_ylabel('Count')
axs[1][0].set_xlabel('Clip duration / seconds')
axs[1][0].set_xlim(0,80)
axs[1][0].set_title('')

#st.pyplot(fig)

#And label mix
#fig, ax = plt.subplots(1, 1, figsize=(6,5))
#vcs = clips[clips.label!='UNLABELLED'].label.value_counts()
#print(vcs.head())
#axs[1][1].hist(clips[clips.label!='UNLABELLED'].label, 
#    bins=clips.label.nunique()-2, color='midnightblue',
#    orientation='horizontal', rwidth=0.8, align='left')
#axs[1][1].set_xlabel('Count')
#axs[1][1].set_title('Clips by labelled class')
tmp = clips[clips.label!='UNLABELLED']
sns.countplot(y='label', data=tmp, orient='h',
    order=tmp.label.value_counts().index, ax=axs[1][1])
axs[1][1].set_ylabel(None)
axs[1][1].set_xlabel('Clip count')

plt.tight_layout()
st.pyplot(fig)

"session state = ", st.session_state

st.write("""## Labelling
### Play and label clips""")

col1, col2 = st.columns(2)
with col1:
    clip_filter = st.selectbox('Show clips from date or with label...', sorted(dates_with_recordings) + classes)

    addtl_unlabelled_filter = st.checkbox('Show UNLABELLED only', value=False)

with col2:
    if clip_filter in dates_with_recordings:
        filtered_clips = clips[clips.date==clip_filter]
        if addtl_unlabelled_filter:
            filtered_clips = filtered_clips[filtered_clips.label=='UNLABELLED']
            st.write(f'Found **{len(filtered_clips)}** UNLABELLED clips from {clip_filter}')
        else:
            st.write(f'Found **{len(filtered_clips)}** clips from {clip_filter}')
    else:
        filtered_clips = clips[clips.label==clip_filter]
        st.write(f'Found **{len(filtered_clips)}** clips with label _{clip_filter}_')

if len(filtered_clips) > 0:

    option = st.selectbox('Select file to play', filtered_clips.filename,
        key='file_selectbox')

    #Instead we can make the setting here, on every run through.
    response = pd.read_sql(f'SELECT label from staging WHERE filename="{option}"', conn)
    print(f'The db says its label is {response.label.iloc[0]} so setting radio selector to that')
    st.session_state['labelling_radio'] = response.label.iloc[0]

    #print(f'After option line; option = {option}')
    #response = pd.read_sql(f'SELECT label from staging WHERE filename="{option}"', conn)
    #print(f'The db says its label is {response.label.iloc[0]} so setting radio selector to that')
    #st.session_state['labelling_radio'] = response.label.iloc[0]
    
    clip_to_play = clips[clips.filename==option].copy()

    clip_samplerate, clip_wav_data = wavfile.read('data/'+clip_to_play.filename.iloc[0])

    simple_time_features = ['max_abs_wav','mean_abs_wav','stdev_wav','max_over_mean_wav',
        'abs_first_last_1s','abs_first_last_1s_over_middle']
    clip_to_play[simple_time_features] = get_simple_audio_features(clip_to_play.filename.iloc[0], data_dir)

    my_specpower_features = ['specpowerfrac_0_100','specpowerfrac_100_400',
                             'specpowerfrac_400_1000','specpowerfrac_1000_3000',
                             'specpowerfrac_3000_8000']
    clip_to_play[my_specpower_features] = get_crude_spectral_power_fractions(clip_to_play.filename.iloc[0], data_dir)

    my_specpower_features_first_1s = [x+'_first_1s' for x in my_specpower_features]
    clip_to_play[my_specpower_features_first_1s] = get_crude_spectral_power_fractions(clip_to_play.filename.iloc[0], data_dir, first_1s=True)

    paa_features = ['paa_zcr_mean','paa_zcr_std',
                    'paa_energy_mean','paa_energy_std',
                    'paa_energyentropy_mean','paa_energyentropy_std',
                    'paa_sc_mean','paa_sc_std',
                    'paa_ss_mean','paa_ss_std',
                    'paa_se_mean','paa_se_std',
                    'paa_sf_mean','paa_sf_std',
                    'paa_sr_mean','paa_sr_std',
                    'paa_mcc1_mean','paa_mcc1_std',
                    'paa_mcc2_mean','paa_mcc2_std',
                    'paa_mcc3_mean','paa_mcc3_std',
                    'paa_mcc4_mean','paa_mcc4_std',
                    'paa_mcc5_mean','paa_mcc5_std',
                    'paa_mcc6_mean','paa_mcc6_std',
                    'paa_mcc7_mean','paa_mcc7_std',
                    'paa_mcc8_mean','paa_mcc8_std',
                    'paa_mcc9_mean','paa_mcc9_std',
                    'paa_mcc10_mean','paa_mcc10_std',
                    'paa_mcc11_mean','paa_mcc11_std',
                    'paa_mcc12_mean','paa_mcc12_std',
                    'paa_mcc13_mean','paa_mcc13_std']
    clip_to_play[paa_features] = get_agg_pyaudioanalysis_features(clip_to_play.filename.iloc[0], 
            data_dir, frame_size=0.5, frame_step=0.25)
    
    pred_probs = model.predict_proba(clip_to_play[model.feature_names_in_])[0]
    pred_class = model.classes_[np.argmax(pred_probs)]

    got_new_label_from_model = False
    if st.session_state['labelling_radio'] == 'UNLABELLED' and max(pred_probs) > 0.5:
        if pred_class in classes:
            st.session_state['labelling_radio'] = pred_class
            got_new_label_from_model = True
        else:
            print(f'Predicted class is {pred_class}, not in list - conversion needed')

    clip_to_play = clip_to_play.iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.audio('data/'+clip_to_play.filename)
        st.write(f"Recorded at {clip_to_play.timestamp.strftime('%H:%M')}")
        st.write(f'Duration = {clip_to_play.duration_s} s')
        st.write(f'Max abs(data) = {max(abs(clip_wav_data)):.0f}')
        st.write(f'Mean abs(data) = {np.mean(abs(clip_wav_data)):.0f}')
        #TODO 'front loaded' ratio indicator
        #TODO other things that might be insightful as to the label
    with col2:
        time_pts = np.linspace(0., clip_to_play.duration_s, len(clip_wav_data))
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(time_pts, clip_wav_data / 2**15, label="Single channel")
        plt.legend()
        plt.xlabel("Time / s")
        plt.ylabel("Normalised amplitdude")
        st.pyplot(fig=fig)
        #st.line_chart(data=resample(clip_wav_data,1000))

        #optionally do *= hann(len(clip_wav_data)) first
        magnitude = 20*np.log10(abs(rfft(clip_wav_data)))
        magnitude -= max(magnitude)
        xf = rfftfreq(len(clip_wav_data), 1/clip_samplerate)
        fig2 = plt.figure(figsize=(6,4))
        #plt.plot(xf, magnitude, 'k')
        w = np.hanning(5001)
        yf_sm = np.convolve(w/w.sum(), magnitude, mode='valid')
        plt.plot(xf[2500:-2500], yf_sm)
        plt.ylabel('dB ?')
        plt.xlabel('Frequency / Hz')
        plt.title('Uneme vowel spectrum')
        st.pyplot(fig=fig2)

    col1, col2 = st.columns(2)
    with col1:
        clip_type_choice = st.radio('This is...', 
            classes, index=classes.index(clip_to_play.label),
            key='labelling_radio', on_change=write_choice_to_db)

        #index only goes to this requested value the first time, then it stays where clicked
        #clip_type_choice = st.selectbox('This is...',
        #    classes, index=classes.index(clip_to_play.label))
        #print('past radio line')
        #st.session_state['clip_type_choice'] = clip_type_choice
        #print('added choice to ss')

        st.write(f"-> set file label to _{clip_type_choice}_")

    with col2:
        st.write(f'Predicted class is **{pred_class}** ({max(pred_probs):.2f})')
        if len(list(filter(lambda p: p >= 0.25, pred_probs))) > 1:
            st.write('Or could be:')
            for c,p in zip(model.classes_, pred_probs):
                if p >= 0.25 and c != pred_class:
                    st.write(f'**{c}** ({p:.2f})')

        if got_new_label_from_model:
            #st.write(f'The model thinks the label is {pred_class} (p = {max(pred_probs):.2f}) so set the radio selector to that')
            if st.button('Accept label and write to db'):
                write_choice_to_db()

    st.write('### Remaining clips in category')

    st.table(filtered_clips[['filename','duration_s','label']])
    #or st.write(long_clips) gives a sortable table

if st.button('Update PROD table from STAGING'):
    #PROD acts more like a backup for STAGING in case I make some
    #  mistakes during a session and want to undo them

    c = conn.cursor()
    #c.execute('INSERT INTO prod SELECT * FROM staging;')
    c.execute('DROP TABLE prod;')
    c.execute('CREATE TABLE prod AS SELECT * FROM staging;')
    c.close()
    conn.commit()
    st.write('Updated prod table')
    st.write(pd.read_sql('SELECT label, COUNT(*) from prod GROUP BY LABEL;', conn))

    #Later, also kick off model retraining


