import pandas as pd, numpy as np
import sqlite3
import joblib

from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
from scipy.integrate import simps  #for rough power spectrum calcs
from pyAudioAnalysis import ShortTermFeatures

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from yellowbrick.model_selection import CVScores
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, ClassPredictionError
import matplotlib.pyplot as plt

from functions import *

# Load data

db_file_path = '/home/david/projects/audio_monitor/audio_labels.db'
data_dir = 'data'

conn = sqlite3.connect(db_file_path)

data = pd.read_sql("SELECT * FROM staging WHERE label!='UNLABELLED' AND duration_s >= 3", conn)
data['time_hour'] = pd.to_datetime(data.timestamp).dt.hour
data['time_hour_night'] = data.time_hour <= 6

print(f'\n{len(data)} labelled samples to train with\n')

# Make features

simple_time_features = ['max_abs_wav','mean_abs_wav','stdev_wav','max_over_mean_wav',
                              'abs_first_last_1s','abs_first_last_1s_over_middle']
print('Calculating simple time features')
data[simple_time_features] = data.filename.apply(get_simple_audio_features, args=(data_dir,))

my_specpower_features = ['specpowerfrac_0_100','specpowerfrac_100_400',
                         'specpowerfrac_400_1000','specpowerfrac_1000_3000',
                         'specpowerfrac_3000_8000']
print('Calculating spectral power features')
data[my_specpower_features] = data.filename.apply(get_crude_spectral_power_fractions, args=(data_dir,))

my_specpower_features_first_1s = [x+'_first_1s' for x in my_specpower_features]
print('Calculating first 1s spectral power features')
data[my_specpower_features_first_1s] = data.filename.apply(get_crude_spectral_power_fractions, args=(data_dir,), first_1s=True)

print('Calculating PyAudioAnalysis features')
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
#Takes 2 minutes for 1000 files if frame_size=0.5
data[paa_features] = data.filename.apply(get_agg_pyaudioanalysis_features, args=(data_dir,),
                                         frame_size=0.5, frame_step=0.25)

# Fit model with testing to measure performance 

#spectral_features = sc_features + sr_features + my_specpower_features
features = ['duration_s','time_hour','time_hour_night'] + \
    simple_time_features + \
    my_specpower_features + my_specpower_features_first_1s + \
    paa_features
#+ zcr_features + rms_features + spectral_features

X, y = data[features], data['label'].copy()

#Combine small classes
y[y.isin(['other','rain','other talking'])] = 'other'
#y[y.isin(['me awake talking','me sleep talking'])] = 'me talking'
#y[y.isin(['watching tv'])] = 'tv/music'
print('Class distribution')
print(y.value_counts())

# Cross val to measure accuracy

cv = StratifiedKFold(n_splits=10, shuffle=True)

# Instantiate the classification model and visualizer
model = RandomForestClassifier(max_depth=8, n_estimators=100, min_samples_split=3, class_weight='balanced')
visualizer = CVScores(model, cv=cv, scoring='accuracy')

visualizer.fit(X, y)        # Fit the data to the visualizer
print(f'Fitted random forest with 10-fold CV; accuracy = {visualizer.cv_scores_mean_:.2f} +/- {np.std(visualizer.cv_scores_):.2f}')
#ax = plt.gca()
plt.title(f'Cross-validation accuracy for random forest model trained on {len(y)} samples with {len(features)} features')
plt.xlabel('Training fold number')
plt.ylabel('Accuracy')
plt.savefig(f'models/clip_classifier_{len(np.unique(y))}classes_latest_crossvalaccuracies.png')
#visualizer.show(outpath=f'clip_classifier_{len(clf.classes_)}classes_latest_crossvalaccuracies.png', clear_figure=True)


# One train test split to get confusion matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=331)

clf = RandomForestClassifier(n_estimators=200, max_depth=8, 
                             min_samples_split=3, class_weight='balanced')
clf.fit(X_train, y_train)

#** Change this to me sleep talking when there are enough cases in y_test
#prec,rec,_ = precision_recall_curve(np.where(y_test=='me sleep talking', 1, 0), 
#    clf.predict_proba(X_test)[:,list(clf.classes_).index('me sleep talking')])
prec,rec,_ = precision_recall_curve(np.where(y_test.isin(['me awake talking','me sleep talking']), 1, 0), 
    clf.predict_proba(X_test)[:,list(clf.classes_).index('me awake talking')] + clf.predict_proba(X_test)[:,list(clf.classes_).index('me sleep talking')])


fig = plt.figure(figsize=(8,7))
visualizer = ConfusionMatrix(clf, is_fitted=True)
acc = visualizer.score(X_test, y_test)
ax = plt.gca()
plt.ylabel('True label')
plt.xlabel('Predicted label')
ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=30, size=10)
ax.set_yticklabels(labels=ax.get_yticklabels(), size=10)
plt.title(f'Random forest model trained on {len(y_train)} samples with {len(features)} features\n' + 
    f'Accuracy = {acc:.2f}; precision for 90% recall on me talking = {prec[np.argmin(rec > 0.9)]:.2f}')
plt.tight_layout()
plt.savefig(f'models/clip_classifier_{len(clf.classes_)}classes_latest_confmatrix.png')
#visualizer.show()
print(f'Accuracy = {acc:.2f}')

# Fit on all X,y and save

clf = RandomForestClassifier(n_estimators=200, max_depth=8, 
                             min_samples_split=3, class_weight='balanced')
clf.fit(X, y)

joblib.dump(clf, f'models/clip_classifier_{len(clf.classes_)}classes_latest.pkl')

#joblib_model = joblib.load(joblib_file)
