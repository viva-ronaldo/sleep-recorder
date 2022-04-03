import subprocess, time, glob, os
import pandas as pd, numpy as np
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
from scipy.integrate import simps  #for rough power spectrum calcs
from pyAudioAnalysis import ShortTermFeatures

def read_all_clips_pd(dates_with_recordings, min_duration=3, dir_prefix='data/'):
    clips = []
    for date_dir in dates_with_recordings:
        this_date_clips = glob.glob(f'{dir_prefix}{date_dir}/*wav')
        #tdc includes dir_prefix, for the os.stat line; remove it when adding to df
        for tdc in this_date_clips:
            response = subprocess.run(['soxi','-D',tdc], stdout=subprocess.PIPE)
            if response.returncode == 0:
                duration = int(float(response.stdout))
                if duration >= min_duration:
                    epoch_time = time.ctime(os.stat(tdc).st_mtime)
                    clips.append(pd.DataFrame({'filename': [tdc.replace(dir_prefix,'')], 
                        'date': [date_dir],
                        'duration_s': [duration],
                        'timestamp': pd.to_datetime([date_dir+'-'+tdc[-10:-4]]),
                        'label': ['UNLABELLED']}))

    return pd.concat(clips).reset_index(drop=True).sort_values('timestamp', ascending=True)

def get_simple_audio_features(clip_filename, data_dir):
    clip_samplerate, clip_wav_data = wavfile.read(data_dir +'/'+ clip_filename)
    
    max_abs = np.max(abs(clip_wav_data))
    
    mean_abs = np.mean(abs(clip_wav_data))
    
    std = np.std(clip_wav_data)
    
    max_over_mean = np.max(abs(clip_wav_data)) / np.mean(abs(clip_wav_data))
    
    abs_first_last_1s = np.mean(np.abs(np.concatenate([clip_wav_data[:clip_samplerate],
                                                clip_wav_data[-clip_samplerate:]])))
    abs_first_last_1s_over_middle = abs_first_last_1s / np.mean(np.abs(clip_wav_data[clip_samplerate:-clip_samplerate]))
    if len(clip_wav_data) < 2*clip_samplerate:
        print('short', len(clip_wav_data), clip_samplerate)
    
    return pd.Series((max_abs, mean_abs, std, max_over_mean, abs_first_last_1s, abs_first_last_1s_over_middle))

def get_crude_spectral_power_fractions(clip_filename, data_dir, first_1s=False):
    sample_rate, wavedata = wavfile.read(data_dir + '/' + clip_filename)
    if first_1s:
        yf = rfft(wavedata[:int(sample_rate)]) 
        xf = rfftfreq(int(sample_rate), 1 / sample_rate)
    else:
        yf = rfft(wavedata) 
        xf = rfftfreq(len(wavedata), 1 / sample_rate)
    
    delta_freq = xf[1]-xf[0]
    total_power = simps(np.abs(yf), dx=delta_freq)
    
    return pd.Series((simps(np.abs(yf)[(xf <= 100)], dx=delta_freq) / total_power,
                      simps(np.abs(yf)[(xf > 100) & (xf <= 400)], dx=delta_freq) / total_power,
                      simps(np.abs(yf)[(xf > 400) & (xf <= 1000)], dx=delta_freq) / total_power,
                      simps(np.abs(yf)[(xf > 1000) & (xf <= 3000)], dx=delta_freq) / total_power,
                      simps(np.abs(yf)[(xf > 3000)], dx=delta_freq) / total_power))

def get_agg_pyaudioanalysis_features(clip_filename, data_dir, frame_size=0.05, frame_step=0.025):
    
    sample_rate, wavedata = wavfile.read(data_dir + '/' + clip_filename)
    wavedata = wavedata.astype(float)
    
    F, f_names = ShortTermFeatures.feature_extraction(
        wavedata, sample_rate,
        frame_size*sample_rate, 
        frame_step*sample_rate,
        deltas=False)
    
    F_means = F.mean(axis=1)
    F_stdevs = F.std(axis=1)
    
    #skip the chroma terms at the end
    zipped_features = [(m,s) for (m,s) in zip(F_means[:21], F_stdevs[:21])]
    return pd.Series([i for j in zipped_features for i in j])
