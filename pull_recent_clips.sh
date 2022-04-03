#!/bin/bash

if [ $# -eq 0 ]
  then
    echo "No date supplied"
    exit 35
fi

date_string=$1
echo 'Getting clips from' $date_string

#Find the pi as it moves around; use head -1 in case it appears at two addresses (have seen this happen but don't know why)
ipaddr=$(nmap -sn 192.168.8.*/24 | grep raspberry | head -1 | sed -E 's|.*\((.*)\).*|\1|g')
if [[ $ipaddr == '' ]]; then
    echo 'rpi not found - exiting'
    exit 45
fi
echo 'rpi is at' $ipaddr

#TODO check for dates < today that we don't have clips for yet

if [[ ! -d data/$date_string ]]; then
  mkdir data/$date_string
fi

scp pi@$ipaddr:./audio_recordings/$date_string/*.wav ./data/$date_string/

#sox $date_string/*wav summ_$date_string.wav
#export duration=$(soxi -D summ_$date_string.wav)
#echo 'Got '$duration' seconds from '$date_string
#play summ_$date_string.wav

echo 'Updating database'
python add_to_database.py
