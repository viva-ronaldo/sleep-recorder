## Sleep audio recorder and classifier

Collect audio recordings from a Raspberry Pi, and train a model to classify clips. 

A continuous audio monitor runs on the RPi using a single **sox** command and saves .wav files for any sound picked up above a background. `bash pull_recent_clips.sh {yyyymmdd}` retrieves clips from a specified date and adds them to a sqlite database.

`streamlit run listen_and_label_app.py` launches a **streamlit** app from which the user can play clips _and_ assign label to each, which are added to the database.

`python train.py` uses all the labelled clips in the database to train and save a classifier model. This model is then used in the streamlit app to speed up future labelling, to yield a better model with more training data.
