# AudioDelossifier
Delossify compressed audio (mp3 and others)

Copy your 32-bit floating point *.wav files to the /audio folder.
Run audio_predict.py and find your delossified files in /audio/out

To train a model, copy compressed audio as 32-bit floating point *.wav files to the /training-data/compressed folder.
Also copy the uncompressed versions with the same file names to the /training-data/uncompressed folder.

Then run audio_train.py. The script first tries to perfectly align the samples and caches the resultig files in /aligned-data. This might fail, when the files are stretched for some reason and can't be aligned.

Some settings can be configured in audio_config.py.
