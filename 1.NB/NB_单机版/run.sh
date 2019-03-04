python DataConvert.py data/ nb_data

# train
python NB.py 1 nb_data.train model

# test
python NB.py 0 nb_data.test model out
