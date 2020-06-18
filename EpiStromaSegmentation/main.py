"""
main.py

This file is where the project's execution begins, and
it contains all of the high level functions for dataset
creation, training, and testing are called in this script.

"""

from KS_lib.prepare_data.routine import split_data
from KS_lib.prepare_data.routine import gen_train_val_data
import flags_epi_stroma_segmentation as flags
from KS_lib.tf_model_epi_stroma_segmentation import tf_model_main
from KS_lib.prepare_data import routine
import os

#split and then generate training and validation sets
split_data(flags.flags)
gen_train_val_data(nth_fold=1,flags=flags.flags)
#routine.write_train_log(flags.flags) #Writes name of all training set patches into log file (used in second iteration of training)


#run model on directory of WSIs
testDir = os.path.join('/data', 'avellal14', 'Adithya_BBD_NHS', 'BBD_NCC_10xExtraction_40x_Part1')
tf_model_main.main(1, 'test_WSI', flags.flags, testDir) #Can be run in either 'train' mode or a variety of different test modes (test, test WSI, test patient, etc;)








