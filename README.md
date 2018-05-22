An In-house Implementation of Prosody Transfer Tacotron
*******************************************************

__AUTHOR__ = "Tong Wang"
__VERSION__ = "0.1"

This project aims at implementing the prosody transfer Tacotron, an text-to-speech deep neural network with multi-gpus and multi-cpus.
Please read these articles for more details.

https://arxiv.org/abs/1803.09047
https://arxiv.org/abs/1703.10135



REQUIREMENTS
************
Numpy = 1.14.2
Scipy = 1.0.1
Matplotlib = 2.0.2

TensorFlow = 1.6.0 (gpu version)



DATA PREPARATION
****************
At the begining stage, we trained the model with LJ Speech Dataset. (https://keithito.com/LJ-Speech-Dataset/)

LJ Speech Dataset is recently widely used as a benchmark dataset in the TTS task because it is publicly available. It has 24 hours of     reasonable quality samples.


Please download and save the file "meta.csv" of LJSppech Dataset at "data/all" and all the audio files with suffix ".wav" in "data/all/wavs"

To generate spectrogrom files, run the following command,
$ python utils/preprocess.py

To split dataset into training and evaluation sets, just run the follwing command,
$ python utils/build_tr_ev_db.py

Then you can see the training set files and evaluation set files in "data/train" and "data/eval", respectively.   



TRAIN and EVALUATION
********************
Adjust hyper parameters in "Hyperparams.py".
Run the following command,
$ python main.py [logdir_idx] [mode] 
Example: python main.py 1 train/eval/synthes



SYNTHESIZE
**********
Speech samples were generated based on the same script as the ones used for the original [web demo](https://google.github.io/tacotron/publications/end_to_end_prosody_transfer/). 
You can check them in "data/synthesize".

To synthesize audios,
Run the following command,
$ python main.py [logdir_idx] synthes
where the logdir_idx is the model dir which you want to use 



TROUBLESHOOTING
***************
Since it is the first version of implementing Tacotron, there are still some extra space to improve the whole performance and accerlate training spped. Also, there might remain few bugs in the project. The improved and more powerful version should be continued.
  
Please contact t-wan14@mails.tsinghua.edu.cn for any problems users may encounter using my in-house project.
