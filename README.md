# clutils
Continual Learning utilities in PyTorch.

Feel free to take a look around if you want to check CL strategies implementations or other stuff.  
But beware :grin: this repo is highly tailored to my research activity and workflow. Hence, it is not meant for general use as CL framework. If you find it useful in some way, then all the better!  
Reach out at any time for any discussion / clarifications :smiley:

## STRUCTURE

`audio`: preprocessing and managing audio signals  
`datasets`: loading datasets adapted for CL  
`experiments`: managing experiments and training  
`extras`: additional stuff  
`metrics`: popular performance metrics  
`models`: implementation of models (MLP, RNNs...)  
`monitors`: monitoring main metrics and experiment logs  
`strategies`: CL strategies  
`video`: preprocessing and managing video

## TO INSTALL
`pip install [-e] git:github.com/AndreaCossu/clutils`

Use `-e` to avoid reinstall in case of code improvements.

Otherwise, clone the repository and add the folder `/path/to/clutils/repo` to `PYTHONPATH`.
