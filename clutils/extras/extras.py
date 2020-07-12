import pandas as pd
import os

def distributed_validation(args):
    '''
    Set args for a distributed job taking input from distributed_validation.csv file.
    The file must be put in the project root folder.
    Separate multiple layers parameter with '_' (no spaces). e.g. 10_10 for two layers with 10 units each 
    In case of parameters accepting multiple layers use trailing '_' if only one layer used. e.g. 10_ for 1 layer with 10 units.
    '''

    data_csv = pd.read_csv('distributed_validation.csv')

    params = data_csv.columns.tolist()

    folder_suffix = '_'

    for p in params: # for every column id
        val = str(data_csv[p][args.job_id]) # take a specific row

        folder_suffix += val
        folder_suffix += '_'

        if '_' in val: # format int and split according to layers
            val = val.split('_')
            if val[-1] == '':
                val = val[:-1]

            val = list(map(int, val))
        else:
            val = float(val)

        setattr(args, p, val)
    
    setattr(args, 'result_folder', os.path.join(args.result_folder, folder_suffix[:-1]))

    return args
