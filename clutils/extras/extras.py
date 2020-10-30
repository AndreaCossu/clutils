import argparse
import pandas as pd
import os
import yaml
import re
from types import SimpleNamespace

def set_gpus(num_gpus):
    try:
        import gpustat
    except ImportError:
        print("gpustat module is not installed. No GPU allocated.")

    try:
        selected = []

        stats = gpustat.GPUStatCollection.new_query()

        for i in range(num_gpus):

            ids_mem = [res for res in map(lambda gpu: (int(gpu.entry['index']),
                                          float(gpu.entry['memory.used']) /\
                                          float(gpu.entry['memory.total'])),
                                      stats) if str(res[0]) not in selected]

            if len(ids_mem) == 0:
                # No more gpus available
                break

            best = min(ids_mem, key=lambda x: x[1])
            bestGPU, bestMem = best[0], best[1]
            # print(f"{i}-th best GPU is {bestGPU} with mem {bestMem}")
            selected.append(str(bestGPU))

        print("Setting GPUs to: {}".format(",".join(selected)))
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(selected)
    except BaseException as e:
        print("GPU not available: " + str(e))


def parse_config(config_file):
    # fix to enable scientific notation
    # https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(config_file, 'r') as f:
        configs = yaml.load(f, Loader=loader)
    configs['config_file'] = config_file
    args = SimpleNamespace()
    for k,v in configs.items():
        args.__dict__[k] = v

    return args


def basic_argparse(parser=None, onemodel=True):

    if parser is None:
        parser = argparse.ArgumentParser()

    # TRAINING
    parser.add_argument('--epochs', type=int, help='epochs to train.')
    if onemodel:
        parser.add_argument('--models', type=str, help='modelname to train')
    else:
        parser.add_argument('--models', nargs='+', type=str, help='modelname to train')
    parser.add_argument('--result_folder', type=str, help='folder in which to save experiment results. Created if not exists.')
    parser.add_argument('--dataroot', type=str, default='/data/cossu', help='folder in which datasets are stored.')

    parser.add_argument('--config_file', type=str, default='', help='path to config file from which to parse args')

    # TASK PARAMETERS
    parser.add_argument('--n_tasks', type=int, default=5, help='Task to train.')
    parser.add_argument('--output_size', type=int, default=10, help='model output size')
    parser.add_argument('--input_size', type=int, default=1, help='model input size')
    parser.add_argument('--max_label_value', type=int, default=10, help='Max value for label.')

    # OPTIMIZER
    parser.add_argument('--weight_decay', type=float, default=0, help='optimizer hyperparameter')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='optimizer hyperparameter')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='Value to clip gradient norm.')

    # EXTRAS
    parser.add_argument('--multitask', action="store_true", help='Multitask learning, all tasks at once.')
    parser.add_argument('--multihead', action="store_true", help='Use task id information at training and test time.')

    parser.add_argument('--test_on_val', action="store_true", help='Test using validation set.')
    parser.add_argument('--not_test', action="store_true", help='disable final test')
    parser.add_argument('--not_intermediate_test', action="store_true", help='disable final test')
    parser.add_argument('--monitor', action="store_true", help='Monitor with tensorboard.')
    parser.add_argument('--save', action="store_true", help='save models')
    parser.add_argument('--load', action="store_true", help='load models')
    parser.add_argument('--cuda', action="store_true", help='use gpu')

    return parser


def add_model_parser(modelnames=['rnn', 'lstm', 'lmn', 'mlp', 'lwta', 'esn', 'cnn'], parser=None):

    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--expand_output', type=int, default=0, help='Expand output layer dynamically.')

    if 'rnn' or 'lstm' in modelnames:
        parser.add_argument('--hidden_size_rnn', type=int, default=128, help='units of RNN')
        parser.add_argument('--layers_rnn', type=int, default=1, help='layers of RNN')

    if 'lmn' in modelnames:
        parser.add_argument('--hidden_size_lmn', type=int, default=128, help='hidden dimension of functional component of LMN')
        parser.add_argument('--memory_size_lmn', type=int, default=128, help='memory size of LMN')
        parser.add_argument('--functional_out', action="store_true", help='compute output from functional instead of memory')
    if 'rnn' or 'lstm' or 'lmn' in modelnames:
        parser.add_argument('--orthogonal', action="store_true", help='Use orthogonal recurrent matrixes')

    if 'mlp' in modelnames:
        parser.add_argument('--hidden_sizes_mlp', nargs='+', type=int, default=[128], help='layers of MLP')
        parser.add_argument('--relu_mlp', action="store_true", help='use relu instead of tanh for MLP')

    if 'lwta' in modelnames:
        parser.add_argument('--units_per_block', nargs='+', type=int, default=[3], help='number of units per block for each hidden layer')
        parser.add_argument('--blocks_per_layer', nargs='+', type=int, default=[10], help='number of blocks for each hidden layer')
        parser.add_argument('--activation_lwta', type=str, choices=['relu', 'tanh', 'none'], default='tanh', help='use `relu`, `tanh` or `none` (no activation) for LWTA')

    if 'esn' in modelnames:
        parser.add_argument('--reservoir_size', type=int, default=128, help='number of neurons in reservoir')
        parser.add_argument('--spectral_radius', type=float, default=0.9, help='spectral radius of reservoir')
        parser.add_argument('--sparsity', type=float, default=0.0, help='percentage of dead connections in reservoir')
        parser.add_argument('--alpha', type=float, default=1.0, help='leakage factor')
        parser.add_argument('--orthogonal_esn', action="store_true", help='Using orthogonal reservoir')

    if 'cnn' in modelnames:
        parser.add_argument('--feed_conv_layers', nargs='+', type=int, default=[256, 128], help='feedforward layers of CNN')
        parser.add_argument('--n_conv_layers', type=int, default=3, help='number of convolutional layers')

    return parser

def add_cl_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # MNIST
    parser.add_argument('--split', action="store_true", help='Use split MNIST.')
    parser.add_argument('--sequential', action="store_true", help='Do not permute MNIST.')
    
    # SPEECH WORDS
    parser.add_argument('--classes_per_task', type=int, default=2, help='How many classes per tasks (total = 30)')
    
    # EWC / MAS
    parser.add_argument('--ewc_lambda', type=float, default=0., help='Use EWC.')
    parser.add_argument('--mas_lambda', type=float, default=0., help='Use MAS.')
    parser.add_argument('--truncated_time', type=int, default=0, help='Truncated time when computing importance gradient in EWC or MAS')

    # CWR* / AR1*
    parser.add_argument('--cwr', action="store_true", help='Use CWR*.')
    parser.add_argument('--ar1', action="store_true", help='Use AR1*.')

    # LwF
    parser.add_argument('--lwf', action="store_true", help='Use LWF.')
    parser.add_argument('--lwf_temp', type=float, default=1, help='Set LWF softmax temperature.')
    parser.add_argument('--warmup_epochs', default=0, type=int, help='how many warmup epochs for lwf')

    # SI
    parser.add_argument('--si_lambda', type=float, default=0., help='Use SI.')
    parser.add_argument('--eps', type=float, default=1e-3, help='SI epsilon.')

    # REHEARSAL
    parser.add_argument('--rehe_patterns', type=int, default=0, help='Number of rehearsal patterns per class.')

    return parser
