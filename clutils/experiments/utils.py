import os
import torch
from ..models import VanillaRNN, LSTM, LMN, MLP, ESN, LWTA, CNN


def create_result_folder(result_folder, path_save_models='saved_models'):
    '''
    Set plot folder by creating it if it does not exist.
    '''

    result_folder = os.path.expanduser(result_folder)
    os.makedirs(os.path.join(result_folder, path_save_models), exist_ok=True)
    return result_folder


def get_device(cuda):
    '''
    Choose device: cpu or cuda
    '''

    mode = 'cpu'
    if cuda:
        if torch.cuda.is_available():
            print(f"Using {torch.cuda.device_count()} GPU(s)")
            mode = 'cuda'
        else:
            print("No GPU found. Using CPUs...")
    else:
        print('No GPU will be used')

    device = torch.device(mode)

    return device


def save_model(model, modelname, base_folder, path_save_models='saved_models', version=''):
    '''
    :param version: specify version of the model. Usually used to represent the model when trained after task 'version'
    '''

    torch.save(model.state_dict(), os.path.join(
        os.path.expanduser(base_folder), 
        path_save_models, modelname+version+'.pt'))


def load_models(model, modelname, device, base_folder, path_save_models='saved_models', version=''):
    check = torch.load(os.path.join(
        os.path.expanduser(base_folder),
        path_save_models, modelname+version+'.pt'), map_location=device)

    model.load_state_dict(check)

    model.eval()

    return model


def create_models(args, device, len_sequence=784, C=1, H=28, W=28, path_save_models='saved_models', version=''):
    '''
    Create models for CL experiment.

    :param version: string representing version of models to load.
    '''

    models = {}

    if 'rnn' in args.models:
        models['rnn'] = VanillaRNN(args.input_size, args.hidden_size_rnn, args.output_size, device,
            num_layers=args.layers_rnn, orthogonal=args.orthogonal)

    if 'lstm' in args.models:
        models['lstm'] = LSTM(args.input_size, args.hidden_size_rnn, args.output_size, device,
            num_layers=args.layers_rnn, orthogonal=args.orthogonal)

    if 'lmn' in args.models:
        models['lmn'] = LMN(args.input_size, args.hidden_size_lmn, args.output_size, args.memory_size_lmn,
            device, orthogonal=args.orthogonal, functional_out=args.functional_out)
    
    if 'esn' in args.models:
        models['esn'] = ESN(args.input_size, args.reservoir_size, args.output_size, device,
            alpha=args.alpha, logistic=args.logistic, spectral_radius=args.spectral_radius, sparsity=args.sparsity, orthogonal=args.orthogonal_esn)

    if 'mlp' in args.models:
        models['mlp'] = MLP(len_sequence*args.input_size, args.hidden_sizes_mlp, device, output_size=args.output_size, relu=args.relu_mlp)
    
    if 'cnn' in args.models:
        models['mlp'] = CNN(C, device, H, W, args.n_conv_layers, args.feed_conv_layers, output_size=args.output_size)

    if 'lwta' in args.models:
        models['lwta'] = LWTA(
            args.units_per_block, args.blocks_per_layer, device, 
            len_sequence*args.input_size, output_size=args.output_size, 
            nonlinear_activation=args.activation_lwta, out_activation=None )

    if args.load:
        for modelname in args.models:
            models[modelname] = load_models(models[modelname], modelname, device, 
            args.result_folder, path_save_models, version=version)
    
    return models


def create_optimizers(models, lr, wd=0.):
    '''
    Associate an optimizer to each model
    '''

    optimizers = {}

    for modelname, model in models.items():
        optimizers[modelname] = torch.optim.Adam(model.parameters(),
            lr=lr, weight_decay=wd)

    return optimizers


def clip_grad(model, max_grad_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


def detach(h):
    if isinstance(h, (tuple, list)):
        return tuple([hh.detach() for hh in h])
    else:
        return h.detach()