import os
import logging
import torch


def create_result_folder(result_folder, path_save_models='saved_models'):
    '''
    Set plot folder by creating it if it does not exist.
    '''

    os.makedirs(result_folder, exist_ok=True)
    os.mkdir(os.path.join(result_folder, path_save_models))


def get_device(cuda):
    '''
    Choose device: cpu or cuda
    '''

    mode = 'cpu'
    if cuda:
        if torch.cuda.is_available():
            logging.info(f"Using {torch.cuda.device_count()} GPU(s)")
            mode = 'cuda'
        else:
            logging.info("No GPU found. Using CPUs...")
    else:
        logging.info('No GPU will be used')

    device = torch.device(mode)

    return device


def save_model(model, modelname, path_save_models='saved_models', version=''):
    '''
    :param version: specify version of the model. Usually used to represent the model when trained after task 'version'
    '''

    torch.save(model.state_dict(), os.path.join(path_save_models, modelname+version+'.pt'))


def load_models(model, modelname, device, path_save_models, version=''):
    check = torch.load(os.path.join(path_save_models, modelname+version+'.pt'), map_location=device)

    model.load_state_dict(check)

    model.eval()

    return model