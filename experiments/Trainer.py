import torch
from .utils import load_models
from ..models import VanillaRNN, LSTM, LMN, MLP, ESN, LWTA

class Trainer():
    
    def __init__(self,
        task_type = 'classification', # element2element, seq2seq
        criterion = None,
        eval_metric = None,
        len_sequence = 784,
        max_grad_norm = 5.0,
        after_forward_callbacks = {},
        after_metrics_callbacks = {},
        after_backward_callbacks = {}
    ):


        self.task_type = task_type
        self.models = {}
        self.optimizers = {}
        if criterion is None:
            self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        else:
            self.criterion = criterion
        self.eval_metric = eval_metric

        self.max_grad_norm = max_grad_norm

        self.len_sequence = len_sequence 

        self.after_forward_callbacks = after_forward_callbacks
        self.after_metrics_callbacks = after_metrics_callbacks
        self.after_backward_callbacks = after_backward_callbacks

    def create_models(self, args, device, path_save_models='saved_models', version=''):
        '''
        Create models for CL experiment.

        :param version: string representing version of models to load.
        '''


        if 'rnn' in args.models:
            self.models['rnn'] = VanillaRNN(args.input_size, args.hidden_size_rnn, args.output_size, device,
                num_layers=args.layers_rnn, orthogonal=args.orthogonal)

        if 'lstm' in args.models:
            self.models['lstm'] = LSTM(args.input_size, args.hidden_size_rnn, args.output_size, device,
                num_layers=args.layers_rnn, orthogonal=args.orthogonal)

        if 'lmn' in args.models:
            self.models['lmn'] = LMN(args.input_size, args.hidden_size_lmn, args.output_size, args.memory_size_lmn,
                device, orthogonal=args.orthogonal, functional_out=args.functional_out)
        
        if 'esn' in args.models:
            self.models['esn'] = ESN(args.input_size, args.reservoir_size, args.output_size, device,
                alpha=args.alpha, logistic=args.logistic, spectral_radius=args.spectral_radius, sparsity=args.sparsity, orthogonal=args.orthogonal_esn)

        if 'mlp' in args.models:
            self.models['mlp'] = MLP(self.len_sequence*args.input_size, args.hidden_sizes_mlp, device, output_size=args.output_size, relu=args.relu_mlp)
        
        if 'lwta' in args.models:
            self.models['lwta'] = LWTA(
                args.units_per_block, args.blocks_per_layer, device, 
                self.len_sequence*args.input_size, output_size=args.output_size, 
                nonlinear_activation=args.activation_lwta, out_activation=None )

        if args.load:
            for modelname in args.models:
                self.models[modelname] = load_models(self.models[modelname], modelname, device, path_save_models, version=version)
        
        return self.models


    def create_optimizers(self, models, lr, wd=0.):
        '''
        Associate an optimizer to each model
        '''

        for modelname, model in self.models.items():
            self.optimizers[modelname] = torch.optim.Adam(model.parameters(),
                lr=lr, weight_decay=wd)

        return self.optimizers

    def do_train(self, modelname, x,y, optimize=True, task_id=None, ewc=False):
        '''
        :param optimize: optimize parameters, False to compute gradients only.
        :param task_id: current task id
        :param ewc: add additional penalty based on EWC
        '''
        
        self.models[modelname].train()

        predictions = self.do_forward(modelname, x,y)

        for train, cb in self.after_forward_callbacks:
            if train:
                cb()

        loss = self.compute_loss(modelname, predictions, y, ewc, task_id)
        ev_metric = self.compute_eval_metric(predictions, y)

        if optimize:
            self.optimize(modelname)

        return loss, ev_metric

    def do_validation(self, modelname, x,y):

        with torch.no_grad():
            self.models[modelname].eval()

            predictions = self.do_forward(modelname, x,y)

            loss = self.compute_loss(modelname, predictions, y)
            ev_metric = self.compute_eval_metric(predictions, y)

            return loss, ev_metric

    def do_forward(self, modelname, x,y):

        if modelname == 'mlp' or modelname == 'lwta':
            predictions = self.models[modelname](x.view(x.size(0), -1))
            if self.task_type == 'element2element':
                predictions = predictions.view(y.size())
        else:
            predictions, _ = self.models[modelname](x)
            if self.task_type == 'classification':
                predictions = predictions[:, -1, :]

        return predictions

    def compute_loss(self, modelname, predictions, y, ewc=False, task_id=None):
        loss = self.criterion(predictions, y)

        if ewc and task_id is not None:
            if task_id > 0:
                loss += self.ewc.ewc_loss(self.models[modelname], modelname, task_id)
        
        if loss.requires_grad:
            loss.backward()

        return loss.item()

    def compute_eval_metric(self, predictions, y):   
        if self.eval_metric is not None:
            with torch.no_grad():
                ev_metric = self.eval_metric(predictions, y)
        else:
            ev_metric = 0.
        
        return ev_metric

    def optimize(self, modelname):
        torch.nn.utils.clip_grad_norm_(self.models[modelname].parameters(), self.max_grad_norm)
        self.optimizers[modelname].step()
        self.optimizers[modelname].zero_grad()