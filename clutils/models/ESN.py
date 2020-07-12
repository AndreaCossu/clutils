import torch
import torch.nn as nn
import math
from sklearn.linear_model import LogisticRegression
from ..globals import OUTPUT_TYPE, choose_output

class ESN(nn.Module):
    """
    Weights are described by the following names:
    'inp2reservoir_w' -> input to reservoir weights
    'inp_b' -> input bias
    'reservoir2reservoir' -> recurrent weights
    'reservoir2out_w' -> readout weights
    'out_b' -> reservoir bias

    Use dict(model.weights[weightname].items()) to get {key : value} dict for parameters of weightname.
    """

    def __init__(self, input_size, reservoir_size, output_size, device,
        alpha=1.0, spectral_radius=0.9, sparsity=0., orthogonal=False, logistic=False, out_activation=None):

        '''
        :param alpha: scalar leaking rate in (0, 1]
        :param beta: ridge regression regularization hyperparameter
        :param spectral_radius: spectral radius of the reservoir connection matrix
        :param sparsity: percentage of dead connections
        :param orthogonal: use orthogonal reservoir (spectral radius = 1)
        :param logistic: use SciKit learn logistic regression to train instead of Pytorch gradient descent
        :param out_activation: torch.Function(args) (e.g. torch.Tanh()) or None if no output function is needed. Default None.
        '''

        super(ESN, self).__init__()

        self.output_type = OUTPUT_TYPE.OUTH

        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.device = device
        self.alpha = alpha if alpha > 0 and alpha <= 1 else 1 # no leaky if alpha is not in (0,1]

        self.out_activation = out_activation

        reservoir_w = nn.init.sparse_(torch.empty(reservoir_size, reservoir_size, requires_grad=False), sparsity)
        if not orthogonal:
            # with complex eigenvalues (reservoir_size x 2 matrix), the spectral radius is the largest sqrt( r^2 + c^2)
            # where r is real part and c is complex part of the eigenvalues
            current_spectral_radius = torch.max(torch.sqrt(torch.sum(torch.eig(reservoir_w)[0]**2, dim=1))).item()
            reservoir_w *= spectral_radius / current_spectral_radius
            reservoir_w = ( reservoir_w - (1 - self.alpha) * torch.eye(self.reservoir_size) ) * (1/self.alpha)
        else:
            reservoir_w = nn.init.orthogonal_(reservoir_w)
        reservoir_w = reservoir_w.to(self.device)

        # fixed weights
        self.fixed_weights = {
            'inp2reservoir_w' : (1 / math.sqrt(reservoir_size)) * torch.randn(input_size, reservoir_size, requires_grad=False, device=device),
            'inp_b' : (1 / math.sqrt(reservoir_size)) * torch.randn(reservoir_size, requires_grad=False, device=device),
            'reservoir2reservoir_w' : reservoir_w
        }

        # trainable weights
        self.weights = nn.ParameterDict([
            [ 'reservoir2out_w', nn.Parameter(torch.randn(reservoir_size, output_size)) ],
            [ 'out_b', nn.Parameter(torch.randn(output_size)) ]
        ])
        self.weights = self.weights.to(device)

        if logistic:
            self.logisticregression = LogisticRegression()


    def forward(self, x, res=None):
        '''
        
        :param x: (B, T, I) batch of input sequences

        '''

        if res is None:
            res = torch.zeros(x.size(0), self.reservoir_size, device=self.device)
        
        outs = torch.empty(x.size(0), x.size(1), self.output_size, device=self.device)  # matrix of all outputs

        for t in range(x.size(1)):
            xt = x[:, t, :]

            resinp = torch.mm(xt, self.fixed_weights['inp2reservoir_w']) #+ self.fixed_weights['inp_b']
            resres = torch.mm(res, self.fixed_weights['reservoir2reservoir_w'])
            res_candidate = torch.tanh(resinp + resres)

            res = (1-self.alpha) * res + self.alpha * res_candidate

            o = torch.mm(res, self.weights['reservoir2out_w']) + self.weights['out_b']
            
            if self.out_activation is not None:
                o = self.out_activation(o)

            outs[:, t, :] = o
            
        
        return choose_output(outs, o, self.output_type)


    
    def logistic_regression(self, x, targets=None, res=None, num_patterns_training=0):
        '''
        x: entire dataset at once (B, T, I) where B = # patterns in dataset
        targets: targets for all patterns (B) containing scalar class values
        num_targets: number of input patterns to be (randomly) selected for training. 0 means all the patterns in x.
        '''

        with torch.no_grad():
            if res is None:
                res = torch.zeros(x.size(0), self.reservoir_size, device=self.device)

            # get reservoir activations
            for t in range(x.size(1)):
                xt = x[:, t, :]

                resinp = torch.mm(xt, self.fixed_weights['inp2reservoir_w']) #+ self.fixed_weights['inp_b']
                resres = torch.mm(res, self.fixed_weights['reservoir2reservoir_w'])
                res_candidate = torch.tanh(resinp + resres)

                res = (1-self.alpha) * res + self.alpha * res_candidate

            
            # train with logistic regression
            if targets is not None:
                if num_patterns_training > 0:
                    random_idx = torch.randperm(x.size(0))[:num_patterns_training]
                    res_training = res[random_idx]
                    targets = targets[random_idx]
                else:
                    res_training = res

                self.logisticregression.fit(res_training.cpu().numpy(), targets.cpu().numpy())
                self.weights['reservoir2out_w'] = torch.nn.Parameter(torch.tensor(self.logisticregression.coef_).float().t())
                self.weights['out_b'] = torch.nn.Parameter(torch.tensor(self.logisticregression.intercept_).float())
                self.weights = self.weights.to(self.device)
            # compute output
            out = torch.mm(res, self.weights['reservoir2out_w']) + self.weights['out_b'] # (batch_size, output_size)

            return choose_output(out, res_training, self.output_type)

