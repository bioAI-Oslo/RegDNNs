import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import functional
import functorch as ft

class UMINN(nn.Module):
    def __init__(self, in_size = 2, out_size = 4, lr = 0.01, bias = False):
        super().__init__()
        self.L1 = nn.Linear(in_size, out_size, bias = bias)
        self.L = nn.CrossEntropyLoss()
        self.opt = Adam(self.parameters(), lr = lr)
    
    def forward(self, x):
        x = self.L1(x)
        return F.softmax(x, dim = -1)
    
    def loss_fn(self, x, y):
        y_pred = self(x.float())
        loss = self.L(y_pred, y)
        return loss
    
    def train_step(self, data, labels):
        self.opt.zero_grad()
        loss = self.loss_fn(data, labels.long())
        loss.backward()
        self.opt.step()
        return loss.item()

    
class UMINN_L1(UMINN):
    def __init__(self, in_size = 2, out_size = 4, lmbd = 0.01, lr = 0.01, bias = False):
        super().__init__(in_size = in_size, out_size = out_size, lr = lr, bias = bias)
        self.lmbd = lmbd

    def loss_fn(self, x, y):
        y_pred = self(x.float())
        loss = self.L(y_pred, y)
        loss += self.lmbd * self.L1.weight.abs().sum() # L1 reg.
        return loss

    
class UMINN_L2(UMINN):
    def __init__(self, in_size = 2, out_size = 4,  lmbd = 0.01, lr = 0.01, bias = False):
        super().__init__(in_size = in_size, out_size = out_size, lr = lr, bias = bias)
        self.lmbd = lmbd

    def loss_fn(self, x, y):
        y_pred = self(x.float())
        loss = self.L(y_pred, y)
        loss += (self.lmbd * self.L1.weight**2).sum() # L2 reg.
        return loss

class UMINN_L1_L2(UMINN):
    def __init__(self, in_size = 2, out_size = 4,  lmbd1 = 0.01, lmbd2 = 0.01, lr = 0.01, bias = False):
        super().__init__(in_size = in_size, out_size = out_size, lr = lr, bias = bias)
        self.lmbd1 = lmbd1
        self.lmbd2 = lmbd2

    def loss_fn(self, x, y):
        y_pred = self(x.float())
        loss = self.L(y_pred, y)
        loss += self.lmbd1 * self.L1.weight.abs().sum() # L1 reg.
        loss += (self.lmbd2 * self.L1.weight**2).sum() # L2 reg.
        return loss

    
class UMINN_SVB(UMINN):
    def __init__(self, in_size = 2, out_size = 4, lr = 0.01, bias = False):
        super().__init__(in_size = in_size, out_size = out_size, lr = lr, bias = bias)
        nn.init.orthogonal_(self.L1.weight, gain = 1.0)
    
    def svb(self, eps = 0.001):
        """Implements hard singular value bounding as described in Jia et al. 2019.
        Keyword Arguments:
            eps -- Small constant that sets the weights a small interval around 1 (default: {0.001})
        """
        old_weights = self.L1.weight.data.clone().detach()
        w = torch.linalg.svd(old_weights, full_matrices=False)
        U, sigma, V = w[0], w[1], w[2]
        for i in range(len(sigma)):
            if sigma[i] > 1 + eps:
                sigma[i] = 1 + eps
            elif sigma[i] < 1/(1 + eps):
                sigma[i] = 1/(1 + eps)
            else:
                pass
        new_weights = U @ torch.diag(sigma) @ V
        self.L1.weight.data = new_weights

class UMINN_SVB_Soft(UMINN):
    def __init__(self, in_size = 2, out_size = 4, lr = 0.01, bias = False, lmbd = 0.01):
        super().__init__(in_size = in_size, out_size = out_size, lr = lr, bias = bias)
        torch.nn.init.orthogonal_(self.L1.weight, gain = 1.0)
        self.lmbd = lmbd
    
    def loss_fn(self, x, y):
        y_pred = self(x.float())
        loss = self.L(y_pred, y)
        w = self.L1.weight
        
        # Main loss function term for soft SVB from Jia et al. 2019:
        w_orth = w.transpose(0,1) @ w # W^T * W
        w_orth = w_orth - torch.eye(w_orth.shape[0]) # W^T * W - I
        loss += self.lmbd * torch.linalg.norm(w_orth, ord = "fro")**2 # Note that since we only have one layer we need not do this over weights from more layers
        return loss

class UMINN_Jacobi(UMINN):
    def __init__(self, in_size = 2, out_size = 4, lr = 0.01, bias = False, lmbd = 0.01):
        super().__init__(in_size = in_size, out_size = out_size, lr = lr, bias = bias)
        torch.nn.init.orthogonal_(self.L1.weight, gain = 1.0)
        self.lmbd = lmbd

    def loss_fn(self, x, y):
        y_pred = self(x.float())
        loss = self.L(y_pred, y)
        for i in x:
            jacobi = functional.jacobian(self.L1, i.float(), create_graph = True, strict = True, vectorize = False) # Jacobian of the first layer
            loss += self.lmbd * (torch.linalg.det(jacobi.transpose(0,1) @ jacobi) - 1) # Determinant of the Jacobian (Jacobian is a square matrix
        return loss

class UMINN_Jacobi_2(UMINN):
    def __init__(self, in_size = 2, out_size = 4, lr = 0.01, bias = False, lmbd = 0.01):
        super().__init__(in_size = in_size, out_size = out_size, lr = lr, bias = bias)
        torch.nn.init.orthogonal_(self.L1.weight, gain = 1.0)
        self.lmbd = lmbd
        self.cross_losses = []
        self.jacobi_losses = []

    def loss_fn(self, x, y):
        y_pred = self(x.float())
        loss = self.L(y_pred, y)
        self.cross_losses.append(loss.item())

        jacobi = ft.vmap(ft.jacfwd(self.L1))(x)
        square_jacobi = (jacobi.transpose(1,2) @ jacobi)
    
        jacobi_loss = self.lmbd * ((torch.linalg.det(square_jacobi) - 1)**2).sum()
        self.jacobi_losses.append(jacobi_loss.item())
        loss += jacobi_loss
        return loss

                
class UMINN_Jacobi_Reg(UMINN):
    def __init__(self, in_size = 2, out_size = 4, lr = 0.01, bias = False, lmbd = 0.01):
        super().__init__(in_size = in_size, out_size = out_size, lr = lr, bias = bias)
        self.lmbd = lmbd
        self.cross_losses = []
        self.jacobi_losses = []

    def loss_fn(self, x, y):
        y_pred = self(x.float())
        loss = self.L(y_pred, y)
        self.cross_losses.append(loss.item())
        jacobi = ft.vmap(ft.jacfwd(self.L1))(x)
        square_jacobi = (jacobi.transpose(1, 2) @ jacobi)
        # Loss minimize jacobi
        jacobi_loss = self.lmbd * (torch.linalg.norm(square_jacobi, ord = "fro", dim = (1, 2))**2).sum()
        self.jacobi_losses.append(jacobi_loss.item())
        loss += jacobi_loss
        return loss
