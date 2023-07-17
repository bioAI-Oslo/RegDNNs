import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.checkpoint import checkpoint

### LENET


class LeNet_MNIST(nn.Module):
    """Class implemented as in Hoffman 2019 for Jacobian regularization."""

    def __init__(
        self,
        lr=0.1,
        momentum=0.9,
        dropout_rate=0.5,
        l2_lmbd=0.0,
        jacobi_reg=False,
        jacobi_reg_lmbd=0.01,
        svb_reg=False,
        svb_freq=100,
        svb_eps=0.01,
    ):
        super(LeNet_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=(5, 5),
            stride=1,
            padding=2,
        )
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=(5, 5), stride=1, padding=0
        )
        self.fc1 = nn.Linear(in_features=400, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        # Orthogonal initialization if svb_reg is True
        if svb_reg:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
        else:
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)

        self.L = nn.CrossEntropyLoss()
        self.opt = torch.optim.SGD(
            self.parameters(), lr=lr, momentum=momentum, weight_decay=l2_lmbd
        )

        # Regularization parameters
        self.jacobi_reg = jacobi_reg
        self.jacobi_reg_lmbd = jacobi_reg_lmbd
        self.svb_reg = svb_reg
        self.svb_freq = svb_freq
        self.svb_eps = svb_eps
        self.training_steps = 0  # Track training steps

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = x.view(-1, 400)
        x = self.dropout(torch.tanh(self.fc1(x)))
        x = self.dropout(torch.tanh(self.fc2(x)))
        x = self.fc3(x)  # No softmax as CrossEntropyLoss does it implicitly
        return x

    def jacobian_regularizer(self, x):
        C = x.shape[1]  # number of classes
        JF = 0
        nproj = 1  # number of random projections

        for _ in range(nproj):
            v = torch.randn(x.shape[0], C).to(x.device)  # random vector
            v_hat = v / torch.norm(v, dim=1, keepdim=True)  # normalize

            z = self(x)  # forward pass
            v_hat_dot_z = torch.einsum("bi,bi->b", v_hat, z)  # dot product

            Jv = grad(v_hat_dot_z.sum(), x, create_graph=True)[
                0
            ]  # Jacobian-vector product

            JF += C * (Jv**2).sum() / (nproj * len(x))

        return JF

    def loss_fn(
        self,
        x,
        y,
    ):
        y_pred = self(x.float())
        loss = self.L(y_pred, y)

        # Jacobi regularization
        if self.jacobi_reg:
            x.requires_grad_(True)
            jacobi_loss = self.jacobi_reg_lmbd * self.jacobian_regularizer(x)
            loss += jacobi_loss
            return loss, jacobi_loss

        return loss, loss

    def train_step(
        self,
        data,
        labels,
    ):
        self.opt.zero_grad()
        loss, reg_loss = self.loss_fn(
            data,
            labels.long(),
        )
        loss.backward()
        self.opt.step()
        self.training_steps += 1

        if self.svb_reg and self.training_steps % self.svb_freq == 0:
            with torch.no_grad():
                for m in self.modules():
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                        U, S, V = torch.svd(m.weight)
                        S = torch.clamp(S, 1 / (1 + self.svb_eps), 1 + self.svb_eps)
                        m.weight.data = torch.matmul(U, torch.matmul(S.diag(), V.t()))
        return loss.item(), reg_loss.item() if reg_loss != 0 else reg_loss
