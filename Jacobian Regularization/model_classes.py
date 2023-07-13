import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

### LENET


class LeNet_MNIST(nn.Module):
    def __init__(
        self,
        lr,
        momentum,
        in_channels=1,
        dropout_rate=0.0,
        orthogonal=False,
        noise_inject_input=False,
        noise_inject_weights=False,
        noise_stddev=0.05,
        N_images=10,
        l1=False,
        l1_lmbd=0.00001,
        l2=False,
        l2_lmbd=0.0001,
        jacobi_reg=False,
        jacobi_reg_lmbd=0.001,
    ):
        super(LeNet_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=6, kernel_size=(5, 5)
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(in_features=84, out_features=N_images)

        if orthogonal:
            # Orthogonal initialization of weight matrices
            nn.init.orthogonal_(self.conv1.weight)
            nn.init.orthogonal_(self.conv2.weight)
            nn.init.orthogonal_(self.fc1.weight)
            nn.init.orthogonal_(self.fc2.weight)
            nn.init.orthogonal_(self.fc3.weight)

        self.noise_stddev = noise_stddev
        self.noise_inject_input = noise_inject_input
        self.noise_inject_weights = noise_inject_weights

        self.L = nn.CrossEntropyLoss()
        self.opt = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

        # Regularization parameters
        self.l1 = l1
        self.l1_lmbd = l1_lmbd
        self.l2 = l2
        self.l2_lmbd = l2_lmbd
        self.jacobi_reg = jacobi_reg
        self.jacobi_reg_lmbd = jacobi_reg_lmbd

    def forward(self, x):
        if self.training and self.noise_inject_input:
            noise = torch.randn_like(x) * self.noise_stddev
            x = x + noise

        if self.training and self.noise_inject_weights:
            conv1_weight = (
                self.conv1.weight.detach()
                + torch.randn_like(self.conv1.weight) * self.noise_stddev
            )
            x = F.conv2d(
                x,
                conv1_weight,
                self.conv1.bias,
                self.conv1.stride,
                self.conv1.padding,
                self.conv1.dilation,
                self.conv1.groups,
            )
        else:
            conv1_weight = self.conv1.weight

        if not self.noise_inject_weights:
            x = self.conv1(x)

        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        x = x.view(-1, 16 * 4 * 4)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
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

        # L1 regularization
        if self.l1:
            l1_loss = 0
            for param in self.parameters():
                l1_loss += self.l1_lmbd * torch.norm(param, 1)
            loss += l1_loss
            return loss, l1_loss

        # L2 regularization
        if self.l2:
            l2_loss = 0
            for param in self.parameters():
                l2_loss += self.l2_lmbd * torch.norm(param, 2) ** 2
            loss += l2_loss
            return loss, l2_loss

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
        return loss.item(), reg_loss.item() if reg_loss != 0 else reg_loss
