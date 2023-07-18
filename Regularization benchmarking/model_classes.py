import torch
import torch.nn as nn
import torch.nn.functional as F


### LENET


class LeNet(nn.Module):
    def __init__(
        self,
        lr,
        momentum,
        in_channels=3,
        dropout_rate=0.0,
        orthogonal=True,
        noise_inject_input=False,
        noise_inject_weights=False,
        noise_stddev=0.05,
        N_images=10,
        l1=False,
        l1_lmbd=0.00001,
        l2=False,
        l2_lmbd=0.0001,
        l1_l2=False,
        soft_svb=False,
        soft_svb_lmbd=0.01,
        jacobi_reg=False,
        jacobi_reg_lmbd=0.001,
        jacobi_det_reg=False,
        jacobi_det_reg_lmbd=0.001,
        conf_penalty=False,
        conf_penalty_lmbd=0.1,
        label_smoothing=False,
        label_smoothing_lmbd=0.1,
    ):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=6, kernel_size=(5, 5)
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
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
        else:
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)

        # Parameters for regularization
        self.l1=l1
        self.l1_lmbd=l1_lmbd
        self.l2=l2
        self.l2_lmbd=l2_lmbd
        self.l1_l2=l1_l2
        self.soft_svb=soft_svb
        self.soft_svb_lmbd=soft_svb_lmbd
        self.jacobi_reg=jacobi_reg
        self.jacobi_reg_lmbd=jacobi_reg_lmbd
        self.jacobi_det_reg=jacobi_det_reg
        self.jacobi_det_reg_lmbd=jacobi_det_reg_lmbd
        self.conf_penalty=conf_penalty
        self.conf_penalty_lmbd=conf_penalty_lmbd
        self.label_smoothing=label_smoothing
        self.label_smoothing_lmbd=label_smoothing_lmbd
        self.counter = 0
        self.noise_stddev = noise_stddev
        self.noise_inject_input = noise_inject_input
        self.noise_inject_weights = noise_inject_weights

        self.L = nn.CrossEntropyLoss()
        self.opt = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def forward(self, x):
        if self.training and self.noise_inject_input:
            noise = torch.randn_like(x) * self.noise_stddev
            x = x + noise

        if self.training and self.noise_inject_weights:
            conv1_weight = (
                self.conv1.weight
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
        x = x.view(-1, 16 * 5 * 5)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = F.softmax(self.fc3(x), dim=1)
        return x

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

        # L1 and L2 regularization
        if self.l1_l2:
            l1_l2_loss = 0
            l1_loss = 0
            l2_loss = 0
            for param in self.parameters():
                l1_loss += self.l1_lmbd * torch.norm(param, 1)
                l2_loss += self.l2_lmbd * torch.norm(param, 2) ** 2
            l1_l2_loss = l1_loss + l2_loss
            loss += l1_l2_loss
            return loss, l1_l2_loss

        # Soft SVB regularization
        if self.soft_svb:
            w = self.fc1.weight
            # Main loss function term for soft SVB from Jia et al. 2019:
            w_orth = w.transpose(0, 1) @ w  # W^T * W
            w_orth = w_orth - torch.eye(w_orth.shape[0])  # W^T * W - I
            soft_svb_loss = self.soft_svb_lmbd * torch.linalg.norm(w_orth, ord="fro") ** 2
            loss += soft_svb_loss
            return loss, soft_svb_loss

        # Jacobi regularization
        if self.jacobi_reg:
            if self.counter % 50 == 0:
                self.counter += 1
                jacobi = torch.autograd.functional.jacobian(self.forward, x)
                jacobi = jacobi.transpose(-2, -1) @ jacobi
                jacobi_loss = self.jacobi_reg_lmbd * (torch.norm(jacobi) ** 2)
                #    jacobi_reg_lmbd
                #    * (torch.linalg.norm(square_jacobi, ord="fro", dim=(1, 2)) ** 2).sum()
                # )
                loss += jacobi_loss
                print("calculated jacobi")
                return loss, jacobi_loss
            else:
                self.counter += 1
                return loss, 0

        # Jacobi determinant regularization
        if self.jacobi_det_reg:
            if self.counter % 100 == 1:
                self.counter += 1
                jacobi = torch.autograd.functional.jacobian(self.forward, x)
                jacobi = jacobi.transpose(-2, -1) @ jacobi
                jacobi_loss = (
                    self.jacobi_det_reg_lmbd * ((torch.linalg.det(jacobi) - 1) ** 2).sum()
                )
                loss += jacobi_loss
                print("calculated jacobi")
                return loss, jacobi_loss
            else:
                self.counter += 1
                return loss, 0

        # Confidence penalty regularization
        if self.conf_penalty:
            conf_penalty_loss = (
                -self.conf_penalty_lmbd * (y_pred * torch.log(y_pred)).sum(dim=1).mean()
            )
            loss += conf_penalty_loss
            return loss, conf_penalty_loss

        return loss, loss

    def train_step(
        self,
        data,
        labels,
    ):
        if self.label_smoothing:
            num_classes = 10
            smoothed_labels = (
                1 - self.label_smoothing_lmbd
            ) * labels + self.label_smoothing_lmbd / num_classes

        self.opt.zero_grad()

        loss, reg_loss = self.loss_fn(
            data,
            smoothed_labels.long() if self.label_smoothing else labels.long(),
        )
        loss.backward()
        self.opt.step()
        if reg_loss != 0:
            return loss.item(), reg_loss.item()
        else:
            return loss.item(), reg_loss
