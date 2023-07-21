import torch
import torch.nn as nn
from torch.autograd import grad
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(
        self,
        lr=0.1,
        momentum=0.9,
        dropout_rate=0.0,
        noise_inject_inputs=False,
        noise_inject_weights=False,
        noise_stddev=0.05,
        l1=False,
        l1_lmbd=0.0005,
        l2=False,
        l2_lmbd=0.0005,
        l1_l2=False,
        svb=False,
        svb_freq=600,
        svb_eps=0.05,
        soft_svb=False,
        soft_svb_lmbd=0.01,
        jacobi_reg=False,
        jacobi_reg_lmbd=0.01,
        jacobi_det_reg=False,
        jacobi_det_reg_lmbd=0.01,
        conf_penalty=False,
        conf_penalty_lmbd=0.05,
        label_smoothing=False,
        label_smoothing_lmbd=0.1,
    ):
        super(LeNet, self).__init__()
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
        if svb or soft_svb:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    weight_mat = m.weight.data.view(
                        m.out_channels, -1
                    )  # Reshape to 2D matrix
                    nn.init.orthogonal_(weight_mat)  # Apply orthogonal initialization
                    m.weight.data = weight_mat.view_as(
                        m.weight.data
                    )  # Reshape back to original dimensions
                elif isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
        else:
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)

        # Parameters for regularization
        self.l1 = l1
        self.l1_lmbd = l1_lmbd
        self.l2 = l2
        self.l2_lmbd = l2_lmbd
        self.l1_l2 = l1_l2
        self.svb = svb
        self.svb_freq = svb_freq
        self.svb_eps = svb_eps
        self.soft_svb = soft_svb
        self.soft_svb_lmbd = soft_svb_lmbd
        self.jacobi_reg = jacobi_reg
        self.jacobi_reg_lmbd = jacobi_reg_lmbd
        self.jacobi_det_reg = jacobi_det_reg
        self.jacobi_det_reg_lmbd = jacobi_det_reg_lmbd
        self.conf_penalty = conf_penalty
        self.conf_penalty_lmbd = conf_penalty_lmbd
        self.label_smoothing = label_smoothing
        self.label_smoothing_lmbd = label_smoothing_lmbd
        self.counter = 0
        self.training_steps = 0
        self.noise_stddev = noise_stddev
        self.noise_inject_inputs = noise_inject_inputs
        self.noise_inject_weights = noise_inject_weights

        self.L = nn.CrossEntropyLoss()
        self.smoothed_L = nn.KLDivLoss(reduction="batchmean")
        self.opt = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def forward(self, x):
        if self.training and self.noise_inject_inputs:
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
            x = self.conv1(x)

        x = self.pool(torch.tanh(x))
        x = self.conv2(x)
        x = self.pool(torch.tanh(x))
        x = x.view(-1, 400)
        x = self.dropout(torch.tanh(self.fc1(x)))
        x = self.dropout(torch.tanh(self.fc2(x)))
        x = self.fc3(x)  # No softmax as CrossEntropyLoss does it implicitly
        return x

    def jacobian_regularizer(self, x):
        """
        Calculates the Jacobian regularization term for an input batch.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        JF : torch.Tensor
            The Jacobian regularization term.
        """
        C = x.shape[1]  # Number of classes in dataset
        JF = 0  # Initialize Jacobian Frobenius norm
        nproj = 1  # Number of random projections

        for _ in range(nproj):
            v = torch.randn(x.shape[0], C).to(x.device)  # Generate random vector
            v_hat = v / torch.norm(v, dim=1, keepdim=True)  # Normalize

            z = self(x)  # Forward pass to get predictions
            v_hat_dot_z = torch.einsum("bi, bi->b", v_hat, z)  # Calculate dot product

            Jv = grad(v_hat_dot_z.sum(), x, create_graph=True)[
                0
            ].detach()  # Compute Jacobian-vector product

            JF += (
                C * (Jv**2).sum() / (nproj * len(x))
            )  # Add square of Jv to Frobenius norm

        return JF

    def loss_fn(
        self,
        x,
        y,
    ):
        y_pred = self(x.float())

        # Label smoothing
        if self.label_smoothing:
            y_smooth = torch.full_like(
                y_pred, fill_value=self.label_smoothing_lmbd / (10 - 1)  # N_images - 1
            )
            y_smooth.scatter_(1, y.unsqueeze(1), 1 - self.label_smoothing_lmbd)
            loss = self.smoothed_L(y_pred.log_softmax(dim=1), y_smooth.detach())
        else:
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
            w_orth = w_orth - torch.eye(w_orth.shape[0]).to(
                w_orth.device
            )  # W^T * W - I
            soft_svb_loss = (
                self.soft_svb_lmbd * torch.linalg.norm(w_orth, ord="fro") ** 2
            )
            loss += soft_svb_loss
            return loss, soft_svb_loss

        # Jacobi regularization
        if self.jacobi_reg:
            x.requires_grad_(True)
            # Compute and add Jacobian regularization term
            jacobi_loss = self.jacobi_reg_lmbd * self.jacobian_regularizer(x)
            loss += jacobi_loss
            return loss, jacobi_loss

        # Jacobi determinant regularization
        if self.jacobi_det_reg:
            if self.counter % 100 == 1:
                self.counter += 1
                jacobi = torch.autograd.functional.jacobian(self.forward, x)
                jacobi = jacobi.transpose(-2, -1) @ jacobi
                jacobi_loss = (
                    self.jacobi_det_reg_lmbd
                    * ((torch.linalg.det(jacobi) - 1) ** 2).sum()
                )
                loss += jacobi_loss
                return loss, jacobi_loss
            else:
                self.counter += 1
                return loss, 0

        # Confidence penalty regularization
        if self.conf_penalty:
            probabilities = torch.nn.functional.softmax(y_pred, dim=1)
            conf_penalty_loss = (
                -self.conf_penalty_lmbd
                * (probabilities * torch.log(probabilities + 1e-8)).sum(dim=1).mean()
            )
            loss += conf_penalty_loss
            return loss, conf_penalty_loss

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

        # If SVB regularization, apply every svb_freq steps
        if self.svb and self.training_steps % self.svb_freq == 0:
            with torch.no_grad():  # Do not track gradients
                for m in self.modules():  # Loop over modules
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                        weight_orig_shape = (
                            m.weight.shape
                        )  # Get original shape of weights
                        weight_matrix = m.weight.view(
                            weight_orig_shape[0], -1
                        )  # Flatten weights
                        U, S, V = torch.svd(
                            weight_matrix
                        )  # Singular value decomposition on weights
                        S = torch.clamp(
                            S, 1 / (1 + self.svb_eps), 1 + self.svb_eps
                        )  # Clamp singular values within range decided by svb_eps
                        m.weight.data = torch.matmul(
                            U, torch.matmul(S.diag(), V.t())
                        ).view(
                            weight_orig_shape
                        )  # Update weights using clamped singular values

        self.training_steps += 1

        return loss.item(), reg_loss.item() if reg_loss != 0 else reg_loss


class DDNet(nn.Module):
    def __init__(
        self,
        lr=0.1,
        momentum=0.9,
        dropout_rate=0.0,
        noise_inject_inputs=False,
        noise_inject_weights=False,
        noise_stddev=0.05,
        l1=False,
        l1_lmbd=0.0005,
        l2=False,
        l2_lmbd=0.0005,
        l1_l2=False,
        svb=False,
        svb_freq=600,
        svb_eps=0.05,
        soft_svb=False,
        soft_svb_lmbd=0.01,
        jacobi_reg=False,
        jacobi_reg_lmbd=0.01,
        jacobi_det_reg=False,
        jacobi_det_reg_lmbd=0.01,
        conf_penalty=False,
        conf_penalty_lmbd=0.05,
        label_smoothing=False,
        label_smoothing_lmbd=0.1,
    ):
        super(DDNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(3, 3),
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=0
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=0
        )
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=0
        )
        self.fc1 = nn.Linear(in_features=3200, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=10)
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        # Orthogonal initialization if svb_reg is True
        if svb or soft_svb:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    weight_mat = m.weight.data.view(
                        m.out_channels, -1
                    )  # Reshape to 2D matrix
                    nn.init.orthogonal_(weight_mat)  # Apply orthogonal initialization
                    m.weight.data = weight_mat.view_as(
                        m.weight.data
                    )  # Reshape back to original dimensions
                elif isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
        else:
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)
            nn.init.xavier_uniform_(self.conv3.weight)
            nn.init.xavier_uniform_(self.conv4.weight)
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)

        # Parameters for regularization
        self.l1 = l1
        self.l1_lmbd = l1_lmbd
        self.l2 = l2
        self.l2_lmbd = l2_lmbd
        self.l1_l2 = l1_l2
        self.svb = svb
        self.svb_freq = svb_freq
        self.svb_eps = svb_eps
        self.soft_svb = soft_svb
        self.soft_svb_lmbd = soft_svb_lmbd
        self.jacobi_reg = jacobi_reg
        self.jacobi_reg_lmbd = jacobi_reg_lmbd
        self.jacobi_det_reg = jacobi_det_reg
        self.jacobi_det_reg_lmbd = jacobi_det_reg_lmbd
        self.conf_penalty = conf_penalty
        self.conf_penalty_lmbd = conf_penalty_lmbd
        self.label_smoothing = label_smoothing
        self.label_smoothing_lmbd = label_smoothing_lmbd
        self.counter = 0
        self.training_steps = 0
        self.noise_stddev = noise_stddev
        self.noise_inject_inputs = noise_inject_inputs
        self.noise_inject_weights = noise_inject_weights

        self.L = nn.CrossEntropyLoss()
        self.smoothed_L = nn.KLDivLoss(reduction="batchmean")
        self.opt = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def forward(self, x):
        if self.training and self.noise_inject_inputs:
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
            x = self.conv1(x)

        x = F.relu(x)
        x = self.conv2(x)
        x = self.pool(F.relu(x))
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.pool(F.relu(x))
        x = x.view(-1, 3200)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)  # No softmax as CrossEntropyLoss does it implicitly
        return x

    def jacobian_regularizer(self, x):
        """
        Calculates the Jacobian regularization term for an input batch.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        JF : torch.Tensor
            The Jacobian regularization term.
        """
        C = x.shape[1]  # Number of classes in dataset
        JF = 0  # Initialize Jacobian Frobenius norm
        nproj = 1  # Number of random projections

        for _ in range(nproj):
            v = torch.randn(x.shape[0], C).to(x.device)  # Generate random vector
            v_hat = v / torch.norm(v, dim=1, keepdim=True)  # Normalize

            z = self(x)  # Forward pass to get predictions
            v_hat_dot_z = torch.einsum("bi, bi->b", v_hat, z)  # Calculate dot product

            Jv = grad(v_hat_dot_z.sum(), x, create_graph=True)[
                0
            ].detach()  # Compute Jacobian-vector product

            JF += (
                C * (Jv**2).sum() / (nproj * len(x))
            )  # Add square of Jv to Frobenius norm

        return JF

    def loss_fn(
        self,
        x,
        y,
    ):
        y_pred = self(x.float())

        # Label smoothing
        if self.label_smoothing:
            y_smooth = torch.full_like(
                y_pred, fill_value=self.label_smoothing_lmbd / (10 - 1)  # N_images - 1
            )
            y_smooth.scatter_(1, y.unsqueeze(1), 1 - self.label_smoothing_lmbd)
            loss = self.smoothed_L(y_pred.log_softmax(dim=1), y_smooth.detach())
        else:
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
            w_orth = w_orth - torch.eye(w_orth.shape[0]).to(
                w_orth.device
            )  # W^T * W - I
            soft_svb_loss = (
                self.soft_svb_lmbd * torch.linalg.norm(w_orth, ord="fro") ** 2
            )
            loss += soft_svb_loss
            return loss, soft_svb_loss

        # Jacobi regularization
        if self.jacobi_reg:
            x.requires_grad_(True)
            # Compute and add Jacobian regularization term
            jacobi_loss = self.jacobi_reg_lmbd * self.jacobian_regularizer(x)
            loss += jacobi_loss
            return loss, jacobi_loss

        # Jacobi determinant regularization
        if self.jacobi_det_reg:
            if self.counter % 100 == 1:
                self.counter += 1
                jacobi = torch.autograd.functional.jacobian(self.forward, x)
                jacobi = jacobi.transpose(-2, -1) @ jacobi
                jacobi_loss = (
                    self.jacobi_det_reg_lmbd
                    * ((torch.linalg.det(jacobi) - 1) ** 2).sum()
                )
                loss += jacobi_loss
                return loss, jacobi_loss
            else:
                self.counter += 1
                return loss, 0

        # Confidence penalty regularization
        if self.conf_penalty:
            probabilities = torch.nn.functional.softmax(y_pred, dim=1)
            conf_penalty_loss = (
                -self.conf_penalty_lmbd
                * (probabilities * torch.log(probabilities + 1e-8)).sum(dim=1).mean()
            )
            loss += conf_penalty_loss
            return loss, conf_penalty_loss

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

        # If SVB regularization, apply every svb_freq steps
        if self.svb and self.training_steps % self.svb_freq == 0:
            with torch.no_grad():  # Do not track gradients
                for m in self.modules():  # Loop over modules
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                        weight_orig_shape = (
                            m.weight.shape
                        )  # Get original shape of weights
                        weight_matrix = m.weight.view(
                            weight_orig_shape[0], -1
                        )  # Flatten weights
                        U, S, V = torch.svd(
                            weight_matrix
                        )  # Singular value decomposition on weights
                        S = torch.clamp(
                            S, 1 / (1 + self.svb_eps), 1 + self.svb_eps
                        )  # Clamp singular values within range decided by svb_eps
                        m.weight.data = torch.matmul(
                            U, torch.matmul(S.diag(), V.t())
                        ).view(
                            weight_orig_shape
                        )  # Update weights using clamped singular values

        self.training_steps += 1

        return loss.item(), reg_loss.item() if reg_loss != 0 else reg_loss
