import torch
import torch.nn as nn
from torch.autograd import grad
import torch.nn.functional as F
from torchvision.models import resnet18


class LeNet_MNIST(nn.Module):
    """
    LeNet model architecture for MNIST digit classification with optional
    Jacobian, SVB, Dropout and L2 regularization, following the design used in Hoffman 2019.

    This model is a deep neural network with two convolutional layers and
    three fully connected layers.

    Parameters
    ----------
    lr : float
        The learning rate for the SGD optimizer.
    momentum : float
        The momentum for the SGD optimizer.
    dropout_rate : float
        The dropout rate applied after each fully connected layer.
    l2_lmbd : float
        The weight decay coefficient for L2 regularization.
    jacobi : bool
        If True, applies Jacobian regularization.
    jacobi_lmbd : float
        The regularization coefficient for Jacobian regularization.
    svb_reg : bool
        If True, applies singular value bounding (SVB) regularization.
    svb_freq : int
        The frequency for computing SVB.
    svb_eps : float
        The epsilon for computing SVB.
    """

    def __init__(
        self,
        lr=0.1,
        momentum=0.9,
        dropout_rate=0.5,
        l2_lmbd=0.0,
        jacobi=False,
        jacobi_lmbd=0.01,
        svb=False,
        svb_freq=600,
        svb_eps=0.05,
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
        if svb:
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
            # Initialize using Glorot initialization (as described in Glorot & Bengio 2010)
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
        self.jacobi = jacobi
        self.jacobi_lmbd = jacobi_lmbd
        self.svb = svb
        self.svb_freq = svb_freq
        self.svb_eps = svb_eps

        # Track training steps
        self.training_steps = 0

    def forward(self, x):
        """
        Forward pass of the LeNet model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor (predictions of the model).
        """
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
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
        """
        Calculates the loss function (cross-entropy) with an optional
        Jacobian regularization term.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        y : torch.Tensor
            The true labels.

        Returns
        -------
        loss : torch.Tensor
            The calculated loss.
        jacobi_loss : torch.Tensor
            The calculated Jacobian regularization loss (if applicable).
        """
        y_pred = self(x.float())  # Forward pass to get predictions
        loss = self.L(y_pred, y)  # Compute loss using cross-entropy

        # Jacobi regularization
        if self.jacobi:
            x.requires_grad_(True)
            # Compute and add Jacobian regularization term
            jacobi_loss = self.jacobi_lmbd * self.jacobian_regularizer(x)
            loss += jacobi_loss
            return loss, jacobi_loss

        return loss, loss

    def train_step(
        self,
        data,
        labels,
    ):
        """
        Performs one step of training: forward pass, loss calculation,
        backward pass and parameters update.

        Parameters
        ----------
        data : torch.Tensor
            The input tensor.
        labels : torch.Tensor
            The true labels.

        Returns
        -------
        loss : float
            The calculated loss.
        reg_loss : float
            The calculated regularization loss (if applicable).
        """
        self.opt.zero_grad()  # Zero out gradients
        loss, reg_loss = self.loss_fn(
            data,
            labels.long(),
        )  # Compute loss
        loss.backward()  # Backward pass to compute gradients
        self.opt.step()  # Update parameters

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
    """
    DDNet model architecture for CIFAR10/CIFAR100 classification with optional
    Jacobian, SVB, Dropout and L2 regularization, following the design used in Hoffman 2019.

    This model is a deep neural network with four convolutional layers and
    three fully connected layers.

    Parameters
    ----------
    lr : float
        The learning rate for the SGD optimizer.
    momentum : float
        The momentum for the SGD optimizer.
    dropout_rate : float
        The dropout rate applied after each fully connected layer.
    l2_lmbd : float
        The weight decay coefficient for L2 regularization.
    jacobi : bool
        If True, applies Jacobian regularization.
    jacobi_lmbd : float
        The regularization coefficient for Jacobian regularization.
    svb_reg : bool
        If True, applies singular value bounding (SVB) regularization.
    svb_freq : int
        The frequency for computing SVB.
    svb_eps : float
        The epsilon for computing SVB.
    """

    def __init__(
        self,
        dataset="cifar10",
        lr=0.01,  # 0.01 in Hoffman 2019 for DDNet on CIFAR10
        momentum=0.9,
        dropout_rate=0.5,
        l2_lmbd=0.0,
        jacobi=False,
        jacobi_lmbd=0.01,  # 0.01 in Hoffman 2019
        svb=False,
        svb_freq=600,
        svb_eps=0.05,
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
        self.fc3 = nn.Linear(
            in_features=256, out_features=10 if dataset == "cifar10" else 100
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        # Orthogonal initialization if svb is True
        if svb:
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
        self.l2_lmbd = l2_lmbd
        self.svb = svb
        self.svb_freq = svb_freq
        self.svb_eps = svb_eps
        self.jacobi = jacobi
        self.jacobi_lmbd = jacobi_lmbd
        self.training_steps = 0

        self.dataset = dataset
        self.L = nn.CrossEntropyLoss()
        self.opt = torch.optim.SGD(
            self.parameters(), lr=lr, momentum=momentum, weight_decay=l2_lmbd
        )

    def forward(self, x):
        """
        Forward pass of the DDNet model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor (predictions of the model).
        """
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
        num_features = self(x).shape[
            1
        ]  # instead of number of classes, get the feature size from the model output

        JF = 0  # Initialize Jacobian Frobenius norm
        nproj = 1  # Number of random projections

        for _ in range(nproj):
            v = torch.randn(x.shape[0], num_features).to(
                x.device
            )  # Generate random vector
            v_hat = v / torch.norm(v, dim=1, keepdim=True)  # Normalize

            z = self(x)  # Forward pass to get predictions
            v_hat_dot_z = torch.einsum("bi, bi->b", v_hat, z)  # Calculate dot product

            Jv = grad(v_hat_dot_z.sum(), x, create_graph=True)[
                0
            ].detach()  # Compute Jacobian-vector product

            JF += (
                num_features * (Jv**2).sum() / (nproj * len(x))
            )  # Add square of Jv to Frobenius norm

        return JF

    def loss_fn(
        self,
        x,
        y,
    ):
        """
        Calculates the loss function (cross-entropy) with an optional
        Jacobian regularization term.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        y : torch.Tensor
            The true labels.

        Returns
        -------
        loss : torch.Tensor
            The calculated loss.
        jacobi_loss : torch.Tensor
            The calculated Jacobian regularization loss (if applicable).
        """
        y_pred = self(x.float())  # Forward pass to get predictions
        loss = self.L(y_pred, y)  # Compute loss using cross-entropy

        # Jacobi regularization
        if self.jacobi:
            x.requires_grad_(True)
            # Compute and add Jacobian regularization term
            jacobi_loss = self.jacobi_lmbd * self.jacobian_regularizer(x)
            loss += jacobi_loss
            return loss, jacobi_loss

        return loss, loss

    def train_step(
        self,
        data,
        labels,
    ):
        """
        Performs one step of training: forward pass, loss calculation,
        backward pass and parameters update.

        Parameters
        ----------
        data : torch.Tensor
            The input tensor.
        labels : torch.Tensor
            The true labels.

        Returns
        -------
        loss : float
            The calculated loss.
        reg_loss : float
            The calculated regularization loss (if applicable).
        """
        self.opt.zero_grad()  # Zero out gradients
        loss, reg_loss = self.loss_fn(
            data,
            labels.long(),
        )  # Compute loss
        loss.backward()  # Backward pass to compute gradients
        self.opt.step()  # Update parameters

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


class ResNet18(nn.Module):  # For CIFAR100
    """
    ResNet18 model architecture for CIFAR100 classification with optional
    Jacobian, SVB, Dropout and L2 regularization, using preloaded pytorch model.

    This model is a deep neural network with 17 convolutional layers and
    one fully connected layers.

    Parameters
    ----------
    lr : float
        The learning rate for the SGD optimizer.
    momentum : float
        The momentum for the SGD optimizer.
    dropout_rate : float
        The dropout rate applied after each fully connected layer.
    l2_lmbd : float
        The weight decay coefficient for L2 regularization.
    jacobi : bool
        If True, applies Jacobian regularization.
    jacobi_lmbd : float
        The regularization coefficient for Jacobian regularization.
    svb_reg : bool
        If True, applies singular value bounding (SVB) regularization.
    svb_freq : int
        The frequency for computing SVB.
    svb_eps : float
        The epsilon for computing SVB.
    """

    def __init__(
        self,
        lr=0.1,  # 0.1 in Hoffman 2019
        momentum=0.9,
        l2_lmbd=0.0,
        jacobi=False,
        jacobi_lmbd=0.01,  # 0.01 in Hoffman 2019
        svb=False,
        svb_freq=600,
        svb_eps=0.05,
    ):
        super(ResNet18, self).__init__()
        # Initialize ResNet18 with pretrained=False since we're using CIFAR100, not ImageNet
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(
            self.model.fc.in_features, 100
        )  # Adapting for CIFAR100

        # Parameters for regularization
        self.l2_lmbd = l2_lmbd
        self.svb = svb
        self.svb_freq = svb_freq
        self.svb_eps = svb_eps
        self.jacobi = jacobi
        self.jacobi_lmbd = jacobi_lmbd
        self.training_steps = 0

        self.L = nn.CrossEntropyLoss()
        self.opt = torch.optim.SGD(
            self.parameters(), lr=lr, momentum=momentum, weight_decay=l2_lmbd
        )

    def forward(self, x):
        """
        Forward pass of the ResNet18 model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x : torch.Tensor
            The output tensor (predictions of the model).
        """
        return self.model(x)

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
        num_features = self(x).shape[
            1
        ]  # instead of number of classes, get the feature size from the model output

        JF = 0  # Initialize Jacobian Frobenius norm
        nproj = 1  # Number of random projections

        for _ in range(nproj):
            v = torch.randn(x.shape[0], num_features).to(
                x.device
            )  # Generate random vector
            v_hat = v / torch.norm(v, dim=1, keepdim=True)  # Normalize

            z = self(x)  # Forward pass to get predictions
            v_hat_dot_z = torch.einsum("bi, bi->b", v_hat, z)  # Calculate dot product

            Jv = grad(v_hat_dot_z.sum(), x, create_graph=True)[
                0
            ].detach()  # Compute Jacobian-vector product

            JF += (
                num_features * (Jv**2).sum() / (nproj * len(x))
            )  # Add square of Jv to Frobenius norm

        return JF

    def loss_fn(
        self,
        x,
        y,
    ):
        """
        Calculates the loss function (cross-entropy) with an optional
        Jacobian regularization term.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        y : torch.Tensor
            The true labels.

        Returns
        -------
        loss : torch.Tensor
            The calculated loss.
        jacobi_loss : torch.Tensor
            The calculated Jacobian regularization loss (if applicable).
        """
        y_pred = self(x.float())  # Forward pass to get predictions
        loss = self.L(y_pred, y)  # Compute loss using cross-entropy

        # Jacobi regularization
        if self.jacobi:
            x.requires_grad_(True)
            # Compute and add Jacobian regularization term
            jacobi_loss = self.jacobi_lmbd * self.jacobian_regularizer(x)
            loss += jacobi_loss
            return loss, jacobi_loss

        return loss, loss

    def train_step(
        self,
        data,
        labels,
    ):
        """
        Performs one step of training: forward pass, loss calculation,
        backward pass and parameters update.

        Parameters
        ----------
        data : torch.Tensor
            The input tensor.
        labels : torch.Tensor
            The true labels.

        Returns
        -------
        loss : float
            The calculated loss.
        reg_loss : float
            The calculated regularization loss (if applicable).
        """
        self.opt.zero_grad()  # Zero out gradients
        loss, reg_loss = self.loss_fn(
            data,
            labels.long(),
        )  # Compute loss
        loss.backward()  # Backward pass to compute gradients
        self.opt.step()  # Update parameters

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
