import torch
import torch.nn as nn
from torch.autograd import grad


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
    jacobi_reg : bool
        If True, applies Jacobian regularization.
    jacobi_reg_lmbd : float
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
        self.jacobi_reg = jacobi_reg
        self.jacobi_reg_lmbd = jacobi_reg_lmbd
        self.svb_reg = svb_reg
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
        JF = 0 # Initialize Jacobian Frobenius norm
        nproj = 1  # Number of random projections

        for _ in range(nproj):
            v = torch.randn(x.shape[0], C).to(x.device)  # Generate random vector
            v_hat = v / torch.norm(v, dim=1, keepdim=True)  # Normalize

            z = self(x)  # Forward pass to get predictions
            v_hat_dot_z = torch.einsum("bi, bi->b", v_hat, z)  # Calculate dot product

            Jv = grad(v_hat_dot_z.sum(), x, create_graph=True)[
                0
            ].detach()  # Compute Jacobian-vector product

            JF += C * (Jv**2).sum() / (nproj * len(x)) # Add square of Jv to Frobenius norm

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
        y_pred = self(x.float()) # Forward pass to get predictions
        loss = self.L(y_pred, y) # Compute loss using cross-entropy

        # Jacobi regularization
        if self.jacobi_reg:
            x.requires_grad_(True)
            # Compute and add Jacobian regularization term
            jacobi_loss = self.jacobi_reg_lmbd * self.jacobian_regularizer(x)
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
        self.opt.zero_grad()
        loss, reg_loss = self.loss_fn(
            data,
            labels.long(),
        )
        loss.backward()
        self.opt.step()

        if self.svb_reg and self.training_steps % self.svb_freq == 0:
            with torch.no_grad():
                for m in self.modules():
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                        weight_orig_shape = m.weight.shape
                        weight_matrix = m.weight.view(weight_orig_shape[0], -1)
                        U, S, V = torch.svd(weight_matrix)
                        S = torch.clamp(S, 1 / (1 + self.svb_eps), 1 + self.svb_eps)
                        m.weight.data = torch.matmul(
                            U, torch.matmul(S.diag(), V.t())
                        ).view(weight_orig_shape)

        self.training_steps += 1
        return loss.item(), reg_loss.item() if reg_loss != 0 else reg_loss
