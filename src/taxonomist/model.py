import lightning.pytorch as pl
import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score


def mse(output, target):
    output = output.flatten()
    loss = F.mse_loss(output, target)
    return loss


def mape(output, target):
    output = output.flatten()
    loss = torch.mean((torch.abs(output - target) / target.abs()))
    return loss


def l1(output, target):
    output = output.flatten()
    loss = F.l1_loss(output, target)
    return loss


def cross_entropy(output, target):
    loss = F.cross_entropy(output, target.long())
    return loss


#alphas for Focal loss
def calculate_alpha(class_counts, num_classes):
    # Calculate alpha values based on class_counts here
    # For example, inverse frequency or any other method
    inverse_frequency = [1.0 / count for count in class_counts]
    # Normalize if necessary
    sum_inv_freq = sum(inverse_frequency)
    alpha = [inv_freq / sum_inv_freq for inv_freq in inverse_frequency]
    # Ensure the length matches the number of classes
    if len(alpha) != num_classes:
        raise ValueError("Length of alpha does not match number of classes")
    return alpha

class FocalLoss(torch.nn.Module):
    def __init__(self, class_counts, gamma=2.0, reduction='mean', num_classes=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if num_classes is None:
            raise ValueError("num_classes must be set when alpha is not provided")
        
        alpha = calculate_alpha(class_counts, num_classes=num_classes)
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha)
        
        if hasattr(self, 'alpha'):
            # Update the alpha buffer if it already exists
            self.alpha = alpha
        else:
            # Register alpha as a buffer if it doesn't exist
            self.register_buffer('alpha', alpha)

    def forward(self, inputs, targets):
        # Ensure targets are of the correct type for indexing
        targets = targets.long()
        
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        
        # Now targets is guaranteed to be of a type suitable for indexing
        alpha_t = self.alpha[targets]
        print("The value of alpha inside focal loss is: ", alpha_t)
        loss = alpha_t * ((1 - pt) ** self.gamma) * CE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# # Define the custom loss function
# class CILoss(torch.nn.Module):
#     def __init__(self, class_counts, k=1.0, theta=0.5, device='cuda'):
#         super(CILoss, self).__init__()
#         self.k = k
#         self.theta = theta
#         self.device = device
#         N_max = max(class_counts)
#         # Convert class_counts to a tensor
#         class_counts_tensor = torch.tensor(class_counts, dtype=torch.float).to(device)
#         # Corrected weights calculation
#         #self.weights = torch.tensor([torch.log((N_max / N_j) + theta) for N_j in class_counts], dtype=torch.float).to(device)
#         self.weights = torch.log((N_max / class_counts_tensor) + theta)

#     def forward(self, inputs, targets):
#         # Ensure targets is of type torch.long for indexing
#         targets = targets.long()
#         # Convert inputs to softmax probabilities
#         probs = F.softmax(inputs, dim=1)
#         # Sort weights according to the target order
#         sorted_weights = self.weights[targets]
#         # Calculate the log of probabilities
#         log_probs = torch.log(probs)
#         # Gather the log probabilities for each target class
#         log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
#         # Calculate the exponent term
#         exp_term = torch.exp(-self.k * probs.gather(1, targets.unsqueeze(1)).squeeze(1))
#         # Calculate the CI loss
#         loss = -sorted_weights * exp_term * log_probs
#         return loss.mean()

# class CILoss(torch.nn.Module):
#     def __init__(self, class_counts, k=1.0, theta=0.5, device='cuda'):
#         super(CILoss, self).__init__()
#         self.k = k
#         self.theta = theta
#         self.device = device
#         N_max = max(class_counts)
#         self.class_counts_tensor = torch.tensor(class_counts, dtype=torch.float).to(device)  # Calculate class_counts_tensor
#         self.weights = torch.log((N_max / self.class_counts_tensor) + theta)

#     def forward(self, inputs, targets):
#         targets = targets.long()
#         probs = F.softmax(inputs, dim=1)  # Probabilities for all classes
        
#         loss = 0.0
#         for i in range(len(targets)):  # Loop over samples
#             for j in range(probs.shape[1]):  # Loop over classes
#                 exp_term = torch.exp(-self.k * probs[i, j] * self.class_counts_tensor[j]) 
#                 loss += -self.weights[j] * exp_term * torch.log(probs[i, j]) 
        
#         return loss.mean() 


class CILoss(torch.nn.Module):
    def __init__(self, class_counts, k=1.0, theta=0.5, device='cuda'):
        super().__init__()
        self.k = k
        self.theta = theta
        self.device = device

        N_max = max(class_counts)
        self.class_counts_tensor = torch.tensor(class_counts, dtype=torch.float).to(device)
        self.weights = torch.log(N_max / self.class_counts_tensor + theta)

    def forward(self, inputs, targets):
        targets = targets.long()

        probs = F.softmax(inputs, dim=1)

        # Get the weights corresponding to the target classes
        weights = self.weights[targets] 

        # Calculate log probabilities directly for numerical stability
        log_probs = F.log_softmax(inputs, dim=1) 
        log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # The exponential term from the paper
        exp_term = torch.exp(-self.k * probs.gather(1, targets.unsqueeze(1)).squeeze(1))

        loss = -weights * exp_term * log_probs  # Removing the sorting step
        return loss.mean()





# class CrossEntropyImbalanceLoss(nn.Module):
#     def __init__(self, class_counts, total_epochs, k=1.0, theta=0.5, device='cuda'):
#         super(CrossEntropyImbalanceLoss, self).__init__()
#         self.cross_entropy = cross_entropy
#         self.ci_loss = CILoss(class_counts, k, theta, device)
#         self.total_epochs = total_epochs

#     def forward(self, inputs, targets, current_epoch):
#         # Calculate weights
#         alpha = 1 - (current_epoch / self.total_epochs)
#         beta = current_epoch / self.total_epochs
#         # Compute losses
#         ce_loss = self.cross_entropy(inputs, targets)
#         ci_loss = self.ci_loss(inputs, targets)
#         # Combine losses
#         loss = alpha * ce_loss + beta * ci_loss
#         return loss

class CrossEntropyImbalanceLoss(nn.Module):
    def __init__(self, class_counts, total_epochs, k=1.0, theta=0.5, device='cuda'):
        super(CrossEntropyImbalanceLoss, self).__init__()
        self.ci_loss = CILoss(class_counts, k, theta, device)
        self.total_epochs = total_epochs

    def forward(self, inputs, targets, current_epoch):
        # Calculate weights
        alpha = 1 - (current_epoch / self.total_epochs)
        beta = current_epoch / self.total_epochs
        
        # Ensure alpha and beta are within [0, 1]
        alpha = max(0.0, min(1.0, alpha))
        beta = max(0.0, min(1.0, beta))

        # Compute losses
        ce_loss = F.cross_entropy(inputs, targets)
        ci_loss = self.ci_loss(inputs, targets)

        # Combine losses
        loss = alpha * ce_loss + beta * ci_loss
        
        return loss





def choose_criterion(name, class_counts=None, n_classes=None, k=1.0, gamma=2.0, theta=0.5, device='cuda', total_epochs=None):
    if name == "cross-entropy":
        return cross_entropy
    elif name == "mse":
        return mse
    elif name == "mape":
        return mape
    elif name == "l1":
        return l1
    elif name == "class-imbalance":
        if class_counts is None:
            raise ValueError("class_counts must be provided for class-imbalance loss")
        return CILoss(class_counts, k=k, theta=theta, device=device)
    elif name == "focal":
        # Note: Adjust alpha and gamma as per your requirement
        return FocalLoss(class_counts, num_classes=n_classes, gamma=gamma)
    elif name == "cross-entropy-imbalance":
        if total_epochs is None:
            raise ValueError("total_epochs must be provided for cross-entropy-imbalance loss")
        return CrossEntropyImbalanceLoss(class_counts, total_epochs, k=k, theta=theta, device=device)
    else:
        raise Exception(f"Invalid criterion name '{name}'")

class Model(nn.Module):
    """PyTorch module for an arbitary timm model, separating the base and projection head"""

    def __init__(
        self,
        model: str = "resnet18",
        freeze_base: bool = True,
        pretrained: bool = True,
        n_classes: int = 1,
    ):
        """Initializes the model

        Args:
            model (str): name of the model to use
            freeze_base (bool): if True, the base is frozen
            pretrained (bool): if True, use pretrained weights
            n_classes (int): output layer size
        """
        super().__init__()

        self.h_dim = (
            timm.create_model(model, pretrained=False, num_classes=1)
            .get_classifier()
            .in_features
        )
        self.base_model = timm.create_model(model, num_classes=0, pretrained=pretrained)

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.proj_head = nn.Sequential(nn.Linear(self.h_dim, n_classes))

    def forward(self, x):
        h = self.base_model(x)
        return self.proj_head(h)

    def base_forward(self, x):
        return self.base_model(x)

    def proj_forward(self, h):
        return self.proj_head(h)


class LitModule(pl.LightningModule):
    """PyTorch Lightning module for training an arbitary model"""

    def __init__(
        self,
        model: str,
        freeze_base: bool = False,
        pretrained: bool = True,
        n_classes: int = 1,
        criterion: str = "mse",
        opt: dict = {"name": "adam"},
        lr: float = 1e-4,
        lr_scheduler: dict = None,
        label_transform=None,
        class_counts=None,  # Add this parameter
        k=1.0,
        gamma=2.0,         # Add this parameter
        theta=0.5,         # Add this parameter
        device='cuda',     # Add this parameter
        total_epochs=None # Add this parameter
    ):
        """Initialize the module
        Args:
            model (str): name of the ResNet model to use

            freeze_base (bool): whether to freeze the base model

            pretrained (bool): whether to use pretrained weights

            n_classes (int): number of outputs. Set 1 for regression

            criterion (str): loss function to use

            lr (float): learning rate

            label_transform: possible transform that is done for the output labels
        """
        super().__init__()
        self.save_hyperparameters(ignore=["label_transform"])
        self.example_input_array = torch.randn((1, 3, 224, 224))
        self.model = Model(
            model=model,
            freeze_base=freeze_base,
            pretrained=pretrained,
            n_classes=n_classes,
        )
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.label_transform = label_transform
        self.criterion = choose_criterion(criterion, class_counts, n_classes, k,  gamma, theta, device, total_epochs)
        self.opt_args = opt
        self.epoch_counter = 0 # Initialize epoch counter

        if criterion in ["cross-entropy", "class-imbalance", "focal", "cross-entropy-imbalance"]:
            self.is_classifier = True
        else:
            self.is_classifier = False

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.batch_size = None

    def predict_func(self, output):
        """Processes the output for prediction"""
        if self.is_classifier:
            return output.argmax(dim=1)
        else:
            return output.flatten()

    def forward(self, x):
        """Forward pass"""
        return self.model(x)

    def configure_optimizers(self):
        """Sets optimizers based on a dict passed as argument"""
        if self.opt_args["name"] == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        elif self.opt_args["name"] == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), self.lr)
        else:
            raise Exception("Invalid optimizer")

        if self.lr_scheduler is not None:
            if self.lr_scheduler["name"] != "CosineAnnealingLR":
                raise Exception("Invalid learning rate schedule")
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.lr_scheduler["T_max"]
            )
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer

    def common_step(self, batch, batch_idx):
        """
        Perform a common processing step during training, validation, or testing.
        """
        x = batch["x"]
        y = batch["y"]
        fname = batch["fname"]
        out = self.model(x)
        if self.hparams.criterion == "cross-entropy-imbalance":
            loss = self.criterion(out, y, self.epoch_counter)  # Pass epoch counter for dynamic weighting
        else:
            loss = self.criterion(out, y)
        #loss = self.criterion(out, y)

        # Set the batch size for logging
        if self.batch_size is None:
            self.batch_size = len(x)
        return x, y, fname, out, loss

    def common_epoch_end(self, outputs, name: str):
        """Combine outputs for calculating metrics at the end of an epoch."""
        y_true = torch.cat([x["y_true"] for x in outputs]).cpu().detach().numpy()
        y_pred = torch.cat([x["y_pred"] for x in outputs]).cpu().detach().numpy()

        if self.label_transform:
            y_true = self.label_transform(y_true)
            y_pred = self.label_transform(y_pred)

        if self.is_classifier:
            self.log(f"{name}/acc", accuracy_score(y_true, y_pred))
            self.log(
                f"{name}/f1",
                f1_score(y_true, y_pred, average="weighted", zero_division=0),
            )

        return y_true, y_pred

    # Training
    def training_step(self, batch, batch_idx):
        """
        Execute a training step and log relevant information.
        """
        _, y, _, out, loss = self.common_step(batch, batch_idx)
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, batch_size=self.batch_size
        )
        outputs = {"loss": loss, "y_true": y, "y_pred": self.predict_func(out)}
        self.training_step_outputs.append(outputs)
        return loss

    def on_train_epoch_end(self):
        """
        Perform actions at the end of each training epoch.

        - Calls common_epoch_end method for additional processing.
        - Clears the list of training step outputs.
        """
        outputs = self.training_step_outputs
        _, _ = self.common_epoch_end(outputs, "train")
        self.training_step_outputs.clear()
        self.epoch_counter += 1 # Increment epoch counter
        self.log(
            "epoch/current",
            self.epoch_counter,
            on_epoch=True,
            batch_size=self.batch_size,
        )


    # Validation
    def validation_step(self, batch, batch_idx):
        """
        Execute a validation step and log relevant information.
        """
        _, y, _, out, val_loss = self.common_step(batch, batch_idx)
        self.log(
            "val/loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        outputs = {"y_true": y, "y_pred": self.predict_func(out)}
        self.validation_step_outputs.append(outputs)

    def on_validation_epoch_end(self):
        """
        Perform actions at the end of each validation epoch.

        - Calls common_epoch_end method for additional processing.
        - Clears the list of validation step outputs.
        """
        outputs = self.validation_step_outputs
        _, _ = self.common_epoch_end(outputs, "val")
        self.validation_step_outputs.clear()

    # Testing
    def test_step(self, batch, batch_idx):
        """Execute a testing step and log relevant information."""
        _, y, fname, out, test_loss = self.common_step(batch, batch_idx)
        self.log(
            "test/loss",
            test_loss,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        outputs = {
            "y_true": y,
            "y_pred": self.predict_func(out),
            "fname": fname,
            "out": out,
        }
        self.test_step_outputs.append(outputs)

    def on_test_epoch_end(self):
        """
        Perform actions at the end of each testing epoch.

        - Extracts outputs from testing steps.
        - If the model is a classifier, computes softmax probabilities and stores them.
        - Calls common_epoch_end method for additional processing.
        """
        outputs = self.test_step_outputs
        xss = [x["fname"] for x in outputs]
        self.fnames = np.array([x for xs in xss for x in xs])
        if self.is_classifier:
            logits = torch.cat([x["out"] for x in outputs])
            self.softmax = logits.softmax(dim=1).cpu().detach().numpy()
            self.logits = logits.cpu().detach().numpy()
        self.y_true, self.y_pred = self.common_epoch_end(outputs, "test")


class FeatureExtractionModule(pl.LightningModule):
    def __init__(
        self,
        feature_extraction_mode: str,
        model: str,
        freeze_base: bool = False,
        pretrained: bool = True,
        n_classes: int = 0,
        criterion: str = "cross-entropy",
        opt: dict = {"name": "adam"},
        lr: float = 1e-4,
        lr_scheduler: dict = None,
        label_transform=None,
    ):
        """
        The feature exctraction module implements the same interface as the basic LitModule
        for passing LitModule parameters.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["label_transform"])
        self.example_input_array = torch.randn((1, 3, 224, 224))

        self.feature_extraction_mode = feature_extraction_mode
        self.model = Model(
            model=model,
            freeze_base=freeze_base,
            pretrained=pretrained,
            n_classes=n_classes,
        )
        self.lr = lr
        self.label_transform = label_transform
        self.criterion = choose_criterion(criterion)
        self.opt_args = opt

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        if self.feature_extraction_mode == "unpooled":
            return self.model.base_model.forward_features(x)
        elif self.feature_extraction_mode == "pooled":
            return self.model.base_forward(x)
        else:
            raise Exception(
                f"Invalid feature extraction mode {self.feature_extraction_mode}"
            )

    def test_step(self, batch, batch_idx):
        bx = batch["x"]
        by = batch["y"]
        fname = batch["fname"]
        out = self.forward(bx)
        outputs = {
            "fname": fname,
            "y_true": by.cpu().detach().numpy(),
            "out": out.cpu().detach().numpy(),
        }
        self.test_step_outputs.append(outputs)

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        self.y_true = [x["y_true"] for x in outputs]
        self.y_pred = [x["out"] for x in outputs]
        self.fnames = [x["fname"] for x in outputs]
