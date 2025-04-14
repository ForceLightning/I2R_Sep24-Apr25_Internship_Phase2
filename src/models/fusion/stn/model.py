# Standard Library

# Third-Party
from einops import repeat

# PyTorch
import torch
from torch import Tensor, nn
from torch.nn import functional as F

# Local folders
from .utils import SpatialTransformerType, init_transformer


# TODO: Include a way to have the motion data drive the transformation of the
# spatial image(s).
class SpatialTransformer(nn.Module):
    def __init__(self, in_channels: int, input_is_3d: bool = False) -> None:
        super().__init__()
        # Spatial transformer localisation-network.
        if input_is_3d:
            self.localisation = nn.Sequential(
                nn.Conv3d(in_channels, 8, kernel_size=(7, 7, 1)),
                nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1)),
                nn.ReLU(True),
                nn.Conv3d(8, 10, kernel_size=(5, 5, 1)),
                nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1)),
                nn.ReLU(True),
            )
        else:
            self.localisation = nn.Sequential(
                nn.Conv2d(in_channels, 8, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
            )

        # Regressor for the 3 * 2 affine matrix
        # NOTE: We need to know the output shape of the image embeddings in advance.
        affine_matrix_regressor = nn.Linear(32, 3 * 2)
        affine_matrix_regressor.weight.data.zero_()
        affine_matrix_regressor.bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), affine_matrix_regressor
        )

        # Initialise the weights/bias with the identity transformation

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the spatial transformer.

        Args:
            x: Image tensor of shape: (N, C, H, W)

        Return:
            Tensor: Image tensor of shape (N, C, H, W)

        """
        # Transform the input
        xs = self.localisation(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, list(x.size()))
        x = F.grid_sample(x, grid)

        return x


class STN(nn.Module):
    """Base class for a Spatial Transformer Network (STN).

    Adapted from: https://github.com/FrederikWarburg/pSTN-baselines. Implement this for
    specific tasks, which requires the method `init_localiser` to be implemented.
    """

    def __init__(
        self,
        transformer_type: SpatialTransformerType,
        num_param: int,
        N: int,
        xdim: int | None = None,
        **kwargs,
    ):
        """Initialise the STN.

        Args:
            transformer_type: Spatial transformer type to use.
            num_param: If we use an affine (s, r, tx, ty) or crop (0.5, 1, tx, ty) transformation.
            N: Number of parallel tracks.
            xdim: Indicator of time series datasets. 1 if timeseries, otherwise 2.

        """
        super().__init__(**kwargs)

        # Hyperparameters
        self.transformer_type = transformer_type
        self.num_param = num_param
        self.N = N
        self.xdim = xdim
        self.transformer, self.theta_dim = init_transformer(
            transformer_type, N, num_param, xdim
        )

        # Spatial transformer localisation network
        self.fc_loc: nn.Sequential
        self.localisation: nn.Module
        self.init_localiser(**kwargs)

        # Initialise the model weights and bias of the regressors
        self.init_model_weights(**kwargs)

    def init_localiser(self, **kwargs):
        """Initialise task-specific localiser."""
        raise NotImplementedError("Implement this method in subclasses.")

    def init_model_weights(self, **kwargs):
        """Initialise model weights."""
        self.fc_loc[-1].weight.data.zero_()  # pyright: ignore

        # Initialise the weights/bias with identity transformation.
        if (
            kwargs.get("transformer_type", self.transformer_type)
            == SpatialTransformerType.AFFINE
        ):
            if self.num_param == 2:
                if self.N == 1:
                    bias = torch.tensor([0, 0], dtype=torch.float)
                else:
                    # Initialise the bounding boxes with tiling.
                    bias = (
                        torch.tensor(
                            [[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=torch.float
                        )
                        * 0.5
                    )
                self.fc_loc[-1].bias.data.copy_(
                    bias[: self.N].view(-1)
                )  # pyright: ignore
            elif self.num_param == 3:
                self.fc_loc[-1].bias.data.copy_(
                    torch.tensor(
                        [1, 0, 0] * self.N, dtype=torch.float
                    )  # pyright: ignore
                )
            elif self.num_param == 4:
                self.fc_loc[-1].bias.data.copy_(
                    torch.tensor(
                        [0, 0.5, 0, 0] * self.N, dtype=torch.float
                    )  # pyright: ignore
                )
            elif self.num_param == 5:
                self.fc_loc[-1].bias.data.copy_(
                    torch.tensor(
                        [0, 1, 1, 0, 0] * self.N, dtype=torch.float
                    )  # pyright: ignore
                )
            elif self.num_param == 6:
                self.fc_loc[-1].bias.data.copy_(
                    torch.tensor(
                        [1, 0, 0, 0, 1, 0] * self.N, dtype=torch.float
                    )  # pyright: ignore
                )

        elif (
            kwargs.get("transformer_type", self.transformer_type)
            == SpatialTransformerType.DIFFEOMORPHIC
        ):
            # Initialise the params as identity, default ok for variance in thise case.
            self.fc_loc[-1].bias.data.copy_(
                torch.tensor([1e-5], dtype=torch.float).repeat(self.theta_dim)
            )  # pyright: ignore

    def forward(self, x: Tensor, x_high_res: Tensor | None = None):
        # Zoom in on the relevant areas with stn
        x, theta = self.forward_localiser(x, x_high_res)
        return x, theta

    def forward_localiser(self, x: Tensor, x_high_res: Tensor | None = None):
        if x_high_res is None:
            x_high_res = x

        b, _c, h, w = x.shape
        theta = self.compute_theta(x)

        # Repeat x in the batch dim so we avoid a for-loop.
        x = repeat(x, "b c h w -> (b n) c h w", n=self.N)
        theta_upsample = theta.view(b * self.N, self.theta_dim)

        x = self.transformer(x_high_res, theta_upsample, small_image_shape=(h, w))
        return x, theta

    def compute_theta(self, x: Tensor):
        batch_size = x.shape[0]
        xs = self.localisation(x)
        xs = xs.view(batch_size, -1)
        theta = self.fc_loc(xs)
        return theta
