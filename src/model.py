"""
model.py
--------
Neural network architectures used as function approximators for the RL agent.

Defines:
  - ``MLP``            – plain multi-layer perceptron backbone
  - ``DuelingMLP``     – dueling advantage/value heads on top of MLP
  - ``build_network``  – factory that selects the correct architecture from
                         a ``ModelConfig`` instance

No learning or optimisation logic lives here; those belong in ``train.py``.
"""

from __future__ import annotations

from typing import Sequence

from config import ModelConfig


class MLP:
    """
    Multi-layer perceptron with configurable hidden layers.

    Parameters
    ----------
    input_dim:
        Dimensionality of the flattened observation vector.
    output_dim:
        Number of output units (typically the action-space size for a Q-net).
    hidden_sizes:
        Number of units in each hidden layer.
    activation:
        Non-linearity applied after every hidden layer.  Supported values:
        ``"relu"``, ``"tanh"``, ``"elu"``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: str = "relu",
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = list(hidden_sizes)
        self.activation = activation
        # Network layers will be constructed here during implementation.
        self._layers: list = []

    def forward(self, x: object) -> object:
        """
        Forward pass through the network.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch_size, input_dim)``.

        Returns
        -------
        object
            Output tensor of shape ``(batch_size, output_dim)``.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"MLP(input={self.input_dim}, hidden={self.hidden_sizes}, "
            f"output={self.output_dim}, activation={self.activation})"
        )


class DuelingMLP(MLP):
    """
    Dueling network architecture (Wang et al., 2016).

    Splits the network into two streams after the shared backbone:
      - **Value stream** V(s)       – scalar estimate of state value
      - **Advantage stream** A(s,a) – relative advantage of each action

    The Q-value is recovered as ``Q(s,a) = V(s) + A(s,a) - mean(A(s,·))``.

    Inherits all parameters from ``MLP``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__(input_dim, output_dim, hidden_sizes, activation)
        # Value and advantage head layers will be added here.
        self._value_head: object = None
        self._advantage_head: object = None

    def forward(self, x: object) -> object:
        """
        Forward pass with dueling aggregation.

        Returns
        -------
        object
            Q-value tensor of shape ``(batch_size, output_dim)``.
        """
        raise NotImplementedError


def build_network(
    input_dim: int,
    output_dim: int,
    cfg: ModelConfig,
) -> MLP:
    """
    Factory function – constructs the correct network from ``ModelConfig``.

    Parameters
    ----------
    input_dim:
        Flattened observation size.
    output_dim:
        Number of discrete actions.
    cfg:
        Model configuration (selects dueling vs. plain MLP, hidden sizes, etc.)

    Returns
    -------
    MLP
        Constructed (but not yet trained) network instance.
    """
    cls = DuelingMLP if cfg.dueling else MLP
    return cls(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
    )
