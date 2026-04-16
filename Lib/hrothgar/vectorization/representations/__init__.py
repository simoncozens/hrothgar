from abc import ABC
from typing import List, Optional, Self, Sequence, Union, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch

from hrothgar.vectorization.nodeglyph import Node, NodeContour

MAX_COORDINATE = 1000

# Implementation node: There's an important notational convention we're using
# in the relative coordinate representations. We want all node commands to use
# relative coordinates. *All* of them, so that the model doesn't have to do
# gymnastics to understand that node 0's coordinates have a different interpretation
# to all the other nodes in the sequence. But then how do you establish initial
# position? The convention is that the first node in a contour gets emitted as
# two separate commands. First, an absolute "Move to" command to set the
# initial position, then the first node itself is emitted as a relative
# command with a delta of (0,0).


# Dammit Python
class classproperty:
    def __init__(self, func):
        self.fget = func

    def __get__(self, instance, owner):
        return self.fget(owner)


class CoordinateRepresentation(ABC):
    @classmethod
    def emit_node_position(cls, n: "Node") -> npt.NDArray[np.float32]: ...

    @classmethod
    def emit_in_handle(cls, n: "Node") -> Optional[npt.NDArray[np.float32]]: ...

    @classmethod
    def emit_out_handle(cls, n: "Node") -> Optional[npt.NDArray[np.float32]]: ...


class AbsoluteCoordinateRepresentation(CoordinateRepresentation):
    @classmethod
    def emit_node_position(cls, n: "Node") -> npt.NDArray[np.float32]:
        return n.coordinates

    @classmethod
    def emit_in_handle(cls, n: "Node") -> Optional[npt.NDArray[np.float32]]:
        return n.in_handle

    @classmethod
    def emit_out_handle(cls, n: "Node") -> Optional[npt.NDArray[np.float32]]:
        return n.out_handle


class AbsolutePositionRelativeHandleRepresentation(CoordinateRepresentation):
    @classmethod
    def emit_node_position(cls, n: "Node") -> npt.NDArray[np.float32]:
        return n.coordinates

    @classmethod
    def emit_in_handle(cls, n: "Node") -> Optional[npt.NDArray[np.float32]]:
        if n.in_handle is None:
            return None
        return n.in_handle - n.coordinates

    @classmethod
    def emit_out_handle(cls, n: "Node") -> Optional[npt.NDArray[np.float32]]:
        if n.out_handle is None:
            return None
        return n.out_handle - n.coordinates


class RelativeCoordinateRepresentation(AbsolutePositionRelativeHandleRepresentation):
    """Handles are also relative to the node position."""

    @classmethod
    def emit_node_position(cls, n: "Node") -> npt.NDArray[np.float32]:
        if n.index == 0:
            return n.coordinates
        previous_node = n.previous
        return n.coordinates - previous_node.coordinates


class RelativePolarCoordinateRepresentation(CoordinateRepresentation):
    """Coordinates are represented in polar form relative to the previous node.
    Handles are also relative to the current node position, in polar form."""

    @classmethod
    def emit_node_position(cls, n: "Node") -> npt.NDArray[np.float32]:
        if n.index == 0:
            # Definitionally this is 0,0 because we emit an explicit move to absolute position first
            return np.array([0.0, 0.0], dtype=np.float32)
        previous_node = n.previous
        delta = n.coordinates - previous_node.coordinates
        r = np.linalg.norm(delta)
        theta = np.arctan2(delta[1], delta[0])
        return np.array([r, theta], dtype=np.float32)

    @classmethod
    def emit_in_handle(cls, n: "Node") -> Optional[npt.NDArray[np.float32]]:
        if n.in_handle is None:
            return None
        delta = n.in_handle - n.coordinates
        r = np.linalg.norm(delta)
        theta = np.arctan2(delta[1], delta[0])
        return np.array([r, theta], dtype=np.float32)

    @classmethod
    def emit_out_handle(cls, n: "Node") -> Optional[npt.NDArray[np.float32]]:
        if n.out_handle is None:
            return None
        delta = n.out_handle - n.coordinates
        r = np.linalg.norm(delta)
        theta = np.arctan2(delta[1], delta[0])
        return np.array([r, theta], dtype=np.float32)


class CommandRepresentation(ABC):
    grammar: dict[str, int]
    command: str
    coordinates: List[Union[int, float]]
    coordinate_representation: type[
        "CoordinateRepresentation"
    ]  # How would you like your coordinates?

    @classmethod
    def emit(cls, nodes: List["Node"]) -> Sequence[Self]: ...

    @classmethod
    def contour_from_commands(cls, commands: Sequence[Self]) -> "NodeContour":
        raise NotImplementedError()

    @classproperty
    def command_width(cls) -> int:
        return len(cls.grammar.keys())

    @classproperty
    def coordinate_width(cls) -> int:
        return max(cls.grammar.values())

    @classmethod
    def encode_command(cls, s: str) -> int:
        return list(cls.grammar.keys()).index(s)

    @classmethod
    def encode_command_one_hot(cls, s: str) -> torch.Tensor:
        index = cls.encode_command(s)
        one_hot = torch.zeros(cls.command_width, dtype=torch.float32)
        one_hot[index] = 1.0
        return one_hot

    @classmethod
    def decode_command(cls, index: int) -> str:
        return list(cls.grammar.keys())[index]

    @classmethod
    def decode_command_one_hot(cls, one_hot: Union[torch.Tensor, npt.NDArray]) -> str:
        if isinstance(one_hot, np.ndarray):
            index = int(np.argmax(one_hot))
        else:
            index = int(torch.argmax(one_hot).item())
        return cls.decode_command(index)

    @classmethod
    def split_tensor(cls, tensor: torch.Tensor) -> Sequence[torch.Tensor]:
        """Splits a tensor of shape (N, command_width + coordinate_width)
        or (B, N, command_width + coordinate_width) into a tuple of
        (commands, coordinates) tensors."""
        command_width = cls.command_width
        dim = tensor.dim() if isinstance(tensor, torch.Tensor) else len(tensor.shape)
        if dim == 2:
            commands = tensor[:, :command_width]
            coordinates = tensor[:, command_width:]
        elif dim == 3:
            commands = tensor[:, :, :command_width]
            coordinates = tensor[:, :, command_width:]
        else:
            raise ValueError(f"Unsupported tensor dimension: {dim}")
        return commands, coordinates

    def __init__(self, command: str, coordinates: List[Union[int, float]]):
        if command not in self.grammar:
            raise ValueError(f"Invalid command: {command}")
        if len(coordinates) != self.grammar[command]:
            raise ValueError(
                f"Invalid number of coordinates for command {command}: expected {self.grammar[command]}, got {len(coordinates)}"
            )
        self.command = command
        self.coordinates = coordinates

    def debug_string(self) -> str:
        return f"{self.command} {' '.join([f"{c:.2f}" for c in self.coordinates])}"
