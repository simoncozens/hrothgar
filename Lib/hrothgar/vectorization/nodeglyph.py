from typing import TYPE_CHECKING, List, Optional, Sequence

if TYPE_CHECKING:
    from .representations import CommandRepresentation

import numpy as np
import numpy.typing as npt

MAX_SEQUENCE_LENGTH = 50


class Node:
    """A node, such as you would find in a vector design tool.

    It stores its position, and optional in/out handles as *absolute* coordinates.
    It knows where it is in a contour and can access its neighbors.
    Conversion to particular command and coordinate systems is handled elsewhere.
    """

    coordinates: npt.NDArray[np.float32]
    in_handle: Optional[npt.NDArray[np.float32]]
    out_handle: Optional[npt.NDArray[np.float32]]
    _contour: "NodeContour"
    ALIGNMENT_EPSILON = 0.1
    _index: Optional[int] = None

    def __init__(self, coordinates, contour, in_handle=None, out_handle=None):
        self.coordinates = np.array(coordinates)
        self.in_handle = np.array(in_handle) if in_handle is not None else None
        self.out_handle = np.array(out_handle) if out_handle is not None else None
        self._contour = contour

    @property
    def index(self) -> int:
        if self._index is None:
            self._contour.index_your_nodes()
            assert self._index is not None, "Node index should have been set by contour"
        return self._index

    @property
    def next(self) -> "Node":
        return self._contour.nodes[(self.index + 1) % len(self._contour.nodes)]

    @property
    def previous(self) -> "Node":
        return self._contour.nodes[self.index - 1]

    @property
    def is_line(self) -> bool:
        return self.in_handle is None and self.out_handle is None

    @property
    def is_horizontal_line(self) -> bool:
        return (
            self.is_line
            and self.index > 0
            and np.isclose(
                self.previous.coordinates[1],
                self.coordinates[1],
                atol=self.ALIGNMENT_EPSILON,
            )
        )

    @property
    def is_vertical_line(self) -> bool:
        return (
            self.is_line
            and self.index > 0
            and np.isclose(
                self.previous.coordinates[0],
                self.coordinates[0],
                atol=self.ALIGNMENT_EPSILON,
            )
        )

    @property
    def handles_horizontal(self) -> bool:
        return (
            self.in_handle is not None
            and self.out_handle is not None
            and abs(self.in_handle[1] - self.coordinates[1]) <= self.ALIGNMENT_EPSILON
            and abs(self.coordinates[1] - self.out_handle[1]) <= self.ALIGNMENT_EPSILON
        )

    @property
    def handles_vertical(self) -> bool:
        return (
            self.in_handle is not None
            and self.out_handle is not None
            and abs(self.in_handle[0] - self.coordinates[0]) <= self.ALIGNMENT_EPSILON
            and abs(self.coordinates[0] - self.out_handle[0]) <= self.ALIGNMENT_EPSILON
        )

    @property
    def is_smooth(self) -> bool:
        if self.in_handle is None or self.out_handle is None:
            return False
        vec_in = self.coordinates - self.in_handle
        vec_out = self.out_handle - self.coordinates
        norm_in = np.linalg.norm(vec_in)
        norm_out = np.linalg.norm(vec_out)
        if norm_in == 0 or norm_out == 0:
            return False
        unit_in = vec_in / norm_in
        unit_out = vec_out / norm_out
        dot_product = np.dot(unit_in, unit_out)
        return np.isclose(dot_product, 1.0, atol=1e-2)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented

        coords_equal = np.allclose(self.coordinates, other.coordinates, atol=1e-6)

        in_handles_equal = (self.in_handle is None and other.in_handle is None) or (
            self.in_handle is not None
            and other.in_handle is not None
            and np.allclose(self.in_handle, other.in_handle, atol=1e-6)
        )

        out_handles_equal = (self.out_handle is None and other.out_handle is None) or (
            self.out_handle is not None
            and other.out_handle is not None
            and np.allclose(self.out_handle, other.out_handle, atol=1e-6)
        )

        if not (coords_equal and in_handles_equal and out_handles_equal):
            # print(f"Node mismatch:")
            # print(
            #     f"  Coords: {self.coordinates} vs {other.coordinates} (Equal: {coords_equal})"
            # )
            # print(
            #     f"  In Handles: {self.in_handle} vs {other.in_handle} (Equal: {in_handles_equal})"
            # )
            # print(
            #     f"  Out Handles: {self.out_handle} vs {other.out_handle} (Equal: {out_handles_equal})"
            # )
            return False

        return True


class NodeContour:
    nodes: List["Node"]

    def __init__(self, nodes: List["Node"]):
        self.nodes = nodes
        # Adopt all the nodes in this list
        for node in self.nodes:
            node._contour = self

    def index_your_nodes(self):
        for i, node in enumerate(self.nodes):
            node._index = i

    def normalize(self) -> None:
        if not self.nodes:
            return

        # Ensure the contour is clockwise
        if not self.is_clockwise():
            self.reverse_direction()

        index_of_bottom_left = min(
            range(len(self.nodes)),
            key=lambda i: (self.nodes[i].coordinates[1], self.nodes[i].coordinates[0]),
        )
        self.nodes = (
            self.nodes[index_of_bottom_left:] + self.nodes[:index_of_bottom_left]
        )

    def __eq__(self, other):
        if not isinstance(other, NodeContour):
            return NotImplemented
        result = len(self.nodes) == len(other.nodes) and all(
            n1 == n2 for n1, n2 in zip(self.nodes, other.nodes)
        )
        return result

    def is_clockwise(self) -> bool:
        """
        Determines if the contour is clockwise using the shoelace formula.
        A positive area indicates counter-clockwise, negative indicates clockwise.
        """
        if not self.nodes or len(self.nodes) < 3:
            return True  # Or handle as an error/special case

        area = 0.0
        for i in range(len(self.nodes)):
            p1 = self.nodes[i].coordinates
            p2 = self.nodes[(i + 1) % len(self.nodes)].coordinates
            area += (p1[0] * p2[1]) - (p2[0] * p1[1])
        return area < 0

    def reverse_direction(self) -> None:
        """
        Reverses the order of nodes in the contour and swaps in/out handles.
        """
        if not self.nodes:
            return

        # Reverse the order of nodes
        self.nodes.reverse()

        # Swap in_handle and out_handle for each node
        for node in self.nodes:
            node.in_handle, node.out_handle = node.out_handle, node.in_handle

    def commands(
        self, vocabulary: type["CommandRepresentation"]
    ) -> Sequence["CommandRepresentation"]:
        return vocabulary.emit(self.nodes)

    def push(
        self,
        coordinates: npt.NDArray[np.float32],
        in_handle: Optional[npt.NDArray[np.float32]],
        out_handle: Optional[npt.NDArray[np.float32]],
    ) -> Node:
        node = Node(
            coordinates=coordinates,
            in_handle=in_handle,
            out_handle=out_handle,
            contour=self,
        )
        self.nodes.append(node)
        return node


class NodeGlyph:
    contours: List[NodeContour]
    origin: str

    def __init__(self, contours: List[NodeContour], origin="unknown"):
        self.contours = contours
        self.origin = origin

    def __eq__(self, other):
        if not isinstance(other, NodeGlyph):
            return NotImplemented
        result = len(self.contours) == len(other.contours) and all(
            c1 == c2 for c1, c2 in zip(self.contours, other.contours)
        )
        # Space for debugging code here
        return result

    def command_lists(
        self, vocabulary: type["CommandRepresentation"]
    ) -> List[Sequence["CommandRepresentation"]]:
        return [contour.commands(vocabulary) for contour in self.contours]

    @classmethod
    def from_command_lists(cls, contour_commands: List[List["CommandRepresentation"]]):
        contours = []
        representation_cls = (
            contour_commands[0][0].__class__
            if contour_commands and contour_commands[0]
            else None
        )
        if not representation_cls or not hasattr(
            representation_cls, "contour_from_commands"
        ):
            raise ValueError(
                "Commands must be of type CommandRepresentation to create NodeGlyph."
            )
        for cmds in contour_commands:
            contours.append(representation_cls.contour_from_commands(cmds))
        return cls(contours)

    def encode(
        self, vocabulary: type["CommandRepresentation"]
    ) -> Optional[List[npt.NDArray[np.float32]]]:
        contour_sequences = []

        for contour in self.contours:
            output: List[np.ndarray] = []

            def push_command(cmd: str, coords: List[float]):
                command_vector = np.zeros(vocabulary.command_width, dtype=np.float32)
                command_vector[vocabulary.encode_command(cmd)] = 1.0
                coord_array = np.array(coords, dtype=np.float32)
                padded_coords = np.pad(
                    coord_array, (0, vocabulary.coordinate_width - len(coords))  # type: ignore
                )
                output.append(np.concatenate((command_vector, padded_coords)))

            for command in contour.commands(vocabulary):
                push_command(command.command, command.coordinates)

            encoded_contour = np.array(output, dtype=np.float32)

            if encoded_contour.shape[0] > MAX_SEQUENCE_LENGTH:
                return None

            contour_sequences.append(encoded_contour)

        return contour_sequences if contour_sequences else None

    @classmethod
    def decode(
        cls,
        contour_sequences: List,
        representation_cls: type["CommandRepresentation"],
        return_raw_command_lists: bool = False,
    ):
        """
        Decodes a list of ndarray sequences into a list of NodeCommand sequences.
        This is a stateless conversion from ndarray representation to command objects.
        """
        command_keys = list(representation_cls.grammar.keys())
        glyph_commands: List[List["CommandRepresentation"]] = []

        for ndarrays in contour_sequences:
            # This must work for tensors and ndarrays, so we don't use
            # split_tensor here
            command_ndarray = ndarrays[:, : representation_cls.command_width]
            coord_ndarray = ndarrays[:, representation_cls.command_width :]

            contour_commands = []
            for i in range(command_ndarray.shape[0]):
                command_str = representation_cls.decode_command_one_hot(
                    command_ndarray[i]
                )
                num_coords = representation_cls.grammar[command_str]
                coords_slice = coord_ndarray[i, :num_coords]
                contour_commands.append(
                    representation_cls(command_str, coords_slice.tolist())
                )
                if command_str == "EOS":
                    break

            glyph_commands.append(contour_commands)

        if return_raw_command_lists:
            return glyph_commands

        return cls.from_command_lists(glyph_commands)

    def to_debug_string(self):
        path_data: List[str] = []
        for contour in self.contours:
            for node in contour.nodes:
                path_data.append(f"N {node.coordinates[0]} {node.coordinates[1]}")
                if node.in_handle is not None:
                    path_data.append(f"IN {node.in_handle[0]} {node.in_handle[1]}")
                if node.out_handle is not None:
                    path_data.append(f"OUT {node.out_handle[0]} {node.out_handle[1]}")
            path_data.append("Z")
        return " ".join(path_data)

    def normalize(self):
        for contour in self.contours:
            contour.normalize()
        self.contours.sort(
            key=lambda c: (
                (c.nodes[0].coordinates[1], c.nodes[0].coordinates[0])
                if c.nodes
                else (float("inf"), float("inf"))
            )
        )
        return self
