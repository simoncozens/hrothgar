from glyphogen.representations import (
    AbsoluteCoordinateRepresentation,
    CommandRepresentation,
)
from typing import List, Sequence, Self
import numpy as np
import torch
from glyphogen.nodeglyph import Node, NodeContour


class SVGCommand(CommandRepresentation):
    grammar = {
        "M": 2,  # Move to (2 coordinates)
        "L": 2,  # Line to (2 coordinates)
        "H": 1,  # Horizontal line to (1 coordinate)
        "V": 1,  # Vertical line to (1 coordinate)
        "C": 6,  # Cubic Bezier curve (6 coordinates)
        "Z": 0,  # Close path (no coordinates)
    }
    coordinate_representation = AbsoluteCoordinateRepresentation

    @classmethod
    def emit(cls, nodes: List["Node"]) -> Sequence[Self]:
        commands = []
        if not nodes:
            return commands
        # Emit move to for the first node
        first_node = nodes[0]
        pos = cls.coordinate_representation.emit_node_position(first_node)
        commands.append(SVGCommand("M", pos.tolist()))
        for node in nodes[1:] + [first_node]:
            # How does the previous node join to this?
            prev_node = node.previous
            if (node.in_handle is None) and (prev_node.out_handle is None):
                # Straight line
                pos = cls.coordinate_representation.emit_node_position(node).tolist()
                commands.append(SVGCommand("L", pos))
            else:
                # Cubic Bezier curve
                out_coords = cls.coordinate_representation.emit_out_handle(prev_node)
                if out_coords is None:
                    out_coords = cls.coordinate_representation.emit_node_position(
                        prev_node
                    )
                in_handle = cls.coordinate_representation.emit_in_handle(node)
                if in_handle is None:
                    in_handle = cls.coordinate_representation.emit_node_position(node)
                pos = cls.coordinate_representation.emit_node_position(node).tolist()
                coords = out_coords.tolist() + in_handle.tolist() + pos
                commands.append(SVGCommand("C", coords))
        # If the last command is a line back to the start, drop it, it's redundant with the Z
        if len(commands) > 1:
            last_cmd = commands[-1]
            if last_cmd.command == "L":
                start_pos = cls.coordinate_representation.emit_node_position(first_node)
                last_pos = np.array(last_cmd.coordinates, dtype=np.float32)
                if np.allclose(start_pos, last_pos):
                    commands.pop()
        commands.append(SVGCommand("Z", []))  # Close path
        return commands

    @classmethod
    def contour_from_commands(
        cls, commands: Sequence[CommandRepresentation]
    ) -> "NodeContour":
        contour = NodeContour([])
        # Expect a M
        if not commands or commands[0].command != "M":
            raise ValueError("SVGCommand sequence must start with an 'M' command.")
        cur_node = contour.push(
            coordinates=np.array(commands[0].coordinates, dtype=np.float32),
            in_handle=None,
            out_handle=None,
        )
        for command in commands[1:]:
            if command.command == "L":
                pos = np.array(command.coordinates, dtype=np.float32)
                cur_node = contour.push(
                    np.array(command.coordinates, dtype=np.float32),
                    in_handle=None,
                    out_handle=None,
                )
            elif command.command == "C":
                coords = command.coordinates
                out_handle = np.array(coords[0:2], dtype=np.float32)
                cur_node.out_handle = out_handle
                in_handle = np.array(coords[2:4], dtype=np.float32)
                pos = np.array(coords[4:6], dtype=np.float32)
                new_node = contour.push(
                    coordinates=pos,
                    in_handle=in_handle,
                    out_handle=None,
                )
                cur_node = new_node
            elif command.command == "Z":
                # Close path, do nothing
                pass
            else:
                raise ValueError(f"Unsupported SVG command: {command.command}")
        # If we ended up back at the start and the last node was a curve, merge the handles and remove the last node
        if np.array_equal(contour.nodes[0].coordinates, contour.nodes[-1].coordinates):
            start_node = contour.nodes[0]
            end_node = contour.nodes[-1]
            start_node.in_handle = end_node.in_handle
            contour.nodes.pop()

        return contour

    @classmethod
    def tensors_to_segments(cls, cmd, coord):
        """Convert an encoded command and coordinate tensor to segment points
        and control point counts for the diffvg renderer.

        This should be fairly simple for SVG commands as it maps to segments
        quite directly.
        """

        command_tensor = torch.argmax(cmd, dim=-1)
        all_points = []
        all_num_cp = []
        contour_splits = []
        point_splits = []
        current_contour_points = []
        current_contour_num_cp = []
        for i in range(len(command_tensor)):
            command = command_tensor[i]
            if command == cls.encode_command("M"):
                # Start a new contour
                if current_contour_points:
                    all_points.extend(current_contour_points)
                    all_num_cp.extend(current_contour_num_cp)
                    contour_splits.append(len(all_num_cp))
                    point_splits.append(len(all_points))
                    current_contour_points = []
                    current_contour_num_cp = []
                # Add the move-to point
                current_contour_points.append(coord[i, 0:2])
            elif command == cls.encode_command("L"):
                # Line to
                current_contour_points.append(coord[i, 0:2])
                current_contour_num_cp.append(0)
            elif command == cls.encode_command("C"):
                # Cubic Bezier curve
                current_contour_points.append(coord[i, 0:2])  # Control point 1
                current_contour_points.append(coord[i, 2:4])  # Control point 2
                current_contour_points.append(coord[i, 4:6])  # End point
                current_contour_num_cp.append(2)
            elif command == cls.encode_command("Z"):
                # Close path, do nothing
                pass
            else:
                raise ValueError(f"Unsupported SVG command in tensor: {command}")
        # Loop back to the start to close the contour
        if current_contour_points:
            current_contour_points.append(current_contour_points[0])
            current_contour_num_cp.append(0)
        if current_contour_points:
            all_points.extend(current_contour_points)
            all_num_cp.extend(current_contour_num_cp)
            contour_splits.append(len(all_num_cp))
            point_splits.append(len(all_points))
        if not all_points:
            return (
                torch.empty(0, 2, dtype=torch.float32, device=coord.device),
                torch.empty(0, dtype=torch.int32, device=coord.device),
                [],
                [],
            )
        return (
            torch.stack(all_points),
            torch.tensor(all_num_cp, dtype=torch.int32, device=coord.device),
            contour_splits,
            point_splits,
        )
