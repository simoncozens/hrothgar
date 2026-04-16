from typing import Dict, List, Sequence, TYPE_CHECKING
import numpy as np
import torch
from jaxtyping import Float
from hrothgar.vectorization.representations import (
    CommandRepresentation,
    RelativeCoordinateRepresentation,
)

if TYPE_CHECKING:
    from hrothgar.vectorization.nodeglyph import Node, NodeContour


class NodeCommand(CommandRepresentation):
    grammar = {
        "SOS": 0,
        "M": 2,  # Absolute move to (x, y)
        "L": 2,  # Relative line to (dx, dy)
        "LH": 1,  # Relative horizontal line to (dx)
        "LV": 1,  # Relative vertical line to (dy)
        "N": 6,  # Relative node with two handles (dx, dy, dhix, dhiy, dhox, dhoy)
        "NS": 5,  # Relative smooth node (dx, dy, angle, length_in, length_out)
        "NH": 4,  # Relative horizontal-handle node (dx, dy, dh_in_x, dh_out_x)
        "NV": 4,  # Relative vertical-handle node (dx, dy, dh_in_y, dh_out_y)
        "NCI": 4,  # Relative node with in-handle only (dx, dy, dhix, dhiy)
        "NCO": 4,  # Relative node with out-handle only (dx, dy, dhox, dhoy)
        "EOS": 0,
    }
    coordinate_representation = RelativeCoordinateRepresentation

    @classmethod
    def emit(cls, nodes: List["Node"]) -> Sequence["NodeCommand"]:
        commands = []
        if not nodes:
            return commands
        # Emit SOS
        commands.append(NodeCommand("SOS", []))
        # Emit move to for the first node
        first_node = nodes[0]
        pos = cls.coordinate_representation.emit_node_position(first_node)
        commands.append(NodeCommand("M", pos.tolist()))
        for ix, node in enumerate(nodes):
            if ix == 0:
                rel_pos = [0.0, 0.0]
            else:
                rel_pos = cls.coordinate_representation.emit_node_position(
                    node
                ).tolist()
            in_handle = cls.coordinate_representation.emit_in_handle(node)
            out_handle = cls.coordinate_representation.emit_out_handle(node)
            if node.is_line:
                if node.is_horizontal_line:
                    # Horizontal line
                    commands.append(NodeCommand("LH", [rel_pos[0]]))
                elif node.is_vertical_line:
                    # Vertical line
                    commands.append(NodeCommand("LV", [rel_pos[1]]))
                else:
                    # Straight line
                    commands.append(NodeCommand("L", rel_pos))
            elif node.handles_horizontal:
                assert in_handle is not None and out_handle is not None
                commands.append(
                    NodeCommand("NH", rel_pos + [in_handle[0], out_handle[0]])
                )
            elif node.handles_vertical:
                assert in_handle is not None and out_handle is not None
                commands.append(
                    NodeCommand("NV", rel_pos + [in_handle[1], out_handle[1]])
                )
            elif node.is_smooth:
                assert in_handle is not None and out_handle is not None
                # For a smooth node, the handles are collinear and opposite.
                # We define the geometry by the angle of the outgoing handle,
                # and the lengths of both handles.
                # To be robust to slight imprecision in the source font,
                # we average the direction of the two handles.
                vec_in = in_handle
                vec_out = out_handle
                vec_in_opposite = -vec_in
                norm_in = np.linalg.norm(vec_in_opposite)
                norm_out = np.linalg.norm(vec_out)

                # Average the unit vectors
                avg_vec = (vec_in_opposite / norm_in) + (vec_out / norm_out)
                angle = np.arctan2(avg_vec[1], avg_vec[0])

                length_in = np.linalg.norm(vec_in)
                length_out = np.linalg.norm(vec_out)
                commands.append(
                    NodeCommand(
                        "NS",
                        rel_pos + [float(angle), float(length_in), float(length_out)],
                    )
                )
            elif in_handle is not None and out_handle is not None:
                commands.append(
                    NodeCommand("N", rel_pos + in_handle.tolist() + out_handle.tolist())
                )
            elif in_handle is not None:
                commands.append(NodeCommand("NCI", rel_pos + in_handle.tolist()))
            elif out_handle is not None:
                commands.append(NodeCommand("NCO", rel_pos + out_handle.tolist()))
            else:
                # This case should be handled by is_line, but as a fallback
                commands.append(NodeCommand("L", rel_pos))

        # Emit EOS
        commands.append(NodeCommand("EOS", []))

        return commands

    @classmethod
    def contour_from_commands(
        cls, commands: Sequence[CommandRepresentation], tolerant=True
    ) -> "NodeContour":
        from hrothgar.vectorization.nodeglyph import NodeContour

        contour = NodeContour([])
        commands = list(commands)
        # Pop SOS
        if commands[0].command == "SOS":
            commands.pop(0)
        # Expect a M
        if len(commands) < 1:
            return contour
        if not commands or commands[0].command != "M" and not tolerant:
            raise ValueError(
                f"NodeCommand second command must be 'M' command, found {commands[0].command}."
            )
        # Use .unroll_relative_coordinates to get absolute positions
        # Make them into tensors
        command_tensors = []
        coord_tensors = []
        max_coords = cls.coordinate_width
        for cmd in commands:
            command_tensors.append(cls.encode_command_one_hot(cmd.command))
            padded = cmd.coordinates + [0] * (max_coords - len(cmd.coordinates))
            coord_tensors.append(torch.tensor(padded, dtype=torch.float32))
        sequence_tensor = torch.cat(
            [torch.stack(command_tensors), torch.stack(coord_tensors)], dim=1
        )
        _, abs_coords = cls.split_tensor(
            cls.unroll_relative_coordinates(sequence_tensor)
        )
        abs_coords_np = abs_coords.cpu().numpy()
        # Reconstruct contour
        # Iterate from the first actual node command (after SOS and M)
        for idx, step in enumerate(commands[1:]):
            command = step.command
            absolute_coords = abs_coords_np[idx + 1]  # +1 to skip M
            cur_coord = absolute_coords[0:2]
            if command in ["L", "LH", "LV"]:
                contour.push(
                    coordinates=cur_coord.copy(), in_handle=None, out_handle=None
                )
            elif command == "N":
                in_handle = np.array(absolute_coords[2:4], dtype=np.float32)
                out_handle = np.array(absolute_coords[4:6], dtype=np.float32)
                contour.push(
                    coordinates=cur_coord.copy(),
                    in_handle=in_handle,
                    out_handle=out_handle,
                )
            elif command == "NH" or command == "NV" or command == "NS":
                in_handle = np.array(absolute_coords[2:4], dtype=np.float32)
                out_handle = np.array(absolute_coords[4:6], dtype=np.float32)
                contour.push(
                    coordinates=cur_coord.copy(),
                    in_handle=in_handle,
                    out_handle=out_handle,
                )
            elif command == "NCI":
                in_handle = np.array(absolute_coords[2:4], dtype=np.float32)
                contour.push(
                    coordinates=cur_coord.copy(),
                    in_handle=in_handle,
                    out_handle=None,
                )
            elif command == "NCO":
                out_handle = np.array(absolute_coords[4:6], dtype=np.float32)
                contour.push(
                    coordinates=cur_coord.copy(),
                    in_handle=None,
                    out_handle=out_handle,
                )
            elif command == "EOS":
                # End of sequence, do nothing
                pass
            elif not tolerant:
                raise ValueError(f"Unsupported Node command: {command}")
        return contour

    @classmethod
    def image_space_to_mask_space(cls, sequence, box: Float[torch.Tensor, "4"]):
        """
        Normalizes a sequence's image-space coordinates to the model's internal
        [-1, 1] range relative to a given bounding box. This version is vectorized,
        differentiates between absolute (M) and relative (other) coordinates,
        and avoids in-place operations to be torch.compile-friendly.
        """
        commands, coords_img_space = cls.split_tensor(sequence)
        x1, y1, x2, y2 = box
        width = torch.clamp(x2 - x1, min=1.0)
        height = torch.clamp(y2 - y1, min=1.0)
        avg_dim = (width + height) / 2.0

        command_indices = torch.argmax(commands, dim=-1)
        m_mask = (command_indices == NodeCommand.encode_command("M")).unsqueeze(1)
        lv_mask = (command_indices == NodeCommand.encode_command("LV")).unsqueeze(1)
        ns_mask = (command_indices == NodeCommand.encode_command("NS")).unsqueeze(1)

        # --- Calculate treatment for ALL coordinates as if they were RELATIVE deltas ---
        coord_width = coords_img_space.shape[1]
        scale_vec_rel = torch.tensor(
            [2.0 / width, 2.0 / height] * (coord_width // 2)
            + [2.0 / width] * (coord_width % 2),
            device=sequence.device,
            dtype=sequence.dtype,
        )
        # LV command's first (and only) coord is a dY, so it should be scaled by height.
        scale_vec_lv = scale_vec_rel.clone()
        if coord_width > 0:
            scale_vec_lv[0] = 2.0 / height

        # NS has special scaling for angle and lengths
        scale_vec_ns = scale_vec_rel.clone()
        if coord_width > 4:
            scale_vec_ns[2] = 1.0 / np.pi  # Angle -> [-1, 1]
            scale_vec_ns[3] = 2.0 / avg_dim  # Lengths
            scale_vec_ns[4] = 2.0 / avg_dim

        relative_result = coords_img_space * scale_vec_rel
        relative_result = torch.where(
            lv_mask, coords_img_space * scale_vec_lv, relative_result
        )
        relative_result = torch.where(
            ns_mask, coords_img_space * scale_vec_ns, relative_result
        )

        # --- Calculate treatment for ALL coordinates as if they were ABSOLUTE positions ---
        absolute_result = coords_img_space.clone()
        # Translate
        absolute_result[:, 0] -= x1
        absolute_result[:, 1] -= y1
        # Scale to [0,1]
        absolute_result[:, 0] /= width
        absolute_result[:, 1] /= height
        # Shift to [-1,1]
        absolute_result = (absolute_result * 2) - 1

        # --- Combine ---
        # Where the command is 'M', use the absolute result, otherwise use the relative result.
        normalized_coords = torch.where(m_mask, absolute_result, relative_result)

        return torch.cat([commands, normalized_coords], dim=-1)

    @classmethod
    def mask_space_to_image_space(cls, sequence, box):
        """
        Denormalizes a sequence's [-1, 1] coordinates back to image space.
        This version is vectorized, differentiates between absolute (M) and
        relative (other) coordinates, and avoids in-place operations to be
        torch.compile-friendly.
        """
        commands, coords_norm = cls.split_tensor(sequence)
        x1, y1, x2, y2 = box
        width = torch.clamp(x2 - x1, min=1.0)
        height = torch.clamp(y2 - y1, min=1.0)
        avg_dim = (width + height) / 2.0

        command_indices = torch.argmax(commands, dim=-1)
        m_mask = (command_indices == NodeCommand.encode_command("M")).unsqueeze(1)
        lv_mask = (command_indices == NodeCommand.encode_command("LV")).unsqueeze(1)
        ns_mask = (command_indices == NodeCommand.encode_command("NS")).unsqueeze(1)

        # --- Handle Relative Coordinates (Scaling from [-1, 1] space) ---
        coord_width = coords_norm.shape[1]
        # Deltas in [-1,1] space should be scaled by width/2 to be correct in image space
        scale_vec_rel = torch.tensor(
            [width / 2.0, height / 2.0] * (coord_width // 2)
            + [width / 2.0] * (coord_width % 2),
            device=sequence.device,
            dtype=sequence.dtype,
        )
        # LV command's first (and only) coord is a dY, so it should be scaled by height.
        scale_vec_lv = scale_vec_rel.clone()
        if coord_width > 0:
            scale_vec_lv[0] = height / 2.0

        # NS has special scaling for angle and lengths
        scale_vec_ns = scale_vec_rel.clone()
        if coord_width > 4:
            scale_vec_ns[2] = np.pi  # Denormalize angle
            scale_vec_ns[3] = avg_dim / 2.0  # Denormalize lengths
            scale_vec_ns[4] = avg_dim / 2.0

        relative_result = coords_norm * scale_vec_rel
        relative_result = torch.where(
            lv_mask, coords_norm * scale_vec_lv, relative_result
        )
        relative_result = torch.where(
            ns_mask, coords_norm * scale_vec_ns, relative_result
        )

        # --- Handle Absolute 'M' Coordinates (Translation and Scaling from [-1, 1]) ---
        absolute_result = coords_norm.clone()
        # Shift from [-1,1] to [0,1]
        absolute_result = (absolute_result + 1) / 2
        # Scale to image dims
        absolute_result[:, 0] *= width
        absolute_result[:, 1] *= height
        # Translate
        absolute_result[:, 0] += x1
        absolute_result[:, 1] += y1

        # --- Combine ---
        # Where the command is 'M', use the absolute result, otherwise use the relative result.
        denormalized_coords = torch.where(m_mask, absolute_result, relative_result)

        return torch.cat([commands, denormalized_coords], dim=-1)

    @classmethod
    def compute_deltas(cls, sequence: torch.Tensor) -> torch.Tensor:
        """
        Computes the (dx, dy) deltas for each step in the sequence.
        Handles L, N, NS, NH, NV, NCI, NCO, LH, LV.
        Returns tensor of shape (..., 2).
        """
        commands, rel_coords = cls.split_tensor(sequence)
        command_indices = torch.argmax(commands, dim=-1)

        l_index = cls.encode_command("L")
        lh_index = cls.encode_command("LH")
        lv_index = cls.encode_command("LV")
        n_index = cls.encode_command("N")
        ns_index = cls.encode_command("NS")
        nh_index = cls.encode_command("NH")
        nv_index = cls.encode_command("NV")
        nci_index = cls.encode_command("NCI")
        nco_index = cls.encode_command("NCO")

        # Mask for commands that have relative XY motion
        relative_xy_mask = (
            (command_indices == l_index)
            | (command_indices == n_index)
            | (command_indices == ns_index)
            | (command_indices == nh_index)
            | (command_indices == nv_index)
            | (command_indices == nci_index)
            | (command_indices == nco_index)
        )

        # Create deltas without in-place assignment to keep dynamo happy
        zeros_like_xy = torch.zeros_like(rel_coords[..., 0:2])

        # Deltas for relative XY commands
        deltas_xy = torch.where(
            relative_xy_mask.unsqueeze(-1), rel_coords[..., 0:2], zeros_like_xy
        )

        # Deltas for LH and LV (single-axis moves)
        lh_mask = command_indices == lh_index
        lv_mask = command_indices == lv_index

        # Helper for scalar zero
        zeros_scalar = torch.zeros_like(rel_coords[..., 0])

        lh_vec = torch.where(lh_mask, rel_coords[..., 0], zeros_scalar)
        lv_vec = torch.where(lv_mask, rel_coords[..., 0], zeros_scalar)

        # Combine into a single delta tensor
        deltas = torch.stack(
            (deltas_xy[..., 0] + lh_vec, deltas_xy[..., 1] + lv_vec), dim=-1
        )
        return deltas

    @classmethod
    def unroll_relative_coordinates(cls, sequence: torch.Tensor) -> torch.Tensor:
        """
        Converts a sequence with relative coordinates to one with absolute coordinates.
        This is a differentiable and vectorized operation.
        """
        commands, rel_coords = cls.split_tensor(sequence)

        # Re-use the new helper
        deltas = cls.compute_deltas(sequence)

        command_indices = torch.argmax(commands, dim=-1)
        m_index = cls.encode_command("M")

        # Seed absolute positions with the absolute move from the M command (vectorized)
        m_mask = command_indices == m_index
        base_pos = torch.where(
            m_mask.unsqueeze(1), rel_coords[:, 0:2], torch.zeros_like(deltas)
        )

        # Calculate absolute positions with cumsum
        abs_positions = torch.cumsum(deltas + base_pos, dim=0)

        # Mask for commands that have position coordinates
        l_index = cls.encode_command("L")
        lh_index = cls.encode_command("LH")
        lv_index = cls.encode_command("LV")
        n_index = cls.encode_command("N")
        ns_index = cls.encode_command("NS")
        nh_index = cls.encode_command("NH")
        nv_index = cls.encode_command("NV")
        nci_index = cls.encode_command("NCI")
        nco_index = cls.encode_command("NCO")

        has_pos_coords_mask = (
            m_mask
            | (command_indices == l_index)
            | (command_indices == lh_index)
            | (command_indices == lv_index)
            | (command_indices == n_index)
            | (command_indices == ns_index)
            | (command_indices == nh_index)
            | (command_indices == nv_index)
            | (command_indices == nci_index)
            | (command_indices == nco_index)
        )

        # Build absolute coordinates column-wise without in-place masked writes
        zeros_scalar = torch.zeros_like(rel_coords[:, 0])

        # Position columns (common to all node types)
        pos_x = torch.where(has_pos_coords_mask, abs_positions[:, 0], zeros_scalar)
        pos_y = torch.where(has_pos_coords_mask, abs_positions[:, 1], zeros_scalar)

        # --- Handle Calculation ---
        in_x = torch.zeros_like(pos_x)
        in_y = torch.zeros_like(pos_y)
        out_x = torch.zeros_like(pos_x)
        out_y = torch.zeros_like(pos_y)

        # Standard Node (N)
        n_mask = command_indices == n_index
        in_x = torch.where(n_mask, abs_positions[:, 0] + rel_coords[:, 2], in_x)
        in_y = torch.where(n_mask, abs_positions[:, 1] + rel_coords[:, 3], in_y)
        out_x = torch.where(n_mask, abs_positions[:, 0] + rel_coords[:, 4], out_x)
        out_y = torch.where(n_mask, abs_positions[:, 1] + rel_coords[:, 5], out_y)

        # Smooth Node (NS)
        ns_mask = command_indices == ns_index
        ns_angle = rel_coords[:, 2]
        ns_len_in = rel_coords[:, 3]
        ns_len_out = rel_coords[:, 4]
        cos_angle = torch.cos(ns_angle)
        sin_angle = torch.sin(ns_angle)

        ns_in_x = abs_positions[:, 0] - ns_len_in * cos_angle
        ns_in_y = abs_positions[:, 1] - ns_len_in * sin_angle
        ns_out_x = abs_positions[:, 0] + ns_len_out * cos_angle
        ns_out_y = abs_positions[:, 1] + ns_len_out * sin_angle

        in_x = torch.where(ns_mask, ns_in_x, in_x)
        in_y = torch.where(ns_mask, ns_in_y, in_y)
        out_x = torch.where(ns_mask, ns_out_x, out_x)
        out_y = torch.where(ns_mask, ns_out_y, out_y)

        # Other node types (NH, NV, NCI, NCO)
        nh_mask = command_indices == nh_index
        nv_mask = command_indices == nv_index
        nci_mask = command_indices == nci_index
        nco_mask = command_indices == nco_index

        # Handles for NH
        in_x = torch.where(nh_mask, abs_positions[:, 0] + rel_coords[:, 2], in_x)
        in_y = torch.where(nh_mask, abs_positions[:, 1], in_y)
        out_x = torch.where(nh_mask, abs_positions[:, 0] + rel_coords[:, 3], out_x)
        out_y = torch.where(nh_mask, abs_positions[:, 1], out_y)

        # Handles for NV
        in_x = torch.where(nv_mask, abs_positions[:, 0], in_x)
        in_y = torch.where(nv_mask, abs_positions[:, 1] + rel_coords[:, 2], in_y)
        out_x = torch.where(nv_mask, abs_positions[:, 0], out_x)
        out_y = torch.where(nv_mask, abs_positions[:, 1] + rel_coords[:, 3], out_y)

        # Handles for NCI (in-handle only)
        in_x = torch.where(nci_mask, abs_positions[:, 0] + rel_coords[:, 2], in_x)
        in_y = torch.where(nci_mask, abs_positions[:, 1] + rel_coords[:, 3], in_y)

        # Handles for NCO (out-handle only)
        out_x = torch.where(nco_mask, abs_positions[:, 0] + rel_coords[:, 2], out_x)
        out_y = torch.where(nco_mask, abs_positions[:, 1] + rel_coords[:, 3], out_y)

        abs_coords = torch.stack((pos_x, pos_y, in_x, in_y, out_x, out_y), dim=1)

        return torch.cat([commands, abs_coords], dim=1)

    # Class-level storage for normalization statistics
    _stats_initialized = False
    _mean_tensor = None
    _std_tensor = None

    @classmethod
    def initialize_stats(cls, stats_path: str = "coord_stats.pt"):
        """Load stats and create broadcastable tensors for standardization."""
        from collections import defaultdict

        try:
            stats = torch.load(stats_path)
        except FileNotFoundError:
            print(
                f"Warning: {stats_path} not found. Using default (0,1) stats. "
                "Run analyze_dataset_stats.py to generate it."
            )
            stats = defaultdict(lambda: {"mean": 0.0, "std": 1.0})

        mean_tensor = torch.zeros(cls.command_width, cls.coordinate_width)
        std_tensor = torch.ones(cls.command_width, cls.coordinate_width)
        cmd_indices = {cmd: cls.encode_command(cmd) for cmd in cls.grammar.keys()}

        for cmd_name, cmd_idx in cmd_indices.items():
            if cmd_name == "M":
                mean_tensor[cmd_idx, 0] = stats["M_abs_x"]["mean"]
                std_tensor[cmd_idx, 0] = stats["M_abs_x"]["std"]
                mean_tensor[cmd_idx, 1] = stats["M_abs_y"]["mean"]
                std_tensor[cmd_idx, 1] = stats["M_abs_y"]["std"]
            elif cmd_name in ["L", "N", "NS", "NH", "NV", "NCI", "NCO"]:
                mean_tensor[cmd_idx, 0] = stats["L_rel_dx"]["mean"]
                std_tensor[cmd_idx, 0] = stats["L_rel_dx"]["std"]
                mean_tensor[cmd_idx, 1] = stats["L_rel_dy"]["mean"]
                std_tensor[cmd_idx, 1] = stats["L_rel_dy"]["std"]
            elif cmd_name == "LH":
                mean_tensor[cmd_idx, 0] = stats["L_rel_dx"]["mean"]
                std_tensor[cmd_idx, 0] = stats["L_rel_dx"]["std"]
            elif cmd_name == "LV":
                mean_tensor[cmd_idx, 0] = stats["L_rel_dy"]["mean"]
                std_tensor[cmd_idx, 0] = stats["L_rel_dy"]["std"]

            if cmd_name == "N":
                mean_tensor[cmd_idx, 2] = stats["C_in_dx"]["mean"]
                std_tensor[cmd_idx, 2] = stats["C_in_dx"]["std"]
                mean_tensor[cmd_idx, 3] = stats["C_in_dy"]["mean"]
                std_tensor[cmd_idx, 3] = stats["C_in_dy"]["std"]
                mean_tensor[cmd_idx, 4] = stats["C_out_dx"]["mean"]
                std_tensor[cmd_idx, 4] = stats["C_out_dx"]["std"]
                mean_tensor[cmd_idx, 5] = stats["C_out_dy"]["mean"]
                std_tensor[cmd_idx, 5] = stats["C_out_dy"]["std"]
            elif cmd_name == "NCI":
                mean_tensor[cmd_idx, 2] = stats["C_in_dx"]["mean"]
                std_tensor[cmd_idx, 2] = stats["C_in_dx"]["std"]
                mean_tensor[cmd_idx, 3] = stats["C_in_dy"]["mean"]
                std_tensor[cmd_idx, 3] = stats["C_in_dy"]["std"]
            elif cmd_name == "NCO":
                mean_tensor[cmd_idx, 2] = stats["C_out_dx"]["mean"]
                std_tensor[cmd_idx, 2] = stats["C_out_dx"]["std"]
                mean_tensor[cmd_idx, 3] = stats["C_out_dy"]["mean"]
                std_tensor[cmd_idx, 3] = stats["C_out_dy"]["std"]
            elif cmd_name == "NH":
                mean_tensor[cmd_idx, 2] = stats["C_in_dx"]["mean"]
                std_tensor[cmd_idx, 2] = stats["C_in_dx"]["std"]
                mean_tensor[cmd_idx, 3] = stats["C_out_dx"]["mean"]
                std_tensor[cmd_idx, 3] = stats["C_out_dx"]["std"]
            elif cmd_name == "NV":
                mean_tensor[cmd_idx, 2] = stats["C_in_dy"]["mean"]
                std_tensor[cmd_idx, 2] = stats["C_in_dy"]["std"]
                mean_tensor[cmd_idx, 3] = stats["C_out_dy"]["mean"]
                std_tensor[cmd_idx, 3] = stats["C_out_dy"]["std"]
            elif cmd_name == "NS":
                mean_tensor[cmd_idx, 2] = stats["NS_angle"]["mean"]
                std_tensor[cmd_idx, 2] = stats["NS_angle"]["std"]
                mean_tensor[cmd_idx, 3] = stats["NS_len_in"]["mean"]
                std_tensor[cmd_idx, 3] = stats["NS_len_in"]["std"]
                mean_tensor[cmd_idx, 4] = stats["NS_len_out"]["mean"]
                std_tensor[cmd_idx, 4] = stats["NS_len_out"]["std"]

        cls._mean_tensor = mean_tensor
        cls._std_tensor = std_tensor
        cls._stats_initialized = True

    @classmethod
    def get_initial_stats_dict(cls) -> Dict[str, List[float]]:
        """Get an initial stats dictionary with empty lists for each relevant field."""
        return {
            "M_abs_x": [],
            "M_abs_y": [],
            "L_rel_dx": [],
            "L_rel_dy": [],
            "C_in_dx": [],
            "C_in_dy": [],
            "C_out_dx": [],
            "C_out_dy": [],
            "NS_angle": [],
            "NS_len_in": [],
            "NS_len_out": [],
        }

    @classmethod
    def update_stats_dict_with_command(cls, STAT_GROUPS, command, coord_vec):
        # Handle absolute coordinates

        if command == "M":
            STAT_GROUPS["M_abs_x"].append(coord_vec[0].item())
            STAT_GROUPS["M_abs_y"].append(coord_vec[1].item())
        elif command in [
            "L",
            "N",
            "NS",
            "NH",
            "NV",
            "NCI",
            "NCO",
        ]:
            # All these commands have a relative dx, dy component
            STAT_GROUPS["L_rel_dx"].append(coord_vec[0].item())
            STAT_GROUPS["L_rel_dy"].append(coord_vec[1].item())

        if command == "LH":
            STAT_GROUPS["L_rel_dx"].append(coord_vec[0].item())
        elif command == "LV":
            STAT_GROUPS["L_rel_dy"].append(coord_vec[0].item())

        # Handle deltas
        if command == "N":
            STAT_GROUPS["C_in_dx"].append(coord_vec[2].item())
            STAT_GROUPS["C_in_dy"].append(coord_vec[3].item())
            STAT_GROUPS["C_out_dx"].append(coord_vec[4].item())
            STAT_GROUPS["C_out_dy"].append(coord_vec[5].item())
        elif command == "NCI":
            STAT_GROUPS["C_in_dx"].append(coord_vec[2].item())
            STAT_GROUPS["C_in_dy"].append(coord_vec[3].item())
        elif command == "NCO":
            STAT_GROUPS["C_out_dx"].append(coord_vec[2].item())
            STAT_GROUPS["C_out_dy"].append(coord_vec[3].item())
        elif command == "NH":
            STAT_GROUPS["C_in_dx"].append(coord_vec[2].item())
            STAT_GROUPS["C_out_dx"].append(coord_vec[3].item())
        elif command == "NV":
            STAT_GROUPS["C_in_dy"].append(coord_vec[2].item())
            STAT_GROUPS["C_out_dy"].append(coord_vec[3].item())
        elif command == "NS":
            STAT_GROUPS["NS_angle"].append(coord_vec[2].item())
            STAT_GROUPS["NS_len_in"].append(coord_vec[3].item())
            STAT_GROUPS["NS_len_out"].append(coord_vec[4].item())

    @classmethod
    def get_stats_for_sequence(cls, command_indices: torch.Tensor):
        """Get mean and std tensors for a sequence of command indices."""
        if not cls._stats_initialized:
            cls.initialize_stats()
        assert cls._mean_tensor is not None and cls._std_tensor is not None
        # Move stats to same device as input
        mean_tensor = cls._mean_tensor.to(command_indices.device)
        std_tensor = cls._std_tensor.to(command_indices.device)
        means = mean_tensor[command_indices]
        stds = std_tensor[command_indices]
        return means, stds

    @classmethod
    def standardize(cls, coords: torch.Tensor, means: torch.Tensor, stds: torch.Tensor):
        """Standardize coordinates using provided means and stds."""
        return (coords - means) / stds

    @classmethod
    def de_standardize(
        cls, coords_std: torch.Tensor, means: torch.Tensor, stds: torch.Tensor
    ):
        """De-standardize coordinates using provided means and stds."""
        return coords_std * stds + means

    @classmethod
    def tensors_to_segments(cls, cmd, coord):
        """Convert an encoded command and coordinate tensor to segment points
        and control point counts for the diffvg renderer.

        This almost certainly needs rewriting for the current representation.
        """

        command_tensor = torch.argmax(cmd, dim=-1)

        all_points = []
        all_num_cp = []
        contour_splits = []
        point_splits = []

        contour_nodes = []
        cmd_sos_val = cls.encode_command("SOS")
        cmd_eos_val = cls.encode_command("EOS")
        cmd_n_val = cls.encode_command("N")

        for i in range(len(command_tensor)):
            command = command_tensor[i]
            is_sos = command == cmd_sos_val
            is_eos = command == cmd_eos_val

            if is_sos:
                # Skip SOS token, it just marks the start
                continue
            elif is_eos:
                if len(contour_nodes) > 0:
                    # Process the collected contour
                    # First point is the start point
                    all_points.append(contour_nodes[0][1][0:2])

                    # Segments from node to node
                    for j in range(len(contour_nodes)):
                        p1_cmd, p1_coord = contour_nodes[j]
                        p2_cmd, p2_coord = contour_nodes[(j + 1) % len(contour_nodes)]

                        is_curve = (p1_cmd == cmd_n_val) and (p2_cmd == cmd_n_val)
                        p1_pos, p2_pos = p1_coord[0:2], p2_coord[0:2]
                        # Handle positions are absolute.
                        p1_hout = p1_coord[4:6]
                        p2_hin = p2_coord[2:4]
                        # If they change to relative, use this instead:
                        # p1_hout, p2_hin = p1_pos + p1_coord[4:6], p2_pos + p2_coord[2:4]

                        if is_curve:
                            all_points.extend([p1_hout, p2_hin, p2_pos])
                            all_num_cp.append(2)
                        else:
                            all_points.append(p2_pos)
                            all_num_cp.append(0)

                    contour_splits.append(len(all_num_cp))
                    point_splits.append(len(all_points))

                contour_nodes = []
                break
            else:
                # It's a node, add it to the current contour
                contour_nodes.append((command, coord[i]))

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
