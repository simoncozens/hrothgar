from typing import List

import numpy as np
from kurbopy import BezPath, Point
from PIL import Image, ImageDraw

from glyphogen.representations.svgcommand import SVGCommand
from glyphogen.coordinate import get_bounds
from glyphogen.hyperparameters import GEN_IMAGE_SIZE
from glyphogen.nodeglyph import NodeGlyph


class SVGGlyph:
    commands: List[SVGCommand]
    origin: str

    def __init__(self, commands, origin="unknown"):
        if commands and commands[0].command != "M":
            raise ValueError("SVG path must start with a 'M' command")
        self.commands = commands
        self.origin = origin

    @classmethod
    def from_node_glyph(cls, node_glyph: "NodeGlyph") -> "SVGGlyph":
        command_lists = node_glyph.command_lists(SVGCommand)
        commands = []
        for contour in command_lists:
            commands.extend(contour)
        return cls(commands, node_glyph.origin)

    def to_node_glyph(self) -> "NodeGlyph":
        if not self.commands:
            return NodeGlyph([], self.origin)

        # Split into a list of contours
        svg_contours: List[List[SVGCommand]] = []
        current_contour: List[SVGCommand] = []
        for command in self.commands:
            current_contour.append(command)
            if command.command == "Z":
                svg_contours.append(current_contour)
                current_contour = []
        if current_contour:
            svg_contours.append(current_contour)

        node_glyph = NodeGlyph(
            [SVGCommand.contour_from_commands(contour) for contour in svg_contours],
            self.origin,
        )
        return node_glyph

    def to_svg_string(self):
        path_data: List[str] = []
        for cmd in self.commands:
            path_data.append(cmd.command)
            path_data.extend(map(lambda x: str(int(x)), cmd.coordinates))
        return " ".join(path_data)

    def to_bezpaths(self) -> List[BezPath]:
        if not self.commands:
            return []

        svg_contours: List[List[SVGCommand]] = []
        current_contour: List[SVGCommand] = []
        for command in self.commands:
            current_contour.append(command)
            if command.command == "Z":
                svg_contours.append(current_contour)
                current_contour = []
        if current_contour:
            svg_contours.append(current_contour)

        kurbopy_contours = []

        for contour_cmds in svg_contours:
            path = BezPath()
            for cmd in contour_cmds:
                if cmd.command == "M":
                    path.move_to(Point(*cmd.coordinates))
                elif cmd.command == "L":
                    path.line_to(Point(*cmd.coordinates))
                elif cmd.command == "C":
                    path.curve_to(
                        Point(cmd.coordinates[0], cmd.coordinates[1]),
                        Point(cmd.coordinates[2], cmd.coordinates[3]),
                        Point(cmd.coordinates[4], cmd.coordinates[5]),
                    )
                elif cmd.command == "Z":
                    path.close_path()
            kurbopy_contours.append(path)
        return kurbopy_contours

    def get_segmentation_data(self):
        segmentation_data = []
        kurbopy_contours = self.to_bezpaths()
        for i, path_i in enumerate(kurbopy_contours):
            segs = path_i.segments()
            if not segs or not segs[0]:
                continue

            points = [(pt.x, pt.y) for pt in path_i.flatten(1.0)]
            if not points:
                continue

            bbox = get_bounds(points)

            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width <= 0 or height <= 0 or len(points) < 3:
                print(
                    f"Warning: Skipping contour with zero width/height in {self.origin}. Bbox: {bbox}"
                )
                continue

            containment_count = 0
            test_point = path_i.segments()[0].start()
            for j, path_j in enumerate(kurbopy_contours):
                if i == j:
                    continue
                if path_j.contains(test_point):
                    containment_count += 1
            is_hole = containment_count % 2 == 1

            img = Image.new("L", (GEN_IMAGE_SIZE[0], GEN_IMAGE_SIZE[1]), 0)
            draw = ImageDraw.Draw(img)
            draw.polygon(points, fill=1)
            mask = np.array(img, dtype=np.uint8)

            segmentation_data.append(
                {
                    "bbox": bbox,
                    "label": 1 if is_hole else 0,
                    "mask": mask,
                }
            )

        return segmentation_data

    @classmethod
    def from_svg_string(cls, svg_string: str, origin="unknown") -> "SVGGlyph":
        # For testing
        tokens = svg_string.strip().split()
        commands: List[SVGCommand] = []
        i = 0
        while i < len(tokens):
            command = tokens[i]
            if command == "M":
                commands.append(
                    SVGCommand(command, [float(tokens[i + 1]), float(tokens[i + 2])])
                )
                i += 3
            elif command == "L":
                commands.append(
                    SVGCommand(command, [float(tokens[i + 1]), float(tokens[i + 2])])
                )
                i += 3
            elif command == "C":
                coords = [
                    float(tokens[i + 1]),
                    float(tokens[i + 2]),
                    float(tokens[i + 3]),
                    float(tokens[i + 4]),
                    float(tokens[i + 5]),
                    float(tokens[i + 6]),
                ]
                commands.append(SVGCommand(command, coords))
                i += 7
            elif command == "Z":
                commands.append(SVGCommand(command, []))
                i += 1
            else:
                raise ValueError(f"Invalid SVG command: {command}")
        return cls(commands, origin)
