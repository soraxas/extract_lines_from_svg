#!/usr/bin/env python

import re
import argparse
import logging
import json

import tempfile
import subprocess

import numpy as np
import plotly.graph_objects as go

from collections import defaultdict
from typing import List, Tuple, Optional

from svgpathtools.parser import parse_transform as spt_parse_transform
from svgpathtools.parser import parse_path as spt_parse_path
from svgpathtools.path import transform as spt_transform

from xml.dom import Node, minidom


EPS = 1e-5

LOGGER = logging.getLogger(__file__)


def list_of_num(arg):
    arg = arg.split(",")
    return tuple(map(float, arg))


parser = argparse.ArgumentParser()
parser.add_argument("svgfile", metavar="FILE", type=str, help="Source file")
parser.add_argument("--show-all-lines", action="store_true")
parser.add_argument("--show-ticks", action="store_true")
parser.add_argument("--show-grouped-plots", action="store_true")
parser.add_argument(
    "--xticks",
    "-x",
    type=list_of_num,
)
parser.add_argument(
    "--yticks",
    "-y",
    type=list_of_num,
)
parser.add_argument("--save-as-json", "-s", action="store_true")


def get_plotly_go(fig=None, reverse: bool = True):
    if fig is None:
        fig = go.Figure()
        if reverse:
            fig.update_yaxes(autorange="reversed")
        # fig.update_yaxes(tickformat="e")
    return fig


def rgb_string_extract_val(rgb_str: str) -> np.ndarray:
    inner = re.search(r"[(]([^)]*)", rgb_str)[1]
    return np.array([float(val.strip()[:-1]) for val in inner.split(",")])


def rgb_string_add_alpha(rgb_str: str, alpha: float) -> str:
    """
    TEST:

        rgb_str = "rgb(83.920288%, 15.293884%, 15.686035%)"

        rgb_string_add_alpha(rgb_str, 0.5)
    """

    rgb_vals = rgb_string_extract_val(rgb_str=rgb_str)
    rgb_vals = (rgb_vals / 100) * 255
    return f"rgba({rgb_vals[0]},{rgb_vals[1]},{rgb_vals[2]},{alpha})"


def path_points(path):
    r"""
    Return the points defining this path.

    This returns the raw points in the `d` attribute, ignoring the
    commands that connect these points, i.e., ignoring whether these
    points are connected by `M` commands that do not actually draw
    anything, or any kind of visible curve.
    """
    return [(path[0].start.real, path[0].start.imag)] + [
        (command.end.real, command.end.imag) for command in path
    ]


def _get_transform(element):
    r"""
    Return the transformation needed to bring `element` into the root
    context of the SVG document.

    EXAMPLES::

        >>> from io import StringIO
        >>> svg = SVG(StringIO(r'''
        ... <svg>
        ...   <g transform="translate(10, 10)">
        ...     <text x="0" y="0">curve: 0</text>
        ...   </g>
        ... </svg>'''))
        >>> SVG._get_transform(svg.svg.getElementsByTagName("text")[0])
        array([[ 1.,  0., 10.],
                [ 0.,  1., 10.],
                [ 0.,  0.,  1.]])

    """

    if element is None or element.nodeType == Node.DOCUMENT_NODE:
        return spt_parse_transform(None)

    return _get_transform(element.parentNode).dot(
        spt_parse_transform(element.getAttribute("transform"))
    )


def _svg_transform(element):
    r"""
    Return a transformed version of `element` with all `transform` attributes applied.

    EXAMPLES:

    Transformations can be applied to text elements::

        >>> from io import StringIO
        >>> svg = SVG(StringIO(r'''
        ... <svg>
        ...   <g transform="translate(100, 10)">
        ...     <text x="0" y="0" transform="translate(100, 10)">curve: 0</text>
        ...   </g>
        ... </svg>'''))
        >>> transformed = svg.transform(svg.svg.getElementsByTagName("text")[0])
        >>> transformed.toxml()
        '<text x="200.0" y="20.0">curve: 0</text>'

    Transformations can be applied to paths::

        >>> svg = SVG(StringIO(r'''
        ... <svg>
        ...   <g transform="translate(100, 10)">
        ...     <path d="M 0 0 L 1 1" transform="translate(100, 10)" />
        ...   </g>
        ... </svg>'''))
        >>> svg.transform(svg.svg.getElementsByTagName("path")[0])
        Path(Line(start=(200+20j), end=(201+21j)))

    """
    transformation = _get_transform(element)

    if element.getAttribute("d"):
        # element is like a path
        element = spt_transform(
            spt_parse_path(element.getAttribute("d")), transformation
        )
    elif element.hasAttribute("x") and element.hasAttribute("y"):
        # elements with an explicit location such as <text>
        x = float(element.getAttribute("x"))
        y = float(element.getAttribute("y"))
        x, y, _ = transformation.dot([x, y, 1])

        element = element.cloneNode(deep=True)
        if element.hasAttribute("transform"):
            element.removeAttribute("transform")
        element.setAttribute("x", str(x))
        element.setAttribute("y", str(y))
    else:
        raise NotImplementedError(f"Unsupported element {element}.")

    return element


import uuid


class NpNdarrayEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs = dict(kwargs)
        del self.kwargs["indent"]
        self._replacement_map = {}

    def default(self, o):
        if isinstance(o, np.ndarray):
            key = uuid.uuid4().hex
            self._replacement_map[key] = json.dumps(
                o.tolist(), **self.kwargs, cls=self.__class__
            )
            return "@@%s@@" % (key,)
        elif isinstance(o, SvgPath):
            key = uuid.uuid4().hex
            self._replacement_map[key] = json.dumps(
                o.to_json(), **self.kwargs, cls=self.__class__
            )
            return "@@%s@@" % (key,)
        else:
            return super().default(o)

    def encode(self, o):
        result = super().encode(o)
        for k, v in self._replacement_map.items():
            result = result.replace('"@@%s@@"' % (k,), v)
        return result


class PotentialTick:
    def __init__(self, line):
        self.line = line
        diff = np.abs(line[0] - line[1])
        if diff[0] < EPS:
            self.direction = "horz"
        elif diff[1] < EPS:
            self.direction = "vert"
        else:
            self.direction = None
        self.length = np.linalg.norm(diff)

    @property
    def x_coor(self):
        if self.direction != "horz":
            raise ValueError()
        return self.line[0][0]

    @property
    def y_coor(self):
        if self.direction != "vert":
            raise ValueError()
        return self.line[0][1]

    def __repr__(self):
        return f"<PT@[{self.line[0][0]:.1f},{self.line[0][1]:.1f}] | {self.direction} L={self.length}>"


def get_transform_func(
    coor_min: float, coor_max: float, val_at_min: float, val_at_max: float
) -> float:
    """
    Given a svg canvas coordinate and actual ticks value, returns a transformation function
    """
    _coor_range = coor_max - coor_min
    assert abs(_coor_range) > 0
    _val_range = val_at_max - val_at_min
    assert _val_range > 0

    def _transform(coor):
        return val_at_min + ((coor - coor_min) / _coor_range) * _val_range

    return _transform


class CanvasTicksNoCanidateError(Exception):
    pass


class CanvasTicks:
    def __init__(self, horziontal_ticks, vertical_ticks):
        self.horziontal_ticks = horziontal_ticks
        self.vertical_ticks = vertical_ticks
        self.x_tick_values = None
        self.y_tick_values = None

    def plot(self, fig=None, transform=lambda x: x):
        fig = get_plotly_go(fig)
        for i, tick in enumerate(self.horziontal_ticks):
            _pts = transform(tick.line)
            fig.add_scatter(x=_pts[:, 0], y=_pts[:, 1], name=f"xtick={i+1}")
        for i, tick in enumerate(self.vertical_ticks):
            _pts = transform(tick.line)
            fig.add_scatter(x=_pts[:, 0], y=_pts[:, 1], name=f"ytick={i+1}")
        return fig

    @classmethod
    def from_list_of_lines(cls, list_of_lines):
        return CanvasTicks(*cls.detect_axes_ticks(list_of_lines))

    @classmethod
    def detect_axes_ticks(cls, list_of_lines):
        # assuming ticks must only contains 2 points (maybe deal with colinearity later)
        list_of_lines = filter(lambda x: len(x) == 2, list_of_lines)
        # filter out lines that are straight (vertical or horziontal)
        potential_ticks = [PotentialTick(line) for line in list_of_lines]
        if len(potential_ticks) < 1:
            raise CanvasTicksNoCanidateError()

        # ticks should be short
        _min_len = min(t.length for t in potential_ticks)
        potential_ticks = [t for t in potential_ticks if t.length == _min_len]

        # sort by x
        horziontal_ticks = sorted(
            (t for t in potential_ticks if t.direction == "horz"),
            key=lambda x: x.line[0][0],
        )
        # sort by y
        vertical_ticks = sorted(
            (t for t in potential_ticks if t.direction == "vert"),
            key=lambda x: x.line[0][1],
        )
        # we reverse y tick as SVG coordinate start at top right
        vertical_ticks = list(reversed(vertical_ticks))

        return horziontal_ticks, vertical_ticks

    def assign_axis_ticks_values(self, x_tick_values, y_tick_values):
        assert len(x_tick_values) == len(self.horziontal_ticks)
        assert len(y_tick_values) == len(self.vertical_ticks)
        self.x_tick_values = x_tick_values
        self.y_tick_values = y_tick_values

        x_transform = get_transform_func(
            self.horziontal_ticks[0].x_coor,
            self.horziontal_ticks[-1].x_coor,
            x_tick_values[0],
            x_tick_values[-1],
        )
        y_transform = get_transform_func(
            self.vertical_ticks[0].y_coor,
            self.vertical_ticks[-1].y_coor,
            y_tick_values[0],
            y_tick_values[-1],
        )

        def full_transform(coordinate):
            coordinate = np.asarray(coordinate)
            xs = x_transform(coordinate[..., 0])
            ys = y_transform(coordinate[..., 1])
            return np.stack([xs, ys]).T

        def verify(expected, got, name):
            if (expected - got) > EPS:
                raise ValueError(
                    f"inconsistency in transform @{name}:\n Should be {expected}, but was {got}"
                )

        # verify that it works
        for i in range(1, len(self.horziontal_ticks) - 1):
            _transformed_val = full_transform([self.horziontal_ticks[i].x_coor, 0])[0]
            verify(_transformed_val, x_tick_values[i], f"x{i}")
        for i in range(1, len(self.vertical_ticks) - 1):
            _transformed_val = full_transform([0, self.vertical_ticks[i].y_coor])[1]
            verify(_transformed_val, y_tick_values[i], f"y{i}")

        return full_transform


class SvgPath:
    def __init__(self, path, label=None):
        self._path = path
        self.label = label
        self.transformation_function = None

    def set_transformation_function(self, transformation_function):
        self.set_transformation_function = transformation_function
        return self

    @property
    def far(self):
        text = self.label.x, self.label.y
        endpoints = [self.points[0], self.points[-1]]
        return max(
            endpoints, key=lambda p: (text[0] - p[0]) ** 2 + (text[1] - p[1]) ** 2
        )

    @classmethod
    def path_points(cls, path):
        return [(path[0].start.real, path[0].start.imag)] + [
            (command.end.real, command.end.imag) for command in path
        ]

    @property
    def points(self):
        return np.asarray(path_points(self.path))

    @property
    def path(self):
        return _svg_transform(self._path)

    @classmethod
    def _ensure_output(cls, _str) -> Optional[str]:
        if _str is None or _str.lower() == "none":
            return None
        return _str

    @property
    def color(self) -> Optional[str]:
        if self._path.hasAttribute("stroke"):
            return self._ensure_output(self._path.getAttribute("stroke"))
        return None

    @property
    def fill_color(self) -> Optional[str]:
        out = None
        # is DEFINITELY fill
        if self._path.hasAttribute("fill"):
            out = self._path.getAttribute("fill")

        # is (probably) fill? if the path forms a loop
        elif np.linalg.norm(self.points[0, :] - self.points[0, 1]) < EPS:
            # // use stroke color as fill color
            out = self.color

        return self._ensure_output(out)

    def plot(self, fig=None, **kwargs):
        _kwargs = dict(kwargs)
        fig = get_plotly_go(fig)
        xy = self.points
        if self.fill_color is not None:
            _kwargs["fill"] = "toself"
            _kwargs["mode"] = "none"
            _kwargs["fillcolor"] = rgb_string_add_alpha(self.fill_color, 0.3)
        elif self.color is not None:
            _kwargs["line_color"] = self.color
            _kwargs["mode"] = "lines"
        else:
            _kwargs["line_color"] = rgb_string_add_alpha("rgb(0%,0%,0%)", 0.25)

        fig.add_scatter(x=xy[:, 0], y=xy[:, 1], **_kwargs)
        return fig

    def __repr__(self):
        return (
            f'<Path "{self.label if self.label is not None else self._path.toxml()}">'
        )

    def to_json(self):
        json_dict = {}
        json_dict["type"] = "line"
        if self.fill_color is not None:
            json_dict["type"] = "shaded"
            json_dict["color"] = self.fill_color
        elif self.color is not None:
            json_dict["color"] = self.color

        xy = self.points
        if self.transformation_function is not None:
            xy = self.transformation_function(xy)

        json_dict["x"] = xy[:, 0]
        json_dict["y"] = xy[:, 1]
        return json_dict


class SvgPaths:
    def __init__(self, paths: "List[SvgPath]", label=None):
        paths = [p if isinstance(p, SvgPath) else SvgPath(p, label) for p in paths]
        self._paths = paths
        self.label = label

    def plot(self, fig=None, **kwargs):
        fig = get_plotly_go(fig)
        if len(self) > 1:
            # ensure all sub lines are same color
            if "line_color" not in kwargs:
                kwargs["line_color"] = "blue"
        for p in self:
            p.plot(fig, **kwargs)
        return fig

    @property
    def points(self):
        return [p.points for p in self]

    def __len__(self):
        return len(self._paths)

    def __iter__(self):
        yield from self._paths

    def __repr__(self):
        return f'<Paths "{self.label if self.label is not None else len(self._paths)}">'


def get_all_vector_paths(svg_element):
    _paths = []

    groups = set(path.parentNode for path in svg_element.getElementsByTagName("path"))

    for group in groups:
        label = None
        for child in group.childNodes:
            if child.nodeType == Node.COMMENT_NODE:
                continue

            # if child.nodeType == Node.TEXT_NODE:
            #     if SVG._text_value(child):
            #         pass
            #                         print(
            #                             f'Ignoring unexpected text node "{SVG._text_value(child)}" grouped with <path>.'
            #                         )

            elif child.nodeType == Node.ELEMENT_NODE:
                if child.tagName == "path":
                    continue
                #                     if child.tagName != "text":
                #                         print(
                #                             f"Unexpected <{child.tagName}> grouped with <path>. Ignoring unexpected <{child.tagName}>."
                #                         )
                #                         continue
                if child.tagName == "text":
                    if label is not None:
                        #                         print(
                        #                             f'More than one <text> label associated to this <path>. Ignoring all but the first one, i.e., ignoring "{SVG._text_value(child)}".'
                        #                         )
                        continue
                    label = child

        # Determine all the <path>s in this <g>.
        paths = [
            path
            for path in group.childNodes
            if path.nodeType == Node.ELEMENT_NODE and path.tagName == "path"
        ]
        assert paths

        _paths.append(SvgPaths(paths, label))
    return _paths


def filter_potential_plots(
    all_vector_paths,
    filter_out_multi_paths: bool = False,
    filter_out_black_path: bool = True,
    filter_out_white_fill: bool = True,
):
    grouped_path = defaultdict(list)
    for paths in all_vector_paths:
        if filter_out_multi_paths and len(paths) > 1:
            # these are (likely) ticks
            continue
        for p in paths:
            _stroke_color = p.color
            if _stroke_color:
                if (
                    filter_out_black_path
                    and not np.isclose(
                        rgb_string_extract_val(_stroke_color), np.zeros(3)
                    ).all()
                ):
                    grouped_path[_stroke_color].append(p)
            else:
                _fill_color = p.fill_color
                if _fill_color:
                    if (
                        filter_out_white_fill
                        and not np.isclose(
                            rgb_string_extract_val(_fill_color), np.ones(3) * 100
                        ).all()
                    ):
                        grouped_path[_fill_color].append(p)
    return grouped_path


def format_fig(fig, width=500, height=300, tickwidth=2.4, linewidth=2.4):
    # choose the figure font
    font_dict = dict(
        #     family='Arial',
        size=12,
        color="black",
    )
    # general figure formatting
    fig.update_layout(
        font=font_dict,  # font formatting
        plot_bgcolor="white",  # background color
        width=width,  # figure width
        height=height,  # figure height
        margin=dict(r=20, t=20, b=10),  # remove white space
    )
    _axis_dict = dict(
        showline=True,  # add line at x=0
        linecolor="black",  # line color
        linewidth=linewidth,  # line size
        ticks="outside",  # ticks outside axis
        tickfont=font_dict,  # tick label font
        mirror="allticks",  # add ticks to top/right axes
        tickwidth=tickwidth,  # tick width
        tickcolor="black",  # tick color
    )
    fig.update_yaxes(
        title_text="Y-axis",  # axis label
        **_axis_dict,
    )
    fig.update_xaxes(
        title_text="X-axis",
        **_axis_dict,
    )


def main(args):
    ## from string
    # svg_file = minidom.parseString(svg)

    with open(args.svgfile, "rb") as f:
        svg_file = minidom.parse(f)

    all_vector_paths = get_all_vector_paths(svg_file)

    if args.show_all_lines:
        fig = None
        for paths in all_vector_paths:
            fig = paths.plot(fig)
        fig.show()

    # ticks normall has more than 1 child
    vector_paths_of_tick = [vp for vp in all_vector_paths if len(vp) > 1]
    if len(vector_paths_of_tick) > 0:
        valid_canvas_ticks = []
        for vector_path_of_tick in vector_paths_of_tick:
            try:
                valid_canvas_ticks.append(
                    CanvasTicks.from_list_of_lines(vector_path_of_tick.points)
                )
                break
            except CanvasTicksNoCanidateError:
                # try next canidate
                pass
        if len(valid_canvas_ticks) > 1:
            LOGGER.warning(
                f"vector_paths_of_tick is ambigious as more than 1 was found: {len(valid_canvas_ticks)}"
            )
        canvas_ticks = valid_canvas_ticks[0]

    else:
        # cannot infer ticks.
        canvas_ticks = None

    transformation_function = None

    if args.xticks and args.yticks and canvas_ticks is not None:
        transformation_function = canvas_ticks.assign_axis_ticks_values(
            args.xticks,
            args.yticks,
        )

    if args.show_ticks:
        if canvas_ticks is None:
            LOGGER.error("Cannot find ticks")
            exit(1)
        fig = canvas_ticks.plot()
        # fig = canvas_ticks.plot(fig=fig, transform=transformation_function)
        fig.show()

    grouped_path = filter_potential_plots(all_vector_paths)

    line_plots = []
    for color, paths in grouped_path.items():
        line_plot = dict(
            paths=[
                p.set_transformation_function(transformation_function) for p in paths
            ],
            color=color,
        )

        line_plots.append(line_plot)

    if args.show_grouped_plots:
        fig = get_plotly_go(reverse=transformation_function is None)

        for line_plot in line_plots:
            color = line_plot["color"]
            for path in line_plot["paths"]:
                path.plot(fig)

        fig.update_layout(template="simple_white")
        # fig.update_layout(xaxis_range=[0, 5], yaxis_range=[0, 1e3 + 100])

        # format_fig(fig, width=500, height=250, tickwidth=1.2, linewidth=1.2)
        fig.show()

    if args.save_as_json:
        # Serializing json
        with open(f"{args.ori_name}.json", "w") as outfile:
            outfile.write(json.dumps(line_plots, indent=4, cls=NpNdarrayEncoder))


def run(args):
    args.ori_name = args.svgfile
    if not any(args.svgfile.endswith(ext) for ext in (".pdf", ".svg")):
        LOGGER.error(f"Invalid extension for input file {args.svgfile}")
        exit(1)

    if args.svgfile.endswith(".pdf"):
        with tempfile.NamedTemporaryFile() as temp_file:
            try:
                subprocess.check_call(["pdf2svg", args.svgfile, temp_file.name])
                args.svgfile = temp_file.name
            except subprocess.CalledProcessError as e:
                print(e.returncode)
                exit(e.returncode)
            # process the output as svg
            main(args)


if __name__ == "__main__":
    run(parser.parse_args())
