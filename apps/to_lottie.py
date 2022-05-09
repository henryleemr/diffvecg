#!/usr/bin/env python3
import sys
import os
from lottie.importers.svg import import_svg
from lottie.exporters.core import export_lottie


def to_lottie(svg_filepath, lottie_filepath):
    animation = import_svg(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        svg_filepath
    ))
    export_lottie(animation,lottie_filepath)
    print("Wrote lottie to: ", lottie_filepath)


if __name__ == "__main__":
    svg_filepath =  "imgs/circle.svg"
    lottie_filepath = "output_lottie.json"
    to_lottie(svg_filepath,lottie_filepath)