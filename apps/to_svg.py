
import os

from lottie.importers.core import import_tgs
from lottie.importers.svg import import_svg
from lottie.parsers.svg.builder import to_svg


def main():
    filename = "ball"

    input_lottie = filename + ".json"
    output_svg = filename + ".svg"

    lottie_filepath = "/Users/henryleemr/Documents/workplace/lottie-files/raster-to-vector/diffvecg/apps/imgs/"+input_lottie

    # input_svg_filepath = "/Users/henryleemr/Documents/workplace/lottie-files/raster-to-vector/diffvecg/apps/imgs/shapes.json"
    # animation = import_svg(os.path.join(
    #     os.path.dirname(os.path.abspath(__file__)),
    #     input_svg_filepath
    # ))

    animation = import_tgs(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        lottie_filepath
    ))


    time = 0
    elem_tree = to_svg(animation, time, animated=False)

    svg_output_filepath = "/Users/henryleemr/Documents/workplace/lottie-files/raster-to-vector/diffvecg/apps/imgs/" + output_svg
    elem_tree.write(svg_output_filepath)

    print("finish")





if __name__ == "__main__":
    main()
