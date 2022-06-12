"""
Scream: python handsome.py imgs/scream.jpg --num_paths 2048 --max_width 4.0
Fallingwater: python handsome.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0
Fallingwater: python handsome.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0 --use_lpips_loss
Baboon: python handsome.py imgs/baboon.png --num_paths 1024 --max_width 4.0 --num_iter 250
Baboon Lpips: python handsome.py imgs/baboon.png --num_paths 1024 --max_width 4.0 --num_iter 500 --use_lpips_loss
Kitty: python handsome.py imgs/kitty.jpg --num_paths 1024 --use_blob

Example command to run this version: (--use_blob is required for this version, as strokes are not present in the
# shape group when using pydiffvg.svg_to_scene(svg_filepath))
python painterly_rendering.py imgs/puke.png --num_iter 101 --num_paths=512 --use_blob 


"""
import argparse
import math
import os
import random
from datetime import datetime

import pydiffvg
import skimage
import skimage.io
import torch
import ttools.modules

pydiffvg.set_print_timing(True)

gamma = 1.0



def test():
    # p= random.randint(1, 3)
    # a = torch.zeros(2, dtype = torch.int32) + 2
    # print(a)

    canvas_width = 500
    canvas_height = 600
    num_paths = 1

    for i in range(num_paths):
            num_segments = random.randint(1, 3)
            print(num_segments)
            num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
            print(num_control_points)
            points = []
            p0 = (random.random(), random.random())
            points.append(p0)
            print(points)
            for j in range(num_segments):
                radius = 0.05
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                print("p1: ", p1)
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                print("p2: ", p2)
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                print("p3: ", p3)
                points.append(p1)
                points.append(p2)
                points.append(p3)
                p0 = p3
            points = torch.tensor(points)
            print("points: ", points)
            points[:, 0] *= canvas_width
            points[:, 1] *= canvas_height

    print("FINISHED")


def setup_results_dir(args):
    # Create path based on target filename
    target_filepath = args.target
    filename_with_ext = os.path.basename(target_filepath)
    filename =  os.path.splitext(filename_with_ext)[0]

    output_filename = filename + "__" + \
    "num_paths_" + str(args.num_paths) + "__" + \
    "max_width" + str(args.max_width) + "__" + \
    "use_lpips_loss" + str(args.use_lpips_loss) + "__" + \
    "num_iter" + str(args.num_iter) + "__" + \
    "use_blob" + str(args.use_blob)

    results_path = "results/" + output_filename

    # Check whether the specified path exists or not
    isExist = os.path.exists(results_path)
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(results_path)
        
    print("Saving to folder: ", results_path)

    # Make a file with the arguments for this run
    # dd/mm/YY H:M:S
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    arguments_filepath = results_path + '/arguments.txt'
    with open(arguments_filepath, 'w') as f:
        f.write("Start: " + str(dt_string) + "\n")
        f.write(str(args) + "\n")


    return results_path, arguments_filepath


def main(args):
    results_path, arguments_filepath = setup_results_dir(args)

    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    
    perception_loss = ttools.modules.LPIPS().to(pydiffvg.get_device())

    #target = torch.from_numpy(skimage.io.imread('imgs/lena.png')).to(torch.float32) / 255.0
    target = skimage.io.imread(args.target)
    # If the image is [,,4] then take only [,,3]
    target = target[:,:,:3]
    target = torch.from_numpy(target).to(torch.float32) / 255.0
    target = target.pow(gamma)
    target = target.to(pydiffvg.get_device())
    target = target.unsqueeze(0)
    target = target.permute(0, 3, 1, 2) # NHWC -> NCHW
    #target = torch.nn.functional.interpolate(target, size = [256, 256], mode = 'area')
    canvas_width, canvas_height = target.shape[3], target.shape[2]

    num_paths = args.num_paths
    max_width = args.max_width
    
    random.seed(1234)
    torch.manual_seed(1234)

    ################################################################################################################################ 
    # shapes = []
    # shape_groups = []
    # if args.use_blob:
    #     for i in range(num_paths):
    #         num_segments = random.randint(3, 5)
    #         num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
    #         points = []
    #         p0 = (random.random(), random.random())
    #         points.append(p0)
    #         for j in range(num_segments):
    #             radius = 0.05
    #             p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
    #             p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
    #             p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
    #             points.append(p1)
    #             points.append(p2)
    #             if j < num_segments - 1:
    #                 points.append(p3)
    #                 p0 = p3
    #         points = torch.tensor(points)
    #         points[:, 0] *= canvas_width
    #         points[:, 1] *= canvas_height
    #         path = pydiffvg.Path(num_control_points = num_control_points,
    #                              points = points,
    #                              stroke_width = torch.tensor(1.0),
    #                              is_closed = True)
    #         shapes.append(path)
    #         path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
    #                                          fill_color = torch.tensor([random.random(),
    #                                                                     random.random(),
    #                                                                     random.random(),
    #                                                                     random.random()]))
    #         shape_groups.append(path_group)
    # else:
    #     for i in range(num_paths):
    #         num_segments = random.randint(1, 3)
    #         num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
    #         points = []
    #         p0 = (random.random(), random.random())
    #         points.append(p0)
    #         for j in range(num_segments):
    #             radius = 0.05
    #             p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
    #             p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
    #             p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
    #             points.append(p1)
    #             points.append(p2)
    #             points.append(p3)
    #             p0 = p3
    #         points = torch.tensor(points)
    #         points[:, 0] *= canvas_width
    #         points[:, 1] *= canvas_height
    #         #points = torch.rand(3 * num_segments + 1, 2) * min(canvas_width, canvas_height)
    #         path = pydiffvg.Path(num_control_points = num_control_points,
    #                             points = points,
    #                             stroke_width = torch.tensor(1.0),
    #                             is_closed = False)
    #         shapes.append(path)
    #         path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
    #                                         fill_color = None,
    #                                         stroke_color = torch.tensor([random.random(),
    #                                                                     random.random(),
    #                                                                     random.random(),
    #                                                                     random.random()]))
    #         shape_groups.append(path_group)
        
    #     scene_args = pydiffvg.RenderFunction.serialize_scene(\
    #         canvas_width, canvas_height, shapes, shape_groups)
        
    #     render = pydiffvg.RenderFunction.apply
    #     img = render(canvas_width, # width
    #                 canvas_height, # height
    #                 2,   # num_samples_x
    #                 2,   # num_samples_y
    #                 0,   # seed
    #                 None,
    #                 *scene_args)

    ################################################################################################################################ 
    # Create the initialisation from an svg seed
    svg_filepath = "./imgs/puke_untuned.svg"
    _canvas_width, _canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_filepath)
    assert(canvas_width==_canvas_width)
    assert(canvas_height==_canvas_height)
    
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    pydiffvg.imwrite(img.cpu(), results_path+'/init.png', gamma=gamma)


    points_vars = []
    stroke_width_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
    if not args.use_blob:
        for path in shapes:
            path.stroke_width.requires_grad = True
            stroke_width_vars.append(path.stroke_width)
    if args.use_blob:
        for group in shape_groups:
            group.fill_color.requires_grad = True
            color_vars.append(group.fill_color)
    # else:
        # print("HERE3")
        # for group in shape_groups:
            # print("STROKE COLOR: ", group.stroke_color)
            # group.stroke_color.requires_grad = True
            # color_vars.append(group.stroke_color)

    
    # Optimize
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    if len(stroke_width_vars) > 0:
        width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)

    color_optim = torch.optim.Adam(color_vars, lr=0.01)

    # Adam iterations.
    for t in range(args.num_iter):
        print('iteration:', t)
        points_optim.zero_grad()
        if len(stroke_width_vars) > 0:
            width_optim.zero_grad()

        color_optim.zero_grad()

        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, # width
                     canvas_height, # height
                     2,   # num_samples_x
                     2,   # num_samples_y
                     t,   # seed
                     None,
                     *scene_args)
        # Compose img with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        # Save the intermediate render.
        pydiffvg.imwrite(img.cpu(), results_path+'/iter_{}.png'.format(t), gamma=gamma)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
        if args.use_lpips_loss:
            loss = perception_loss(img, target) + (img.mean() - target.mean()).pow(2)
        else:
            loss = (img - target).pow(2).mean()
        print('render loss:', loss.item())
    
        # Backpropagate the gradients.
        loss.backward()

        # Take a gradient descent step.
        points_optim.step()
        if len(stroke_width_vars) > 0:
            width_optim.step()

        color_optim.step()
        
        if len(stroke_width_vars) > 0:
            for path in shapes:
                path.stroke_width.data.clamp_(1.0, max_width)
        if args.use_blob:
            for group in shape_groups:
                group.fill_color.data.clamp_(0.0, 1.0)
        
        else:
            for group in shape_groups:
                group.stroke_color.data.clamp_(0.0, 1.0)


        if t % 10 == 0 or t == args.num_iter - 1:
            pydiffvg.save_svg(results_path+'/iter_{}.svg'.format(t),
                              canvas_width, canvas_height, shapes, shape_groups)
    
    # Render the final result.
    img = render(target.shape[1], # width
                 target.shape[0], # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), results_path+'/final.png'.format(t), gamma=gamma)

    # Log time
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    with open(arguments_filepath, 'a') as f:
        # f.write("\n")
        f.write("End: " + str(dt_string))


    # Convert the intermediate renderings to a video.
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i",
        results_path+"/iter_%d.png", "-vb", "20M",
        results_path+"/out.mp4"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="target image path")
    parser.add_argument("--num_paths", type=int, default=512)
    parser.add_argument("--max_width", type=float, default=2.0)
    parser.add_argument("--use_lpips_loss", dest='use_lpips_loss', action='store_true')
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--use_blob", dest='use_blob', action='store_true')
    args = parser.parse_args()
    main(args)
