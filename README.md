# diffvg

Differentiable Rasterizer for Vector Graphics
https://people.csail.mit.edu/tzumao/diffvg

diffvg is a differentiable rasterizer for 2D vector graphics. See the webpage for more info.

![teaser](https://user-images.githubusercontent.com/951021/92184822-2a0bc500-ee20-11ea-81a6-f26af2d120f4.jpg)

![circle](https://user-images.githubusercontent.com/951021/63556018-0b2ddf80-c4f8-11e9-849c-b4ecfcb9a865.gif)
![ellipse](https://user-images.githubusercontent.com/951021/63556021-0ec16680-c4f8-11e9-8fc6-8b34de45b8be.gif)
![rect](https://user-images.githubusercontent.com/951021/63556028-12ed8400-c4f8-11e9-8072-81702c9193e1.gif)
![polygon](https://user-images.githubusercontent.com/951021/63980999-1e99f700-ca72-11e9-9786-1cba14d2d862.gif)
![curve](https://user-images.githubusercontent.com/951021/64042667-3d9e9480-cb17-11e9-88d8-2f7b9da8b8ab.gif)
![path](https://user-images.githubusercontent.com/951021/64070625-7a52b480-cc19-11e9-9380-eac02f56f693.gif)
![gradient](https://user-images.githubusercontent.com/951021/64898668-da475300-d63c-11e9-917a-825b94be0710.gif)
![circle_outline](https://user-images.githubusercontent.com/951021/65125594-84f7a280-d9aa-11e9-8bc4-669fd2eff2f4.gif)
![ellipse_transform](https://user-images.githubusercontent.com/951021/67149013-06b54700-f25b-11e9-91eb-a61171c6d4a4.gif)

# Debugging running on Macbook Air

Installing

```
git submodule update --init --recursive
python3.7 -m venv ./env
source env/bin/activate
```

If you get

```
urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1091)>
```

Do this:
https://stackoverflow.com/questions/52805115/certificate-verify-failed-unable-to-get-local-issuer-certificate

This doesn't work: (already added into `requirements.txt`)
https://medium.com/@yen.hoang.1904/resolve-issue-ssl-certificate-verify-failed-when-trying-to-open-an-url-with-python-on-macos-46d868b44e10

```
pip install --upgrade certifi
```

Then install

```
pip3 install -r requirements.txt
python setup.py install
```

## Using docker

Install docker on Linux machine

```
https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04
```

Make sure you do git submodule update first before copying the contents to build the image!!! Unless it is already in image.

```
git submodule update --init --recursive
```

### On Macbook Pro:

Mounting the volume with your host aka `-v <host_repo_root_path>:<container_repo_root_path>`

```
docker build . --tag gcc_macbook_pro_diffvg

docker run -v /Users/henryleemr/Documents/workplace/lottie-files/raster-to-vector/diffvecg:/app -it diffvecg
```

If you get this error:

```
???  diffvecg git:(train_handsome) ??? docker build . --tag diffvecg

[+] Building 30.1s (3/3) FINISHED
 => [internal] load build definition from Dockerfile                                    0.1s
 => => transferring dockerfile: 1.62kB                                                  0.1s
 => [internal] load .dockerignore                                                       0.0s
 => => transferring context: 2B                                                         0.0s
 => ERROR [internal] load metadata for docker.io/library/python:3.7-slim               30.0s
------
 > [internal] load metadata for docker.io/library/python:3.7-slim:
------
failed to solve with frontend dockerfile.v0: failed to create LLB definition: failed to do request: Head "https://registry-1.docker.io/v2/library/python/manifests/3.7-slim": Failed to lookup host: registry-1.docker.io
```

From this: https://github.com/docker/buildx/issues/426, do this:

```
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0
```

### On Ubuntu Machine:

```
docker build . --tag gcc
```

Then

```
docker run --name gcc --rm -i -t gcc bash
```

OR if you want to mount:

```
docker run --name gcc -v /home/ubuntu/workspace/pro-diffvecg:/app --rm -i -t gcc bash
```

### Then train

Go into the CLI on Docker Desktop then

```
cd apps
python painterly_original.py imgs/fallingwater.jpg --num_paths 2 --max_width 4.0 --use_lpips_loss

```

Or for debugging:

```
python painterly_rendering.py imgs/line.png --num_paths 4
```

# Install

```
git submodule update --init --recursive
conda install -y pytorch torchvision -c pytorch
conda install -y numpy
conda install -y scikit-image
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite
pip install svgpathtools
pip install cssutils
pip install numba
pip install torch-tools
pip install visdom
python setup.py install
```

# Install using poetry

## prerequisite

install python 3.7, poetry and ffmpeg

```
# install poetry (mac, linux)
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

# install ffmpeg

(macos)
brew install ffmpeg

(linux)
sudo apt install ffmpeg

or use conda
conda install -y -c conda-forge ffmpeg
```

## Install python packages

```
# install all python dependencies
poetry install

# install pydiffvg
poetry run python setup.py install
```

Now to run the apps, just add `poetry run` before each of the commands below, e.g.

```
poetry run python single_circle.py
```

# Building in debug mode

```
python setup.py build --debug install
```

# Run

```
cd apps
```

Optimizing a single circle to a target.

```
python single_circle.py
```

Finite difference comparison.

```
finite_difference_comp.py [-h] [--size_scale SIZE_SCALE]
                               [--clamping_factor CLAMPING_FACTOR]
                               [--use_prefiltering USE_PREFILTERING]
                               svg_file
```

e.g.,

```
python finite_difference_comp.py imgs/tiger.svg
```

Interactive editor

```
python svg_brush.py
```

Painterly rendering

```
painterly_rendering.py [-h] [--num_paths NUM_PATHS]
                       [--max_width MAX_WIDTH] [--use_lpips_loss]
                       [--num_iter NUM_ITER] [--use_blob]
                       target
```

e.g.,

```
python painterly_rendering.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0 --use_lpips_loss
```

Image vectorization

```
python refine_svg.py [-h] [--use_lpips_loss] [--num_iter NUM_ITER] svg target
```

e.g.,

```
python refine_svg.py imgs/flower.svg imgs/flower.jpg
```

Seam carving

```
python seam_carving.py [-h] [--svg SVG] [--optim_steps OPTIM_STEPS]
```

e.g.,

```
python seam_carving.py imgs/hokusai.svg
```

Vector variational autoencoder & vector GAN:

For the GAN models, see `apps/generative_models/train_gan.py`. Generate samples from a pretrained using `apps/generative_models/eval_gan.py`.

For the VAE models, see `apps/generative_models/mnist_vae.py`.

If you use diffvg in your academic work, please cite

```
@article{Li:2020:DVG,
    title = {Differentiable Vector Graphics Rasterization for Editing and Learning},
    author = {Li, Tzu-Mao and Luk\'{a}\v{c}, Michal and Gharbi Micha\"{e}l and Jonathan Ragan-Kelley},
    journal = {ACM Trans. Graph. (Proc. SIGGRAPH Asia)},
    volume = {39},
    number = {6},
    pages = {193:1--193:15},
    year = {2020}
}
```
