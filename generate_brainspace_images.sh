######################################################
# Generate a Dockerfile and Singularity recipe for building a BrainSpace container
# (https://brainspace.readthedocs.io/en/latest/).
# The Dockerfile and/or Singularity recipe installs most of BrainSpace's dependencies.
#
# Steps to build, upload, and deploy the BrainSpace docker and/or singularity image:
#
# 1. Create or update the Dockerfile and Singuarity recipe:
# bash generate_brainspace_images.sh
#
# 2. Build the docker image:
# docker build -t brainspace -f Dockerfile .
# OR
# bash generate_brainspace_images.sh docker
#
#    and/or singularity image:
# singularity build mindboggle.simg Singularity
# OR
# bash generate_brainspace_images.sh singularity
#
#   and/or both:
# bash generate_brainspace_images.sh both
#
# 3. Push to Docker hub:
# (https://docs.docker.com/docker-cloud/builds/push-images/)
# export DOCKER_ID_USER="your_docker_id"
# docker login
# docker tag brainspace your_docker_id/brainspace:tag  # See: https://docs.docker.com/engine/reference/commandline/tag/
# docker push your_docker_id/brainspace:tag
#
# 4. Pull from Docker hub (or use the original):
# docker pull your_docker_id/brainspace
#
# In the following, the Docker container can be the original (brainspace)
# or the pulled version (ypur_docker_id/brainspace:tag), and is given access to /Users/brainspace
# on the host machine.
#
# 5. Enter the bash shell of the Docker container, and add port mappings:
# docker run --rm -ti -v /Users/brainspace:/home/brainspace -p 8888:8888 -p 5000:5000 your_docker_id/brainspace
#
#
###############################################################################

image="kaczmarj/neurodocker:0.6.0"

set -e

generate_docker() {
 docker run --rm ${image} generate docker \
            --base ubuntu:latest \
            --pkg-manager apt \
            --run-bash 'apt-get update' \
            --install git libsm6 libxext6 libgl1-mesa-dev libvtk6.3 xvfb\
            --user=root \
            --run-bash "curl https://raw.githubusercontent.com/PeerHerholz/BrainSpace/master/requirements.txt > requirements.txt && chmod 777 requirements.txt"\
            --user=brainspace \
            --miniconda \
               conda_install="python=3.7 panel pyqt pyvista notebook ipython" \
               pip_install='-r requirements.txt git+https://github.com/PeerHerholz/BrainSpace.git@notebook_binder_support xvfbwrapper ipywidgets ipyevents jupytext seaborn' \
               create_env='brainspace' \
               activate=true \
            --run 'mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \"0.0.0.0\" > ~/.jupyter/jupyter_notebook_config.py' \
            --entrypoint="/neurodocker/startup.sh" \
            --workdir "/opt/miniconda-latest/envs/brainspace/lib/python3.7/site-packages/brainspace/examples" \
            --add-to-entrypoint='jupytext --set-formats ipynb,py *.py && rm *.ipynb' \
            --cmd jupyter notebook
}

generate_singularity() {
  docker run --rm ${image} generate singularity \
              --base ubuntu:latest \
              --pkg-manager apt \
              --run-bash 'apt-get update' \
              --install  git libsm6 libxext6 libgl1-mesa-dev libvtk6.3 xvfb\
              --user=root \
              --run-bash "curl https://raw.githubusercontent.com/PeerHerholz/BrainSpace/master/requirements.txt > requirements.txt && chmod 777 requirements.txt"\
              --user=brainspace \
              --miniconda \
                 conda_install="python=3.7 panel pyqt pyvista notebook ipython" \
                 pip_install='-r requirements.txt git+https://github.com/PeerHerholz/BrainSpace.git@notebook_binder_support xvfbwrapper ipywidgets ipyevents jupytext seaborn' \
                 create_env='brainspace' \
                 activate=true \
              --run 'mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \"0.0.0.0\" > ~/.jupyter/jupyter_notebook_config.py' \
              --entrypoint="/neurodocker/startup.sh" \
              --workdir "/opt/miniconda-latest/envs/brainspace/lib/python3.7/site-packages/brainspace/examples" \
              --add-to-entrypoint='jupytext --set-formats ipynb,py *.py && rm *.ipynb'
 }

# generate files
generate_docker > Dockerfile
generate_singularity > Singularity

# check if images should be build locally or not
if [ '$1' = 'docker' ]; then
 echo "docker image will be build locally"
 # build image using the saved files
 docker build -t brainspace .
elif [ '$1' = 'singularity' ]; then
 echo "singularity image will be build locally"
 # build image using the saved files
 singularity build brainspace.simg Singularity
elif [ '$1' = 'both' ]; then
 echo "docker and singularity images will be build locally"
 # build images using the saved files
 docker build -t brainspace .
 singularity build brainspace.simg Singularity
else
echo "Image(s) won't be build locally."
fi
