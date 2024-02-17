#FROM jupyter/datascience-notebook
#FROM jusher/jupyterlab-minimalist:latest
FROM quay.io/jupyter/scipy-notebook
# next try this mltooling/ml-workspace:latest

RUN git clone --depth 1 https://github.com/gerritgr/graphdiffusion.git
RUN rm -rf graphdiffusion/.git && cp -a graphdiffusion/. . &&  rmdir graphdiffusion

RUN mamba env create -f environment_docker.yml -n graphdiffusionenv
RUN /opt/conda/envs/graphdiffusionenv/bin/python -m ipykernel install --user --name=graphdiffusionenv

RUN /opt/conda/envs/graphdiffusionenv/bin/python -m pip install -e .

RUN mamba clean -ya

RUN conda env export --name graphdiffusionenv
RUN conda env export --name graphdiffusionenv >> environment_after_creation.yml


# Set as default
# Remove the default environment to save space (optional)
# RUN conda env remove -n base

# Activate the new environment by default
# This is done by appending the activation command to the .bashrc file
RUN echo "conda activate graphdiffusionenv" >> ~/.bashrc

# Ensure that the environment is activated when you run the container
ENV CONDA_DEFAULT_ENV graphdiffusionenv
