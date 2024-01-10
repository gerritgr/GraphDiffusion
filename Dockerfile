FROM jupyter/datascience-notebook

RUN git clone https://github.com/gerritgr/graphdiffusion.git
RUN rm -rf graphdiffusion/.git 

RUN cd graphdiffusion && mamba env create -f environment.yml -n graphdiffusionenv
RUN /opt/conda/envs/graphdiffusionenv/bin/python -m ipykernel install --user --name=graphdiffusionenv

RUN mamba clean -ya

RUN conda env export --name graphdiffusionenv >> environment.yml


# Set as default
# Remove the default environment to save space (optional)
RUN conda env remove -n base

# Activate the new environment by default
# This is done by appending the activation command to the .bashrc file
RUN echo "conda activate graphdiffusionenv" >> ~/.bashrc

# Ensure that the environment is activated when you run the container
ENV CONDA_DEFAULT_ENV graphdiffusionenv