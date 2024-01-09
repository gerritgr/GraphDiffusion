FROM jupyter/datascience-notebook

RUN git clone https://github.com/gerritgr/graphdiffusion.git
RUN rm -rf graphdiffusion/.git 

RUN cd graphdiffusion && mamba env create -f environment.yml -n graphdiffusionenv
RUN /opt/conda/envs/graphdiffusionenv/bin/python -m ipykernel install --user --name=graphdiffusionenv

RUN mamba clean -ya

RUN conda env export --name graphdiffusionenv >> environment.yml