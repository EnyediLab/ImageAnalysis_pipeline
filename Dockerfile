# V1
FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Install base utilities
RUN apt-get update \
    && apt-get install -y python3-pyqt5 \
    && apt-get install -y build-essential \
    && apt-get install -y wget unzip\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Add MCR to path
RUN mkdir /matlab \
    && mkdir /opt/mcr \
    && cd /matlab \
    && wget --no-check-certificate -q https://ssd.mathworks.com/supportfiles/downloads/R2022a/Release/6/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2022a_Update_6_glnxa64.zip \
    && unzip -q MATLAB_Runtime_R2022a_Update_6_glnxa64.zip \
    && chmod +x install \
    && ./install -agreeToLicense yes -destinationFolder /opt/mcr -mode silent
# Remove folder and set path
RUN cd / && rm -rf matlab/

# RUN echo "export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/opt/mscr/v912/runtime/glnxa64:/opt/mcr/v912/bin/glnxa64:/opt/mcr/v912/sys/os/glnxa64:/opt/mcr/v912/sys/opengl/lib/glnxa64" >> ~/.bashrc
# ENV QT_DEBUG_PLUGINS=1
RUN touch ~/matlab_shell.sh && echo "#!/bin/sh" >> ~/matlab_shell.sh \
    && echo "LD_LIBRARY_PATH=/opt/mcr/v912/runtime/glnxa64:/opt/mcr/v912/bin/glnxa64:/opt/mcr/v912/sys/os/glnxa64:/opt/mcr/v912/sys/opengl/lib/glnxa64" >> ~/matlab_shell.sh \
    && echo "exec ${SHELL:-/bin/sh} $*" >> ~/matlab_shell.sh
RUN chmod u+x ~/matlab_shell.sh

# Create conda env and start it right away
COPY environment.yml .
RUN conda init bash \
    && conda update conda \
    && conda env create --name cp_dock -f environment.yml
RUN echo "conda activate cp_dock"  >> ~/.bashrc
ENV PATH /opt/conda/envs/cp_dock/bin:$PATH
ENV CONDA_DEFAULT_ENV $cp_dock

# Install Cellpose and BaxTrack
RUN conda run --no-capture-output -n cp_dock python -m pip install cellpose[gui]
RUN mkdir /opt/baxTrack
COPY for_redistribution_files_only /opt/baxTrack
RUN cd /opt/baxTrack \
    && python setup.py install



# V2
# FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# # set bash as current shell
# RUN chsh -s /bin/bash
# SHELL ["/bin/bash", "-c"]

# # Install base utilities
# RUN apt-get update \
#     && apt-get install -y python3-pyqt5 \
#     && apt-get install -y git \
#     && apt-get install -y build-essential \
#     && apt-get install -y wget \
#     && apt-get install -y unzip \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Install MCR
# # RUN wget --no-check-certificate -q https://ssd.mathworks.com/supportfiles/downloads/R2022a/Release/6/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2022a_Update_6_glnxa64.zip \
# RUN mkdir /matlab \
#     && mkdir /opt/mcr \
#     && cd /matlab \
#     && wget --no-check-certificate -q https://ssd.mathworks.com/supportfiles/downloads/R2022a/Release/6/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2022a_Update_6_glnxa64.zip \
#     && unzip -q MATLAB_Runtime_R2022a_Update_6_glnxa64.zip \
#     && chmod +x install \
#     && ./install -agreeToLicense yes -destinationFolder /opt/mcr -mode silent
# # Remove folder and set path
# RUN cd / && rm -rf matlab/
# # RUN rm -rf /tmp/*
# ENV LD_LIBRARY_PATH=/opt/mcr/v912/runtime/glnxa64:/opt/mcr/v912/bin/glnxa64:/opt/mcr/v912/sys/os/glnxa64:/opt/mcr/v912/sys/opengl/lib/glnxa64

# # Install miniconda
# ENV CONDA_DIR /opt/conda
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#     /bin/bash ~/miniconda.sh -b -p /opt/conda

# # Put conda in path so we can use conda activate
# ENV PATH=$CONDA_DIR/bin:$PATH

# # Create conda env and start it right away
# COPY environment.yml .
# RUN conda init bash \
#     && conda update conda \
#     && conda env create --name cp_dock -f environment.yml
# RUN echo "conda activate cp_dock"  >> ~/.bashrc
# ENV PATH /opt/conda/envs/cp_dock/bin:$PATH
# ENV CONDA_DEFAULT_ENV $cp_dock
# RUN conda run --no-capture-output -n cp_dock python -m pip install cellpose[gui]

# # Install baxTrack
# RUN mkdir /opt/baxTrack
# COPY for_redistribution_files_only /opt/baxTrack
# RUN cd /opt/baxTrack \
#     && python setup.py install