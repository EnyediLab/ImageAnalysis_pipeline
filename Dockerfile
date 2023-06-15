FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Install base utilities
RUN apt-get update \
    && apt-get install -y python3-pyqt5 \
    # && apt-get install -y pyqt5-dev \
    # && apt-get install -y libqt5multimedia5-plugins\
    # && apt-get install -y libqt5quickcontrols2-5 \
    # && apt-get install -y libqt5multimedia5 \
    # && apt-get install -y libqt5webengine5 \
    # && apt-get install -y libqt5quick5 \
    # && apt-get install -y libqt5qml5 \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

COPY environment.yml .
RUN conda init bash \
    && conda update conda \
    && conda env create --name cp_dock -f environment.yml
RUN echo "conda activate cp_dock"  >> ~/.bashrc
ENV PATH /opt/conda/envs/cp_dock/bin:$PATH
ENV CONDA_DEFAULT_ENV $cp_dock
RUN conda run --no-capture-output -n cp_dock python -m pip install cellpose[gui]
