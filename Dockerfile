FROM nvcr.io/nvidia/tensorflow:18.03-py3

MAINTAINER Florian Kromp <florian.kromp@ccri.at>

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update 
RUN apt-get install -qy libsm6 libxext6 libxrender-dev python3-tk apt-utils openmpi-bin liblzma-dev libfreetype6 libpng16-16 
RUN apt-get install -qy libxmu6 libxt6 #x11-apps
#CMD xclock
RUN pip install --upgrade pip
RUN pip install config

# Packages for tensorflow Mask R-CNN
RUN pip install numpy==1.18.5 scipy==1.1.0 tqdm==4.28.1 opencv-python==3.4.3.18 tifffile==2019.7.26.2 keras==2.1.2 IPython==7.1.1 h5py==2.8.0 pandas==0.24.2 scikit-learn fcsy config scikit-image==0.14.2 holoviews

# add current folder where Dockerfile and requirements are located to workspace
WORKDIR /workspace

# copy sample data to container
COPY ./rawdata /workspace/data

# copy code to container
COPY ./code /workspace/code

# install Matlab runtime
RUN mkdir /workspace/installation_tmp
COPY ./required_prerequisites/MCR_R2013a_glnxa64_installer.zip /workspace/installation_tmp/
WORKDIR /workspace/installation_tmp
RUN unzip MCR_R2013a_glnxa64_installer.zip
RUN ./install -mode silent -agreeToLicense yes

# Remove installation files
RUN rm -r /workspace/installation_tmp/*

# Extract cidre
RUN mkdir /workspace/cidre_tmp
COPY ./required_prerequisites/cidre.tgz /workspace/cidre_tmp/
WORKDIR /workspace/cidre_tmp/
RUN tar xvfz /workspace/cidre_tmp/cidre.tgz

# Create other directories
RUN mkdir /workspace/processing_tmp
RUN mkdir /workspace/cidre_tmp/destination

WORKDIR /workspace