# Use an official Ubuntu base image
FROM ubuntu:20.04

# Avoid prompts from apt
ARG DEBIAN_FRONTEND=noninteractive

# Update and install basic packages
RUN apt update && apt install -y \
    curl \
    vim \
    wget \ 
    gcc=9.4.0 \ 
    wget \ 
    gpg \ 
    software-properties-common \
    make \ 
    git 



# Install cmake 
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
    gpg --dearmor - | \
    tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && \ 
    apt update && apt install -y cmake



# Install Python 3.6.13 and pip (>=20)

RUN apt update && apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl git && \
    curl https://pyenv.run | bash && \
    export PATH="$HOME/.pyenv/bin:$PATH" && \
    eval "$(pyenv init --path)" && \
    eval "$(pyenv virtualenv-init -)" && \
    exec "$SHELL" && \
    pyenv install 3.6.13 && \
    pyenv global 3.6.13 && \
    pip install --upgrade --user pip

# Install HPVM 
RUN git clone --recursive -b main https://gitlab.engr.illinois.edu/llvm/hpvm-release.git && \
    cd ./hpvm-release/hpvm/ && \
    ./install.sh -j20 DCMAKE_BUILD_TYPE=Release --no-pypkg 


# Set the working directory
WORKDIR /usr/src/app

# Command to run on container start
CMD ["bash"]
