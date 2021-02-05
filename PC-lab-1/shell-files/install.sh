# Ubuntu 20.10 vm, full install
# user: student
# passwd: student

sudo apt purge ubuntu-web-launchers

# install virtualbox guest additions

sudo apt install \
     curl \
     git \
     cargo \
     libssl-dev \
     libcurl4-openssl-dev \
     libfreetype6-dev \
     libpoppler-cpp-dev \
     libxml2-dev \
     libgit2-dev \
     libgsl-dev \
     libgit2-dev \
     librsvg2-dev \
     libharfbuzz-dev \
     libfribidi-dev \
     libfreetype6-dev \
     libpng-dev \
     libtiff5-dev \
     libudunits2-dev \
     libjpeg-dev \
     libmagick++-dev \
     ripgrep \
     fd-find \
     xcape \
     r-base \
     firefox \
     vim-gtk3 \
     docker.io \
     python3 \
     python3-pip \
     gnome-tweaks \
     gdebi-core

sudo apt clean

pip3 install jupyterlab notebook

# input and language settings

# better kit for emacs
git clone --depth 1 https://github.com/hlissner/doom-emacs ~/.emacs.d
~/.emacs.d/bin/doom install

# install R packages
Rscript install.R

# install Rstudio
wget https://download1.rstudio.org/desktop/bionic/amd64/rstudio-1.3.1073-amd64.deb
sudo gdebi rstudio-server-1.3.1073-amd64.deb

# install vscode
sudo snap install --classic code

# ...
# https://docs.conda.io/en/latest/miniconda.html#linux-installers

