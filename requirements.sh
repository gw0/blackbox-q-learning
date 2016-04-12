#!/bin/bash
# Requirements installation for blackbox-q-learning
#
# Author: gw0 [http://gw.tnode.com/] <gw.2016@tnode.com>
# License: All rights reserved

NAME="(`basename $(realpath ${0%/*})`)"
SRC="venv/src"
PYTHON_EXE="python2"
SITE_PACKAGES='venv/lib/python*/site-packages'
DIST_PACKAGES='/usr/lib/python*/dist-packages'

# Initialization and essentials
set -e
sudo() { [ -x "/usr/bin/sudo" ] && /usr/bin/sudo "$@" || "$@"; }
sudo apt-get install -y python-pip python-virtualenv git

# Create virtualenv
cd "${0%/*}"
virtualenv --prompt="$NAME" --python="$PYTHON_EXE"  venv || exit 1
source venv/bin/activate
[ ! -e "$SRC" ] && mkdir "$SRC"

# Prerequisites for Theano
sudo apt-get install -y g++ gfortran python-dev libopenblas-dev liblapack-dev
#sudo apt-get install -y python-numpy python-scipy
#[ ! -d $SITE_PACKAGES/numpy ] && cp -a $DIST_PACKAGES/numpy* $SITE_PACKAGES
#[ ! -d $SITE_PACKAGES/scipy ] && cp -a $DIST_PACKAGES/scipy* $SITE_PACKAGES

# Prerequisites for Keras
sudo apt-get install -y python-h5py libyaml-dev graphviz
[ ! -d $SITE_PACKAGES/h5py ] && cp -a $DIST_PACKAGES/h5py* $SITE_PACKAGES

# Prerequisites for matplotlib
sudo apt-get install -y pkg-config libpng-dev libfreetype6-dev

# Requirements
pip install git+https://github.com/Theano/Theano.git
#pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.6.0-cp27-none-linux_x86_64.whl
pip install pydot-ng
pip install git+https://github.com/fchollet/keras.git
pip install pyparsing
pip install matplotlib

wget http://blackboxchallenge.com/static/blackbox.zip
unzip blackbox.zip
git clone https://github.com/EderSantana/X EderSantana-X

echo
echo "Use: . venv/bin/activate"
echo
