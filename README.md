A C++ Implementation of SMPL - A Skinned Multi-Person Linear Model.

![SMPL_Modle](docs/media/front_page.png)

## Overview

Project based on https://github.com/YeeCY/SMPLpp repository.

This project implements a 3D human skinning model - SMPL: A Skinned
Multi-Person Linear Model with C++. The official SMPL model is available at http://smpl.is.tue.mpg.de.

Project adopted to use with OpenCV library.

## Prerequisites

- OS
  Windows 10
  MSVS 2019 (C17 support)

- Packages

1. [libTorch](https://pytorch.org/get-started/locally/): Pytorch C++ API.    
3. Eigen lineae algebra library
4. [CMake](https://cmake.org/download/): A tool to build, test and pack up 
   C++ program.
5. Opencv 4.x

## Model preprocessing ##

You need to preprocess initial pkl model format to npz using script  SMPL++/scripts/preprocess.py and copy result npz file to exe's folder.

![body](body.jpg)
![hand](hand.jpg)
![head](head.jpg)