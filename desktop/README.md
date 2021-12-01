# Taichi VK Launcher Desktop(for linux)

## Requirements

This repo should be tested with taichi repo.
At this moment, we are using our local taichi, since we are still working on some local changes pending merge.

###  Prepare for taichi repo
1. Clone taichi from https://gitlab.com/innopeaktech-seattle/graphics/taichi
2. Checkout to "dev" branch
3. and set "TAICHI_REPO_DIR" to that directory. e.x. *TAICHI_REPO_DIR=/home/xiaohan/repo/local_taichi/taichi/*
4. Use the command below to build taichi:

     TAICHI_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_OPENGL:BOOL=ON -DTI_EXPORT_CORE:BOOL=ON -DTI_WITH_LLVM:BOOL=OFF" python3 setup.py install --user

5. Add the build result path to LD_LIBRARY_PATH. e.x. export LD_LIBRARY_PATH=$TAICHI_REPO_DIR/build/:$LD_LIBRARY_PATH

## Build & run

Under this directory, run *make* to build the project and run *./taichi-vk-launcher* to run it.
