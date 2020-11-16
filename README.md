# Realsense disparity refinement with CUDA

I made the disparity refinement project for Intel Realsense D400 series. It's based on the C++ & CUDA.

The environment for testing is below.
- Intel Realsense D435 camera ([848 x 480], [1280 x 720])
- OS : Windows 10
- IDE : Visual studio 2015 community
- CPU : Intel(R) Core(TM) i7-9700K (3.60GHz)
- GPU : Geforce RTX 2080 ti
- RAM : 64 GB

# Dependency for testing
- Opencv 4.1.0
- Realsense SDK 2.24
- SDL2
- glew
- CUDA 10.1

# This method consists of several modules.

- (1) Census transform
- (2) Disparity to tile
- (3) Tile disparity refinement using the parabola fitting.
- (4) Tile slant estimation using the eigen value decomposition.
- (5) per pixel estimation based on the tile.

# Tile slant visualization

![figure 1](https://user-images.githubusercontent.com/23024027/99226134-98abcc80-282c-11eb-96b3-9bc9cb33f949.png)

# Disparity refinement before/after

- Before
![before refinement](https://user-images.githubusercontent.com/23024027/99225965-45d21500-282c-11eb-85b9-198123efc12b.gif)

- After
![after refinement](https://user-images.githubusercontent.com/23024027/99226112-8f226480-282c-11eb-94f0-d56ab29e2052.gif)


# TODO
Please use this method and give me a comment about this.
