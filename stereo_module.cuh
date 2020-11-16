#ifndef STEREO_MODULE_CUH
#define STEREO_MODULE_CUH

#define USE_CHECK_TIME (0)

#pragma comment(lib,"cudart.lib")

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <device_functions.h>
#include <iostream>

namespace disparity_refinement
{

#define MIN_DISPARITY (5)
#define MAX_DISPARITY (128)

#define MAXIMUM_SLANT_DEGREE (30.f)
#define INVALIDATION_SLANT_DEGREE (75.f)

#define TILE_SIZE (16)

#define MIN_MASK_SIZE_HALF (5)
#define MAX_MASK_SIZE_HALF (15)

#define STANDARD_DISPARITY (13)
#define STANDARD_MASK_SIZE (12)

	__device__
	float tile_disparity(float d, float dx = 0, float dy = 0, float kx = 0, float ky = 0);

	static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

	static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
		if (err == cudaSuccess)
			return;
		std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
		exit(1);
	}

	struct TILE
	{
		float d, dx, dy;
	};	

	class Stereo_module
	{
	public:

		Stereo_module()
		{
		}

		~Stereo_module()
		{
		}

		void create(const int cols, const int rows);
		void process(const unsigned char *left, const unsigned char *right, const float* disp, float* refine);

		void slant_visualization();
		void release();
		

	private:

		dim3 grid_size;
		int stereo_cols, stereo_rows;

		unsigned char *left_image;
		unsigned char *right_image;

		float *init_disparity;
		float *final_disparity;

		TILE *tile;

		uint32_t *left_census;
		uint32_t *right_census;

	};
}

#endif
