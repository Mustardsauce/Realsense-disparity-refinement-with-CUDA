#include "stereo_module.cuh"

#include <fstream>
#include <time.h>

#include <opencv2/highgui/highgui.hpp>

#include <thrust/device_vector.h>
#include <thrust/sort.h>


namespace disparity_refinement
{
	
// do no touch
#define CENSUS_WIDTH (9)
#define CENSUS_HEIGHT (7)
#define LINES_PER_BLOCK (16)

#define BLOCK_SIZE (128)

static inline int divUp(int total, int grain) {
	return (total + grain - 1) / grain;
}

__device__ __forceinline__
float get_radian(float degree)
{
	return degree * ((3.141592f) / 180.f);
}

__device__ __forceinline__
float3 f3_sub(const float3& f1, const float3& f2) {
	return make_float3(f1.x - f2.x, f1.y - f2.y, f1.z - f2.z);
}

__device__ __forceinline__
float3 f3_add(const float3& f1, const float3& f2) {
	return make_float3(f2.x + f1.x, f2.y + f1.y, f2.z + f1.z);
}

__device__ __forceinline__
float3 f3_mul_scalar(const float& scalar, const float3& vec) {
	return make_float3(vec.x * scalar, vec.y * scalar, vec.z * scalar);
}

__device__ __forceinline__
float3 f3_div_elem(const float3& f1, const float3& f2) {
	return make_float3(f1.x / f2.x, f1.y / f2.y, f1.z / f2.z);
}

__device__ __host__ __forceinline__
float3 f3_div_elem(const float3& f, const dim3& i) {
	return make_float3(f.x / i.x, f.y / i.y, f.z / i.z);
}

__device__ __host__ __forceinline__
float3 f3_div_elem(const float3& f, const int& i) {
	return f3_div_elem(f, dim3(i, i, i));
}

__device__ __forceinline__
float3 f3_normalize(const float3& vec) {
	const float l = sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);

	if (l == 0)
		return make_float3(0, 0, 0);
	else
		return make_float3(vec.x / l, vec.y / l, vec.z / l);
}

__device__ __forceinline__
float f3_inner_product(const float3& vec1, const float3& vec2) {
	return (vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z);
}

__device__ __forceinline__
float3 f3_cross_product(const float3& vec1, const float3& vec2) {
	return make_float3(vec1.y * vec2.z - vec1.z * vec2.y
		, vec1.z * vec2.x - vec1.x * vec2.z
		, vec1.x * vec2.y - vec1.y * vec2.x);
}

__device__ __forceinline__
float3 f3_mul_elem(const float3& f, const float3& i) {
	return make_float3(f.x * i.x, f.y * i.y, f.z * i.z);
}

__device__ __forceinline__
float tile_disparity(float d, float dx, float dy, float kx, float ky)
{
	return (kx * dx) + (ky * dy) - (d);
}



__device__
float middleOfThree(float a, float b, float c)
{
	// Compare each three number to find middle  
	// number. Enter only if a > b 
	if (a > b)
	{
		if (b > c)
			return b;
		else if (a > c)
			return c;
		else
			return a;
	}
	else
	{
		// Decided a is not greater than b. 
		if (a > c)
			return a;
		else if (b > c)
			return c;
		else
			return b;
	}
}

__global__
void census_transform_kernel(
	uint32_t *dest,
	const uchar *src,
	int width,
	int height)
{
	using pixel_type = uchar;
	static const int SMEM_BUFFER_SIZE = CENSUS_HEIGHT + 1;

	const int half_kw = CENSUS_WIDTH / 2;
	const int half_kh = CENSUS_HEIGHT / 2;

	__shared__ pixel_type smem_lines[SMEM_BUFFER_SIZE][BLOCK_SIZE];

	const int tid = threadIdx.x;
	const int x0 = blockIdx.x * (BLOCK_SIZE - CENSUS_WIDTH + 1) - half_kw;
	const int y0 = blockIdx.y * LINES_PER_BLOCK;

	for (int i = 0; i < CENSUS_HEIGHT; ++i) {
		const int x = x0 + tid, y = y0 - half_kh + i;
		pixel_type value = 0;
		if (0 <= x && x < width && 0 <= y && y < height) {
			value = src[x + y * width];
		}
		smem_lines[i][tid] = value;
	}
	__syncthreads();

#pragma unroll
	for (int i = 0; i < LINES_PER_BLOCK; ++i) {
		if (i + 1 < LINES_PER_BLOCK) {
			// Load to smem
			const int x = x0 + tid, y = y0 + half_kh + i + 1;
			pixel_type value = 0;
			if (0 <= x && x < width && 0 <= y && y < height) {
				value = src[x + y * width];
			}
			const int smem_x = tid;
			const int smem_y = (CENSUS_HEIGHT + i) % SMEM_BUFFER_SIZE;
			smem_lines[smem_y][smem_x] = value;
		}

		if (half_kw <= tid && tid < BLOCK_SIZE - half_kw) {
			// Compute and store
			const int x = x0 + tid, y = y0 + i;
			if (half_kw <= x && x < width - half_kw && half_kh <= y && y < height - half_kh) {
				const int smem_x = tid;
				const int smem_y = (half_kh + i) % SMEM_BUFFER_SIZE;
				uint32_t f = 0;
				for (int dy = -half_kh; dy < 0; ++dy) {
					const int smem_y1 = (smem_y + dy + SMEM_BUFFER_SIZE) % SMEM_BUFFER_SIZE;
					const int smem_y2 = (smem_y - dy + SMEM_BUFFER_SIZE) % SMEM_BUFFER_SIZE;
					for (int dx = -half_kw; dx <= half_kw; ++dx) {
						const int smem_x1 = smem_x + dx;
						const int smem_x2 = smem_x - dx;
						const auto a = smem_lines[smem_y1][smem_x1];
						const auto b = smem_lines[smem_y2][smem_x2];
						f = (f << 1) | (a > b);
					}
				}
				for (int dx = -half_kw; dx < 0; ++dx) {
					const int smem_x1 = smem_x + dx;
					const int smem_x2 = smem_x - dx;
					const auto a = smem_lines[smem_y][smem_x1];
					const auto b = smem_lines[smem_y][smem_x2];
					f = (f << 1) | (a > b);
				}
				dest[x + y * width] = f;
			}
		}
		__syncthreads();
	}
}

__device__ __forceinline__ 
void computeRoots2(const float& b, const float& c, float3& roots)
{
	roots.x = 0.f;
	float d = b * b - 4.f * c;
	if (d < 0.f) // no real roots!!!! THIS SHOULD NOT HAPPEN!
		d = 0.f;

	float sd = sqrtf(d);

	roots.z = 0.5f * (b + sd);
	roots.y = 0.5f * (b - sd);
}

__device__ __forceinline__ 
void swap(float& a, float& b)
{
	const float temp = a;
	a = b;
	b = temp;
}

__device__ __forceinline__ 
void computeRoots3(float c0, float c1, float c2, float3& roots)
{
	if (abs(c0) < 1.192092896e-07F)// one root is 0 -> quadratic equation
	{
		computeRoots2(c2, c1, roots);
	}
	else
	{
		const float s_inv3 = 1.f / 3.f;
		const float s_sqrt3 = sqrtf(3.f);
		// Construct the parameters used in classifying the roots of the equation
		// and in solving the equation for the roots in closed form.
		float c2_over_3 = c2 * s_inv3;
		float a_over_3 = (c1 - c2*c2_over_3)*s_inv3;
		if (a_over_3 > 0.f)
			a_over_3 = 0.f;

		float half_b = 0.5f * (c0 + c2_over_3 * (2.f * c2_over_3 * c2_over_3 - c1));

		float q = half_b * half_b + a_over_3 * a_over_3 * a_over_3;
		if (q > 0.f)
			q = 0.f;

		// Compute the eigenvalues by solving for the roots of the polynomial.
		float rho = sqrtf(-a_over_3);
		float theta = atan2(sqrtf(-q), half_b)*s_inv3;
		float cos_theta = __cosf(theta);
		float sin_theta = __sinf(theta);
		roots.x = c2_over_3 + 2.f * rho * cos_theta;
		roots.y = c2_over_3 - rho * (cos_theta + s_sqrt3 * sin_theta);
		roots.z = c2_over_3 - rho * (cos_theta - s_sqrt3 * sin_theta);

		// Sort in increasing order.
		if (roots.x >= roots.y)
			swap(roots.x, roots.y);

		if (roots.y >= roots.z)
		{
			swap(roots.y, roots.z);

			if (roots.x >= roots.y)
				swap(roots.x, roots.y);
		}
		if (roots.x <= 0) // eigenval for symmetric positive semi-definite matrix can not be negative! Set it to 0
			computeRoots2(c2, c1, roots);
	}
}

__device__  __forceinline__ 
static bool isMuchSmallerThan(float x, float y)
{
	// copied from <eigen>/include/Eigen/src/Core/NumTraits.h
	const float prec_sqr = 1.192092896e-07F * 1.192092896e-07F;
	return x * x <= prec_sqr * y * y;
}

__forceinline__ __device__ 
static float3
unitOrthogonal(const float3& src)
{
	float3 perp;
	/* Let us compute the crossed product of *this with a vector
	* that is not too close to being colinear to *this.
	*/

	/* unless the x and y coords are both close to zero, we can
	* simply take ( -y, x, 0 ) and normalize it.
	*/
	if (!isMuchSmallerThan(src.x, src.z) || !isMuchSmallerThan(src.y, src.z))
	{
		float invnm = rsqrtf(src.x*src.x + src.y*src.y);
		perp.x = -src.y * invnm;
		perp.y = src.x * invnm;
		perp.z = 0.0f;
	}
	/* if both x and y are close to zero, then the vector is close
	* to the z-axis, so it's far from colinear to the x-axis for instance.
	* So we take the crossed product with (1,0,0) and normalize it.
	*/
	else
	{
		float invnm = rsqrtf(src.z * src.z + src.y * src.y);
		perp.x = 0.0f;
		perp.y = -src.z * invnm;
		perp.z = src.y * invnm;
	}

	return perp;
}

__device__
void solve_eigen_decomposition(float cov[6], float evecs[9], float3& evals)
{
	float max01 = fmaxf(abs(cov[0]), abs(cov[1]));
	float max23 = fmaxf(abs(cov[2]), abs(cov[3]));
	float max45 = fmaxf(abs(cov[4]), abs(cov[5]));
	float m0123 = fmaxf(max01, max23);
	float scale = fmaxf(max45, m0123);

	if (scale <= FLT_MIN)
		scale = 1.f;

	cov[0] /= scale;
	cov[1] /= scale;
	cov[2] /= scale;
	cov[3] /= scale;
	cov[4] /= scale;
	cov[5] /= scale;

	float c0 = cov[0] * cov[3] * cov[5]
		+ 2.f * cov[1] * cov[2] * cov[4]
		- cov[0] * cov[4] * cov[4]
		- cov[3] * cov[2] * cov[2]
		- cov[5] * cov[1] * cov[1];

	float c1 = cov[0] * cov[3] -
		cov[1] * cov[1] +
		cov[0] * cov[5] -
		cov[2] * cov[2] +
		cov[3] * cov[5] -
		cov[4] * cov[4];

	float c2 = cov[0] + cov[3] + cov[5];

	computeRoots3(c0, c1, c2, evals);

	if (evals.z - evals.x <= 1.192092896e-07F)
	{
		evecs[0] = evecs[4] = evecs[8] = 1.f;
		evecs[1] = evecs[2] = evecs[3] = evecs[5] = evecs[6] = evecs[7] = 0.f;
	}
	else if (evals.y - evals.x <= 1.192092896e-07F)
	{
		float3 row_tmp[3];
		row_tmp[0] = make_float3(cov[0] - evals.z, cov[1], cov[2]);
		row_tmp[1] = make_float3(cov[1], cov[3] - evals.z, cov[4]);
		row_tmp[2] = make_float3(cov[2], cov[4], cov[5] - evals.z);

		float3 vec_tmp_0 = f3_cross_product(row_tmp[0], row_tmp[1]);
		float3 vec_tmp_1 = f3_cross_product(row_tmp[0], row_tmp[2]);
		float3 vec_tmp_2 = f3_cross_product(row_tmp[1], row_tmp[2]);

		float len1 = f3_inner_product(vec_tmp_0, vec_tmp_0);
		float len2 = f3_inner_product(vec_tmp_1, vec_tmp_1);
		float len3 = f3_inner_product(vec_tmp_2, vec_tmp_2);

		if (len1 >= len2 && len1 >= len3)
		{
			const float sqr_len = rsqrtf(len1);

			evecs[6] = vec_tmp_0.x * sqr_len;
			evecs[7] = vec_tmp_0.y * sqr_len;
			evecs[8] = vec_tmp_0.z * sqr_len;

			//evecs[2] = vec_tmp[0] * rsqrtf(len1);
		}
		else if (len2 >= len1 && len2 >= len3)
		{
			const float sqr_len = rsqrtf(len2);

			evecs[6] = vec_tmp_1.x * sqr_len;
			evecs[7] = vec_tmp_1.y * sqr_len;
			evecs[8] = vec_tmp_1.z * sqr_len;

			//evecs[2] = vec_tmp[1] * rsqrtf(len2);
		}
		else
		{
			const float sqr_len = rsqrtf(len3);

			evecs[6] = vec_tmp_2.x * sqr_len;
			evecs[7] = vec_tmp_2.y * sqr_len;
			evecs[8] = vec_tmp_2.z * sqr_len;

			//evecs[2] = vec_tmp[2] * rsqrtf(len3);
		}

		float3 evecs_2 = make_float3(evecs[6], evecs[7], evecs[8]);

		float3 evecs_1 = unitOrthogonal(evecs_2);

		evecs[3] = evecs_1.x;
		evecs[4] = evecs_1.y;
		evecs[5] = evecs_1.z;

		float3 evecs_0 = f3_cross_product(evecs_1, evecs_2);

		evecs[0] = evecs_0.x;
		evecs[1] = evecs_0.y;
		evecs[2] = evecs_0.z;

	}
	else if (evals.z - evals.y <= 1.192092896e-07F)
	{
		float3 row_tmp[3];
		row_tmp[0] = make_float3(cov[0] - evals.x, cov[1], cov[2]);
		row_tmp[1] = make_float3(cov[1], cov[3] - evals.x, cov[4]);
		row_tmp[2] = make_float3(cov[2], cov[4], cov[5] - evals.x);

		float3 vec_tmp_0 = f3_cross_product(row_tmp[0], row_tmp[1]);
		float3 vec_tmp_1 = f3_cross_product(row_tmp[0], row_tmp[2]);
		float3 vec_tmp_2 = f3_cross_product(row_tmp[1], row_tmp[2]);

		float len1 = f3_inner_product(vec_tmp_0, vec_tmp_0);
		float len2 = f3_inner_product(vec_tmp_1, vec_tmp_1);
		float len3 = f3_inner_product(vec_tmp_2, vec_tmp_2);

		if (len1 >= len2 && len1 >= len3)
		{
			const float sqr_len = rsqrtf(len1);

			evecs[0] = vec_tmp_0.x * sqr_len;
			evecs[1] = vec_tmp_0.y * sqr_len;
			evecs[2] = vec_tmp_0.z * sqr_len;

			//evecs[0] = vec_tmp[0] * rsqrtf(len1);
		}
		else if (len2 >= len1 && len2 >= len3)
		{
			const float sqr_len = rsqrtf(len2);

			evecs[0] = vec_tmp_1.x * sqr_len;
			evecs[1] = vec_tmp_1.y * sqr_len;
			evecs[2] = vec_tmp_1.z * sqr_len;

			//evecs[0] = vec_tmp[1] * rsqrtf(len2);
		}
		else
		{
			const float sqr_len = rsqrtf(len3);

			evecs[0] = vec_tmp_2.x * sqr_len;
			evecs[1] = vec_tmp_2.y * sqr_len;
			evecs[2] = vec_tmp_2.z * sqr_len;

			//evecs[0] = vec_tmp[2] * rsqrtf(len3);
		}

		float3 evecs_0 = make_float3(evecs[0], evecs[1], evecs[2]);

		float3 evecs_1 = unitOrthogonal(evecs_0);

		evecs[3] = evecs_1.x;
		evecs[4] = evecs_1.y;
		evecs[5] = evecs_1.z;

		float3 evecs_2 = f3_cross_product(evecs_0, evecs_1);

		evecs[6] = evecs_2.x;
		evecs[7] = evecs_2.y;
		evecs[8] = evecs_2.z;

	}
	else
	{
		float3 row_tmp[3];
		row_tmp[0] = make_float3(cov[0] - evals.z, cov[1], cov[2]);
		row_tmp[1] = make_float3(cov[1], cov[3] - evals.z, cov[4]);
		row_tmp[2] = make_float3(cov[2], cov[4], cov[5] - evals.z);

		float3 vec_tmp_0 = f3_cross_product(row_tmp[0], row_tmp[1]);
		float3 vec_tmp_1 = f3_cross_product(row_tmp[0], row_tmp[2]);
		float3 vec_tmp_2 = f3_cross_product(row_tmp[1], row_tmp[2]);

		float len1 = f3_inner_product(vec_tmp_0, vec_tmp_0);
		float len2 = f3_inner_product(vec_tmp_1, vec_tmp_1);
		float len3 = f3_inner_product(vec_tmp_2, vec_tmp_2);

		float mmax[3];

		unsigned int min_el = 2;
		unsigned int max_el = 2;
		if (len1 >= len2 && len1 >= len3)
		{
			mmax[2] = len1;
			const float sqr_len = rsqrtf(len1);

			evecs[6] = vec_tmp_0.x * sqr_len;
			evecs[7] = vec_tmp_0.y * sqr_len;
			evecs[8] = vec_tmp_0.z * sqr_len;

			//evecs[2] = vec_tmp[0] * rsqrtf(len1);
		}
		else if (len2 >= len1 && len2 >= len3)
		{
			mmax[2] = len2;

			const float sqr_len = rsqrtf(len2);

			evecs[6] = vec_tmp_1.x * sqr_len;
			evecs[7] = vec_tmp_1.y * sqr_len;
			evecs[8] = vec_tmp_1.z * sqr_len;

			//evecs[2] = vec_tmp[1] * rsqrtf(len2);
		}
		else
		{
			mmax[2] = len3;

			const float sqr_len = rsqrtf(len3);

			evecs[6] = vec_tmp_2.x * sqr_len;
			evecs[7] = vec_tmp_2.y * sqr_len;
			evecs[8] = vec_tmp_2.z * sqr_len;

			//evecs[2] = vec_tmp[2] * rsqrtf(len3);
		}

		row_tmp[0] = make_float3(cov[0] - evals.y, cov[1], cov[2]);
		row_tmp[1] = make_float3(cov[1], cov[3] - evals.y, cov[4]);
		row_tmp[2] = make_float3(cov[2], cov[4], cov[5] - evals.y);

		vec_tmp_0 = f3_cross_product(row_tmp[0], row_tmp[1]);
		vec_tmp_1 = f3_cross_product(row_tmp[0], row_tmp[2]);
		vec_tmp_2 = f3_cross_product(row_tmp[1], row_tmp[2]);

		len1 = f3_inner_product(vec_tmp_0, vec_tmp_0);
		len2 = f3_inner_product(vec_tmp_1, vec_tmp_1);
		len3 = f3_inner_product(vec_tmp_2, vec_tmp_2);


		if (len1 >= len2 && len1 >= len3)
		{
			mmax[1] = len1;

			const float sqr_len = rsqrtf(len1);

			evecs[3] = vec_tmp_0.x * sqr_len;
			evecs[4] = vec_tmp_0.y * sqr_len;
			evecs[5] = vec_tmp_0.z * sqr_len;

			//evecs[1] = vec_tmp[0] * rsqrtf(len1);

			min_el = len1 <= mmax[min_el] ? 1 : min_el;
			max_el = len1  > mmax[max_el] ? 1 : max_el;
		}
		else if (len2 >= len1 && len2 >= len3)
		{
			mmax[1] = len2;

			const float sqr_len = rsqrtf(len2);

			evecs[3] = vec_tmp_1.x * sqr_len;
			evecs[4] = vec_tmp_1.y * sqr_len;
			evecs[5] = vec_tmp_1.z * sqr_len;

			//evecs[1] = vec_tmp[1] * rsqrtf(len2);
			min_el = len2 <= mmax[min_el] ? 1 : min_el;
			max_el = len2  > mmax[max_el] ? 1 : max_el;
		}
		else
		{
			mmax[1] = len3;

			const float sqr_len = rsqrtf(len3);

			evecs[3] = vec_tmp_2.x * sqr_len;
			evecs[4] = vec_tmp_2.y * sqr_len;
			evecs[5] = vec_tmp_2.z * sqr_len;

			//evecs[1] = vec_tmp[2] * rsqrtf(len3);
			min_el = len3 <= mmax[min_el] ? 1 : min_el;
			max_el = len3 >  mmax[max_el] ? 1 : max_el;
		}

		row_tmp[0] = make_float3(cov[0] - evals.x, cov[1], cov[2]);
		row_tmp[1] = make_float3(cov[1], cov[3] - evals.x, cov[4]);
		row_tmp[2] = make_float3(cov[2], cov[4], cov[5] - evals.x);

		vec_tmp_0 = f3_cross_product(row_tmp[0], row_tmp[1]);
		vec_tmp_1 = f3_cross_product(row_tmp[0], row_tmp[2]);
		vec_tmp_2 = f3_cross_product(row_tmp[1], row_tmp[2]);

		len1 = f3_inner_product(vec_tmp_0, vec_tmp_0);
		len2 = f3_inner_product(vec_tmp_1, vec_tmp_1);
		len3 = f3_inner_product(vec_tmp_2, vec_tmp_2);


		if (len1 >= len2 && len1 >= len3)
		{
			mmax[0] = len1;

			const float sqr_len = rsqrtf(len1);

			evecs[0] = vec_tmp_0.x * sqr_len;
			evecs[1] = vec_tmp_0.y * sqr_len;
			evecs[2] = vec_tmp_0.z * sqr_len;

			//evecs[0] = vec_tmp[0] * rsqrtf(len1);
			min_el = len3 <= mmax[min_el] ? 0 : min_el;
			max_el = len3  > mmax[max_el] ? 0 : max_el;
		}
		else if (len2 >= len1 && len2 >= len3)
		{
			mmax[0] = len2;

			const float sqr_len = rsqrtf(len2);

			evecs[0] = vec_tmp_1.x * sqr_len;
			evecs[1] = vec_tmp_1.y * sqr_len;
			evecs[2] = vec_tmp_1.z * sqr_len;

			//evecs[0] = vec_tmp[1] * rsqrtf(len2);
			min_el = len3 <= mmax[min_el] ? 0 : min_el;
			max_el = len3  > mmax[max_el] ? 0 : max_el;
		}
		else
		{
			mmax[0] = len3;

			const float sqr_len = rsqrtf(len3);

			evecs[0] = vec_tmp_2.x * sqr_len;
			evecs[1] = vec_tmp_2.y * sqr_len;
			evecs[2] = vec_tmp_2.z * sqr_len;

			//evecs[0] = vec_tmp[2] * rsqrtf(len3);
			min_el = len3 <= mmax[min_el] ? 0 : min_el;
			max_el = len3  > mmax[max_el] ? 0 : max_el;
		}

		unsigned mid_el = 3 - min_el - max_el;

		const int min_el_1 = ((min_el + 1) % 3) * 3;
		const int min_el_2 = ((min_el + 2) % 3) * 3;
		const int mid_el_1 = ((mid_el + 1) % 3) * 3;
		const int mid_el_2 = ((mid_el + 2) % 3) * 3;

		float3 evecs_min_el = f3_normalize(f3_cross_product(
			make_float3(evecs[min_el_1 + 0], evecs[min_el_1 + 1], evecs[min_el_1 + 2])
			, make_float3(evecs[min_el_2 + 0], evecs[min_el_2 + 1], evecs[min_el_2 + 2])));

		float3 evecs_mid_el = f3_normalize(f3_cross_product(
			make_float3(evecs[mid_el_1 + 0], evecs[mid_el_1 + 1], evecs[mid_el_1 + 2])
			, make_float3(evecs[mid_el_2 + 0], evecs[mid_el_2 + 1], evecs[mid_el_2 + 2])));

		evecs[min_el * 3 + 0] = evecs_min_el.x;
		evecs[min_el * 3 + 1] = evecs_min_el.y;
		evecs[min_el * 3 + 2] = evecs_min_el.z;

		evecs[mid_el * 3 + 0] = evecs_mid_el.x;
		evecs[mid_el * 3 + 1] = evecs_mid_el.y;
		evecs[mid_el * 3 + 2] = evecs_mid_el.z;

		//evecs[min_el] = normalized(cross(evecs[(min_el + 1) % 3], evecs[(min_el + 2) % 3]));
		//evecs[mid_el] = normalized(cross(evecs[(mid_el + 1) % 3], evecs[(mid_el + 2) % 3]));
	}
	// Rescale back to the original size.
	evals = make_float3(evals.x * scale, evals.y * scale, evals.z * scale);
	//evals *= scale;
}

__device__ __forceinline__
void parabola_fitting(float& min_value, float& min_cost, const float x1, const float x2, const float x3, const float y1, const float y2, const float y3)
{
	float denom = (x1 - x2) * (x1 - x3) * (x2 - x3);

	float A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom;
	float B = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom;
	float C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom;

	if (A == 0)
	{
		min_value = FLT_MAX;
		min_cost = FLT_MAX;
		return;
	}

	min_cost = C - B*B / (4 * A);
	min_value = -B / (2 * A);

	/*if (min_cost > min(y1, min(y2, y3)))
	{
	min_value = FLT_MAX;
	min_cost = FLT_MAX;
	}*/

	float min_range = min(x1, min(x2, x3));
	float max_range = max(x1, max(x2, x3));

	if (min_value < min_range || min_value > max_range)
	{
		min_value = FLT_MAX;
		min_cost = FLT_MAX;
	}
}

__device__ __forceinline__
float get_popcnt(const uint32_t* left, const uint32_t* right, const float target_x, const int refer_x, const int refer_y, const int width)
{
	const int x1 = target_x;
	const int x2 = min((int)(target_x + 1), width - 1);

	const float ratio = target_x - x1;

	const uint32_t refer_left = left[refer_y*width + refer_x];

	const float comp = __popc(refer_left ^ right[refer_y*width + x1])*(1.0f - ratio) + __popc(refer_left ^ right[refer_y*width + x2]) * ratio;

	return comp;
	//return comp *comp;
}

__global__
void disparity2tile_census(TILE* tile, const float* disparity, const uint32_t* left, const uint32_t* right, const float mask_component, const int width, const int height)
{
	const int tile_idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int tile_idy = blockIdx.y*blockDim.y + threadIdx.y;

	const int tile_cols = width / TILE_SIZE;
	const int tile_rows = height / TILE_SIZE;

	const int search_from = TILE_SIZE / 4;
	const int search_to = TILE_SIZE / 2 + TILE_SIZE / 4;

	if (tile_idx < tile_cols && tile_idy < tile_rows)
	{
		const int offset_idx = tile_idx*TILE_SIZE;
		const int offset_idy = tile_idy*TILE_SIZE;

		float min_disparity = 0;
		float min_cost = FLT_MAX;

		int min_range = INT_MAX;
		int max_range = 0;
		for (int my = search_from; my < search_to; ++my)
		{
			for (int mx = search_from; mx < search_to; ++mx)
			{
				int target_disparity = (int)(disparity[(offset_idy + my)*width + (offset_idx + mx)] + 0.5f);

				if (target_disparity > 0)
				{
					min_range = min(min_range, target_disparity);
					max_range = max(max_range, target_disparity);
				}
			}
		}

		min_range = max(min_range, MIN_DISPARITY);
		max_range = min(max_range, MAX_DISPARITY);

		//min_range = 0;
		//max_range = MAX_DISPARITY;

		for (int d = min_range; d <= max_range; ++d)
		{
			const int tile_offset_idx = offset_idx + TILE_SIZE / 2;
			const int tile_offset_idy = offset_idy + TILE_SIZE / 2;
			
			const int mask_half = max((MAX_MASK_SIZE_HALF * expf(-(d*d) / mask_component) + 0.5f), (float)MIN_MASK_SIZE_HALF);
			const int total_mask_size = (mask_half * 2 + 1) *(mask_half * 2 + 1);

			float cost = 0;
			for (int my = -mask_half; my <= mask_half; ++my)
			{
				const int mask_idy = middleOfThree(0, tile_offset_idy + my, height - 1);
				for (int mx = -mask_half; mx <= mask_half; ++mx)
				{
					const int mask_idx = middleOfThree(0, tile_offset_idx + mx, width - 1);
					const int disp_idx = middleOfThree(0, tile_offset_idx + mx - d, width - 1);

					cost += __popc(left[mask_idy*width + mask_idx] ^ right[mask_idy*width + disp_idx]);
				}
			}

			cost /= total_mask_size;	

			if (min_cost > cost)
			{
				min_cost = cost;
				min_disparity = d;
			}			
		}

		tile[tile_idy*tile_cols + tile_idx].d = min_disparity;
		tile[tile_idy*tile_cols + tile_idx].dx = 0;
		tile[tile_idy*tile_cols + tile_idx].dy = 0;
	}
}

__global__
void tile_refinement_census(TILE* tile, const uint32_t* left, const uint32_t* right, const float mask_component, const int width, const int height)
{
	const int tile_idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int tile_idy = blockIdx.y*blockDim.y + threadIdx.y;

	const int tile_cols = width / TILE_SIZE;
	const int tile_rows = height / TILE_SIZE;


	if (tile_idx < tile_cols && tile_idy < tile_rows)
	{
		const int offset_idx = tile_idx*TILE_SIZE;
		const int offset_idy = tile_idy*TILE_SIZE;

		TILE& target_tile = tile[tile_idy*tile_cols + tile_idx];

		const float disparity = target_tile.d;

		if (disparity < MIN_DISPARITY)
			return;

		float cost_left = 0;
		float cost_center = 0;
		float cost_right = 0;

		const int tile_offset_idx = offset_idx + TILE_SIZE / 2;
		const int tile_offset_idy = offset_idy + TILE_SIZE / 2;

		const int mask_half = max((MAX_MASK_SIZE_HALF * expf(-(disparity*disparity) / mask_component) + 0.5f), (float)MIN_MASK_SIZE_HALF);
		const int total_mask_size = (mask_half * 2 + 1) *(mask_half * 2 + 1);

		for (int my = -mask_half; my <= mask_half; ++my)
		{
			const int mask_idy = middleOfThree(0, tile_offset_idy + my, height - 1);
			for (int mx = -mask_half; mx <= mask_half; ++mx)
			{
				const float kx = mx - (TILE_SIZE / 2);
				const float ky = my - (TILE_SIZE / 2);

				const int mask_idx = middleOfThree(0, tile_offset_idx + mx, width - 1);

				cost_left += get_popcnt(left, right, middleOfThree(0, tile_offset_idx + mx + tile_disparity(disparity - 1, target_tile.dx, target_tile.dy, kx, ky), width - 1), mask_idx, mask_idy, width);
				cost_center += get_popcnt(left, right, middleOfThree(0, tile_offset_idx + mx + tile_disparity(disparity + 0, target_tile.dx, target_tile.dy, kx, ky), width - 1), mask_idx, mask_idy, width);
				cost_right += get_popcnt(left, right, middleOfThree(0, tile_offset_idx + mx + tile_disparity(disparity + 1, target_tile.dx, target_tile.dy, kx, ky), width - 1), mask_idx, mask_idy, width);
			}
		}

		float refine_cost, refine_disp;
		parabola_fitting(refine_disp, refine_cost, disparity - 1, disparity, disparity + 1, cost_left, cost_center, cost_right);
		if (refine_cost != FLT_MAX)
		{
			target_tile.d = refine_disp;
		}
		else
		{
			target_tile.d = 0;
		}
	}
}

__global__
void tile_slant_estimation_eigen(TILE* tile, const int width, const int height)
{
	const int tile_idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int tile_idy = blockIdx.y*blockDim.y + threadIdx.y;

	const int tile_cols = width / TILE_SIZE;
	const int tile_rows = height / TILE_SIZE;

	const int SAD = TILE_SIZE;
	const int tile_step = 3;
	const float threshold = 3.0f; // unit : disparity

	const bool limit_slant = true;
	const float maximum_slant = tanf(get_radian(MAXIMUM_SLANT_DEGREE));

	if (tile_idx < tile_cols && tile_idy < tile_rows)
	{
		const int offset_x = tile_idx*TILE_SIZE;
		const int offset_y = tile_idy*TILE_SIZE;

		TILE& target_tile = tile[tile_idy*tile_cols + tile_idx];

		const float target_disparity = target_tile.d;

		if (target_disparity < MIN_DISPARITY)
			return;

		float3 centroid = make_float3(0.f, 0.f, 0.f);
		float3 for_direction = make_float3(0.f, 0.f, 0.f);
		dim3 direction_cnt(0,0,0);

		int cnt = 0;
		for (int ny = -tile_step; ny <= tile_step; ++ny)
		{
			const int cy = middleOfThree(0, tile_idy + ny, tile_rows - 1);
			for (int nx = -tile_step; nx <= tile_step; ++nx)
			{
				const int cx = middleOfThree(0, tile_idx + nx, tile_cols - 1);
				const TILE t = tile[cy * tile_cols + cx];

				if (t.d < MIN_DISPARITY || abs(t.d - target_disparity) > threshold)
					continue;

				centroid.x += nx * TILE_SIZE;
				centroid.y += ny * TILE_SIZE;
				centroid.z += t.d;

				++cnt;

				if(nx != 0)
				{
					for_direction.x += (t.d - target_disparity) / (TILE_SIZE*nx);
					++direction_cnt.x;
				}
				if (ny != 0)
				{
					for_direction.y += (t.d - target_disparity) / (TILE_SIZE*ny);
					++direction_cnt.y;
				}

			}
		}

		if (cnt < 5)
		{
			return;
		}

		for_direction.x = (direction_cnt.x == 0) ? 0 : for_direction.x / direction_cnt.x;
		for_direction.y = (direction_cnt.y == 0) ? 0 : for_direction.y / direction_cnt.y;

		centroid = f3_div_elem(centroid, cnt);

		float cov[] = { 0, 0, 0, 0, 0, 0 };
		for (int ny = -tile_step; ny <= tile_step; ++ny)
		{
			const int cy = middleOfThree(0, tile_idy + ny, tile_rows - 1);
			for (int nx = -tile_step; nx <= tile_step; ++nx)
			{
				const int cx = middleOfThree(0, tile_idx + nx, tile_cols - 1);
				const TILE t = tile[cy * tile_cols + cx];

				if (t.d < MIN_DISPARITY || abs(t.d - target_disparity) > threshold)
					continue;

				float3 d = f3_sub(make_float3(nx * TILE_SIZE, ny * TILE_SIZE, t.d), centroid);

				cov[0] += d.x * d.x;               //cov (0, 0)
				cov[1] += d.x * d.y;               //cov (0, 1)
				cov[2] += d.x * d.z;               //cov (0, 2)
				cov[3] += d.y * d.y;               //cov (1, 1)
				cov[4] += d.y * d.z;               //cov (1, 2)
				cov[5] += d.z * d.z;               //cov (2, 2)
			}
		}

		__syncthreads();

		float evecs[9];
		float3 evals;
		solve_eigen_decomposition(cov, evecs, evals);
		float3 n = f3_normalize(make_float3(evecs[0], evecs[1], evecs[2]));

		float non_scaled_depth = 1.0f / target_disparity;
		float3 pt = f3_normalize(centroid);

		float inner = f3_inner_product(n, pt);

		// flip slant
		if(inner < 0)
		{
			evecs[0] *= -1;
			evecs[1] *= -1;
		}

		/*if(abs(evecs[0]) > maximum_slant)
			target_tile.dx = middleOfThree(-maximum_slant, evecs[0], maximum_slant);
		if (abs(evecs[1]) > maximum_slant)
			target_tile.dy = middleOfThree(-maximum_slant, evecs[1], maximum_slant);*/

		if (limit_slant)
		{
			target_tile.dx = middleOfThree(-maximum_slant, evecs[0], maximum_slant);
			target_tile.dy = middleOfThree(-maximum_slant, evecs[1], maximum_slant);
		}
		else
		{
			target_tile.dx = evecs[0];
			target_tile.dy = evecs[1];
		}
	}
}

__global__
void per_pixel_estimation_census(float* disparity, const TILE *tile, const uint32_t* left, const uint32_t* right, const float delta, const float mask_component, const int cols, const int rows)
{
	const int idx = (blockIdx.x*blockDim.x + threadIdx.x);
	const int idy = (blockIdx.y*blockDim.y + threadIdx.y);

	const int index = idy*cols + idx;

	const int tile_cols = cols / TILE_SIZE;
	const int tile_rows = rows / TILE_SIZE;

	const int tile_idx = idx / TILE_SIZE;
	const int tile_idy = idy / TILE_SIZE;

	const int direction_x = (idx%TILE_SIZE < TILE_SIZE / 2) ? -1 : 1;
	const int direction_y = (idy%TILE_SIZE < TILE_SIZE / 2) ? -1 : 1;
			

	if (idx < cols && idy < rows)
	{
		float min_cost = FLT_MAX;
		float min_disp = 0;

		const TILE current_tile = tile[tile_idy * tile_cols + tile_idx];
		for (int candi_y = 0; candi_y <= 1; ++candi_y)
		{
			const int candi_tile_idy = middleOfThree(0, tile_idy + candi_y*direction_y, tile_rows - 1);
			const int tile_offset_y = candi_tile_idy * TILE_SIZE + TILE_SIZE / 2;

			for (int candi_x = 0; candi_x <= 1; ++candi_x)
			{
				const int candi_tile_idx = middleOfThree(0, tile_idx + candi_x*direction_x, tile_cols - 1);
				const int tile_offset_x = candi_tile_idx * TILE_SIZE + TILE_SIZE / 2;

				const TILE target_tile = tile[candi_tile_idy * tile_cols + candi_tile_idx];

				if(target_tile.d < MIN_DISPARITY)
				{
					continue;
				}

				const int mask_size_half = max((MAX_MASK_SIZE_HALF * expf(-(target_tile.d*target_tile.d) / mask_component) + 0.5f), (float)MIN_MASK_SIZE_HALF);				
				const int total_mask_size = (mask_size_half * 2 + 1)*(mask_size_half * 2 + 1);

				float cost1 = 0.0f;
				float cost2 = 0.0f;
				float cost3 = 0.0f;

				for (int my = -mask_size_half; my <= mask_size_half; ++my)
				{
					const int mask_idy = middleOfThree(0, idy + my, rows - 1);
					const int ky = (idy + my) - tile_offset_y;

					for (int mx = -mask_size_half; mx <= mask_size_half; ++mx)
					{
						const int kx = (idx + mx) - tile_offset_x;
						const int mask_idx = middleOfThree(0, idx + mx, cols - 1);

						cost1 += get_popcnt(left, right, max(mask_idx + tile_disparity(target_tile.d - delta, target_tile.dx, target_tile.dy, kx, ky), 0.f), mask_idx, mask_idy, cols);
						cost2 += get_popcnt(left, right, max(mask_idx + tile_disparity(target_tile.d +     0, target_tile.dx, target_tile.dy, kx, ky), 0.f), mask_idx, mask_idy, cols);
						cost3 += get_popcnt(left, right, max(mask_idx + tile_disparity(target_tile.d + delta, target_tile.dx, target_tile.dy, kx, ky), 0.f), mask_idx, mask_idy, cols);

					}
				}

				cost1 /= total_mask_size;
				cost2 /= total_mask_size;
				cost3 /= total_mask_size;


				const float disp2 = tile_disparity(target_tile.d + 0, target_tile.dx, target_tile.dy, (idx - tile_offset_x), (idy - tile_offset_y));
				const float disp1 = tile_disparity(target_tile.d - delta, target_tile.dx, target_tile.dy, (idx - tile_offset_x), (idy - tile_offset_y));
				const float disp3 = tile_disparity(target_tile.d + delta, target_tile.dx, target_tile.dy, (idx - tile_offset_x), (idy - tile_offset_y));

				float target_cost, target_disp;
				parabola_fitting(target_disp, target_cost, disp1, disp2, disp3, cost1, cost2, cost3);

				if (target_cost != FLT_MAX)
				{	
					const float disp_distance = abs(target_disp - tile_disparity(target_tile.d));
					const float tile_distance = sqrtf((float)(idx - tile_offset_x)*(idx - tile_offset_x) + (float)(idy - tile_offset_y)*(idy - tile_offset_y));
					const float invalidation = tile_distance == 0 ? 0 : (disp_distance / tile_distance);

					if (invalidation < tanf(get_radian(INVALIDATION_SLANT_DEGREE)))
					{
						if (min_cost > target_cost)
						{
							min_cost = target_cost;
							min_disp = target_disp;
						}
					}
				}
			}
		}



		if (min_cost != FLT_MAX)
		{
			min_disp *= -1;
			if (min_disp > MIN_DISPARITY && min_disp < MAX_DISPARITY)
			{
				disparity[index] = min_disp;
			}
			else
			{
				disparity[index] = 0;
			}
		}
		else
		{
			disparity[index] = 0;
		}

	}

}

__global__
void invalidation(float* disparity, const float* init_disparity, const int cols, const int rows)
{
	const int idx = (blockIdx.x*blockDim.x + threadIdx.x);
	const int idy = (blockIdx.y*blockDim.y + threadIdx.y);

	const int index = idy*cols + idx;

	if (idx < cols && idy < rows)
	{
		const float disp = disparity[index];
		const float init_disp = init_disparity[index];

		if (init_disp == 0 || abs(disp - init_disp) > 3.0f)
		{
			disparity[index] = 0;
		}
	}
}

__global__
void tile2disparity(const TILE *tile, float* disparity, const int cols, const int rows)
{
	const int tile_idx = (blockIdx.x*blockDim.x + threadIdx.x);
	const int tile_idy = (blockIdx.y*blockDim.y + threadIdx.y);

	const int tile_cols = cols / TILE_SIZE;
	const int tile_rows = rows / TILE_SIZE;

	const int tile_index = tile_idy*tile_cols + tile_idx;

	const int idx = tile_idx * TILE_SIZE;
	const int idy = tile_idy * TILE_SIZE;


	if (tile_idx < tile_cols && tile_idy < tile_rows)
	{
		const TILE target_tile = tile[tile_index];

		for (int my = 0; my < TILE_SIZE; my++)
		{
			for (int mx = 0; mx < TILE_SIZE; mx++)
			{
				const float kx = mx - (TILE_SIZE / 2);
				const float ky = my - (TILE_SIZE / 2);

				disparity[(idy + my) * cols + (idx + mx)] = -tile_disparity(target_tile.d, target_tile.dx, target_tile.dy, kx, ky);
			}
		}
	}
}

__global__
void tile_slant_visualizer(const TILE *tile, uchar* vis_x, uchar* vis_y, const int cols, const int rows)
{
	const int tile_idx = (blockIdx.x*blockDim.x + threadIdx.x);
	const int tile_idy = (blockIdx.y*blockDim.y + threadIdx.y);

	const int tile_cols = cols / TILE_SIZE;
	const int tile_rows = rows / TILE_SIZE;

	const int tile_index = tile_idy*tile_cols + tile_idx;

	const int idx = tile_idx * TILE_SIZE;
	const int idy = tile_idy * TILE_SIZE;

	const float maximum = tanf(get_radian(30));

	if (tile_idx < tile_cols && tile_idy < tile_rows)
	{
		const TILE target_tile = tile[tile_index];

		for (int my = 0; my < TILE_SIZE; my++)
		{
			for (int mx = 0; mx < TILE_SIZE; mx++)
			{
				const float kx = mx - (TILE_SIZE / 2);
				const float ky = my - (TILE_SIZE / 2);

				//vis_x[(idy + my) * cols + (idx + mx)] = ((target_tile.dx + maximum) / (2 * maximum)) * 255;
				//vis_y[(idy + my) * cols + (idx + mx)] = ((target_tile.dy + maximum) / (2 * maximum)) * 255;
				vis_x[(idy + my) * cols + (idx + mx)] = (abs(target_tile.dx) / (maximum)) * 255;
				vis_y[(idy + my) * cols + (idx + mx)] = (abs(target_tile.dy) / (maximum)) * 255;
			}
		}
	}
}


void Stereo_module::create(const int cols, const int rows)
{
	this->stereo_cols = cols;
	this->stereo_rows = rows;

	const int img_size = stereo_cols*stereo_rows;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&this->left_image, sizeof(unsigned char)*img_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&this->right_image, sizeof(unsigned char)*img_size));

	CUDA_CHECK_RETURN(cudaMalloc((void **)&this->left_census, sizeof(uint32_t)*img_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&this->right_census, sizeof(uint32_t)*img_size));

	CUDA_CHECK_RETURN(cudaMalloc((void **)&this->tile, sizeof(TILE)*(this->stereo_cols / TILE_SIZE)*(this->stereo_rows / TILE_SIZE)));

	CUDA_CHECK_RETURN(cudaMalloc((void **)&this->init_disparity, sizeof(float)*img_size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&this->final_disparity, sizeof(float)*img_size));
}

void Stereo_module::process(const unsigned char *left, const unsigned char *right, const float* disp, float* refine)
{
	const int img_size = stereo_cols*stereo_rows;

	CUDA_CHECK_RETURN(cudaMemcpy(this->left_image, left, sizeof(unsigned char)*img_size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(this->right_image, right, sizeof(unsigned char)*img_size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(this->init_disparity, disp, sizeof(float)*img_size, cudaMemcpyHostToDevice));

	const float mask_sigma = -(float)(STANDARD_DISPARITY*STANDARD_DISPARITY) / log1p((float)STANDARD_MASK_SIZE / (float)MAX_MASK_SIZE_HALF - 1.f);
	
#if USE_CHECK_TIME
	cudaEvent_t start, stop;
	float gpu_time = 0.0f;
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));
	cudaEventRecord(start, 0);
#endif

	const int width_per_block = BLOCK_SIZE - CENSUS_WIDTH + 1;
	const int height_per_block = LINES_PER_BLOCK;
	const dim3 census_grid(
		(this->stereo_cols + width_per_block - 1) / width_per_block,
		(this->stereo_rows + height_per_block - 1) / height_per_block);
	const dim3 thread_block(BLOCK_SIZE);	
	
	dim3 tile_grid = dim3(divUp(stereo_cols / TILE_SIZE, thread_block.x), divUp(stereo_rows / TILE_SIZE, thread_block.y));
	dim3 pixel_grid = dim3(divUp(stereo_cols, thread_block.x), divUp(stereo_rows, thread_block.y));
				
	census_transform_kernel << <census_grid, thread_block >> >(this->left_census, this->left_image, stereo_cols, stereo_rows);
	census_transform_kernel << <census_grid, thread_block >> >(this->right_census, this->right_image, stereo_cols, stereo_rows);
	
	disparity2tile_census << <tile_grid, thread_block >> > (this->tile, this->init_disparity, this->left_census, this->right_census, mask_sigma, this->stereo_cols, this->stereo_rows);
	tile_refinement_census << <tile_grid, thread_block >> > (this->tile, this->left_census, this->right_census, mask_sigma, this->stereo_cols, this->stereo_rows);
	tile_slant_estimation_eigen << <tile_grid, thread_block >> > (this->tile, this->stereo_cols, this->stereo_rows);
	per_pixel_estimation_census << <pixel_grid, thread_block >> > (this->final_disparity, this->tile, this->left_census, this->right_census, 1.0f, mask_sigma, this->stereo_cols, this->stereo_rows);

	invalidation << <pixel_grid, thread_block >> >(this->final_disparity, this->init_disparity, this->stereo_cols, this->stereo_rows);
		
	CUDA_CHECK_RETURN(cudaMemcpy(refine, this->final_disparity, sizeof(float)*img_size, cudaMemcpyDeviceToHost));

#if USE_CHECK_TIME
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	CUDA_CHECK_RETURN(cudaEventElapsedTime(&gpu_time, start, stop));
	printf("Time spent: %.5f\n", gpu_time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	std::cout << "error : " << cudaGetErrorString(cudaGetLastError()) << std::endl;
#endif	
}

void Stereo_module::slant_visualization()
{
	uchar*  vis_x;
	uchar*	vis_y;

	cudaMalloc((void**)&vis_x, sizeof(uchar)*this->stereo_cols * this->stereo_rows);
	cudaMalloc((void**)&vis_y, sizeof(uchar)*this->stereo_cols * this->stereo_rows);
	
	const dim3 thread_block(BLOCK_SIZE);
	dim3 tile_grid = dim3(divUp(stereo_cols / TILE_SIZE, thread_block.x), divUp(stereo_rows / TILE_SIZE, thread_block.y));

	tile_slant_visualizer << <tile_grid, thread_block >> > (this->tile, vis_x, vis_y, this->stereo_cols, this->stereo_rows);
	
	cv::Mat visual_x = cv::Mat(cv::Size(this->stereo_cols, this->stereo_rows), CV_8UC1);
	cv::Mat visual_y = cv::Mat(cv::Size(this->stereo_cols, this->stereo_rows), CV_8UC1);
	
	CUDA_CHECK_RETURN(cudaMemcpy(visual_x.data, vis_x, sizeof(uchar) * this->stereo_cols * this->stereo_rows, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(visual_y.data, vis_y, sizeof(uchar) * this->stereo_cols * this->stereo_rows, cudaMemcpyDeviceToHost));
	
	cv::imshow("visual_x", visual_x);
	cv::imshow("visual_y", visual_y);
	
	cudaFree(vis_x);
	cudaFree(vis_y);
}


void Stereo_module::release()
{

	CUDA_CHECK_RETURN(cudaFree(this->left_image));
	CUDA_CHECK_RETURN(cudaFree(this->right_image));

	CUDA_CHECK_RETURN(cudaFree(this->tile));

	CUDA_CHECK_RETURN(cudaFree(this->init_disparity));
	CUDA_CHECK_RETURN(cudaFree(this->final_disparity));


}
}