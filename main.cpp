#include "stereo_module.cuh"
#include "display.h"

#include <librealsense2/rs.hpp>     // Include RealSense Cross Platform API
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>

cv::Point3f deprojection(const cv::Point2f pixel, float depth, float fx, float fy, float ppx, float ppy)
{
	cv::Point3f point_3d = cv::Point3f();
	float x = (pixel.x - ppx) / fx;
	float y = (pixel.y - ppy) / fy;

	point_3d.x = depth * x;
	point_3d.y = depth * y;
	point_3d.z = depth;
	return point_3d;
}

int main(int argc, char * argv[])
{	
	const int window_width = 1280;
	const int window_height = 720;

	const int depth_width = 848;
	const int depth_height = 480;
	const int fps = 30;

	const float depth_scale = 0.001f;

	Display window(window_width, window_height, "visualization");
	sdl_event_state handler(window_width, window_height);	

	rs2::pipeline p;
	rs2::config c;

	c.enable_stream(RS2_STREAM_DEPTH	  , depth_width, depth_height, RS2_FORMAT_Z16, fps);
	c.enable_stream(RS2_STREAM_INFRARED, 1, depth_width, depth_height, RS2_FORMAT_Y8, fps);
	c.enable_stream(RS2_STREAM_INFRARED, 2, depth_width, depth_height, RS2_FORMAT_Y8, fps);

	rs2::pipeline_profile profile = p.start(c);
	
	auto ir_l_stream = profile.get_stream(RS2_STREAM_INFRARED, 1);
	auto ir_r_stream = profile.get_stream(RS2_STREAM_INFRARED, 2);

	rs2_intrinsics ir_intrinsic;
	rs2_get_video_stream_intrinsics(ir_l_stream, &ir_intrinsic, nullptr);

	rs2_extrinsics ir_extrinsic = ir_r_stream.get_extrinsics_to(ir_l_stream);
	
	const float stereo_fx = ir_intrinsic.fx; // meter	
	const float baseline = ir_extrinsic.translation[0]; // meter
	
	disparity_refinement::Stereo_module stereo;
	stereo.create(depth_width, depth_height);

	std::vector<cv::Point3f> vis_vertex;
	std::vector<cv::Vec3f> vis_color;

	while (!handler.quit) // Application still alive?
	{		
		rs2::frameset frameset = p.wait_for_frames();

		handler.event_handler();
		window.Clear(0.0f, 0.0f, 0.0f, 1.0f);

		if (frameset)
		{
			vis_color.clear();
			vis_vertex.clear();

			auto depth_frame = frameset.get_depth_frame();
			auto ir_left_frame = frameset.get_infrared_frame(1);
			auto ir_right_frame = frameset.get_infrared_frame(2);

			cv::Mat dMat_depth_16U = cv::Mat(cv::Size(depth_width, depth_height), CV_16UC1, (void*)depth_frame.get_data());
			cv::Mat dMat_left = cv::Mat(cv::Size(depth_width, depth_height), CV_8UC1, (void*)ir_left_frame.get_data());
			cv::Mat dMat_right = cv::Mat(cv::Size(depth_width, depth_height), CV_8UC1, (void*)ir_right_frame.get_data());
									
							
			if (dMat_left.empty() || dMat_right.empty())
				continue;

			cv::Mat disp = cv::Mat::zeros(cv::Size(depth_width, depth_height), CV_32FC1);

			for (int y = 0; y < depth_height; ++y)
			{
				for (int x = 0; x < depth_width; ++x)
				{
					const cv::Point2i pixel = cv::Point2i(x, y);
					const float depth = dMat_depth_16U.at<ushort>(pixel) * depth_scale;
					if (depth > 0)
					{
						disp.at<float>(pixel) = (stereo_fx * baseline) / depth;
					}
				}
			}

			bool refine = handler.get_r_key();

			if (refine) 
			{
				stereo.process(dMat_left.data, dMat_right.data, (float*)disp.data, (float*)disp.data);
				stereo.slant_visualization();
				printf("refinement!\n");
			}
			else
			{
				printf("original!\n");
			}
			

			cv::Mat ir_mask_left = cv::Mat::zeros(cv::Size(depth_width, depth_height), CV_8UC1);

			for (int y = 0; y < depth_height; ++y)
			{
				for (int x = 0; x < depth_width; ++x)
				{
					const cv::Point2i pixel = cv::Point2i(x, y);
					const float depth = dMat_depth_16U.at<ushort>(pixel) * depth_scale;
										
					if (disp.at<float>(pixel) > 0)
					{
						disp.at<float>(pixel) = (stereo_fx * baseline) / (disp.at<float>(pixel));

						ir_mask_left.at<uchar>(pixel) = dMat_left.at<uchar>(pixel);				
						
						//abs_map.at<uchar>(pixel) = abs((int)dMat_left.at<uchar>(pixel) - (int)dMat_right.at<uchar>(cv::Point2i(x-(int)((stereo_fx * baseline) / (disp.at<float>(pixel)) + 0.5f),y)));


						//const float color = dMat_depth_visualizer.at<float>(pixel);
						const uchar color = dMat_left.at<uchar>(pixel);
						vis_color.push_back(cv::Vec3f(color / 255.f, color / 255.f, color / 255.f));
						auto pt = deprojection(pixel, disp.at<float>(pixel), ir_intrinsic.fx, ir_intrinsic.fy, ir_intrinsic.ppx, ir_intrinsic.ppy);
						vis_vertex.push_back(pt);
					}
					else
					{
						
						disp.at<float>(pixel) = 0;
					}					
				}
			}		
			cv::imshow("ir mask - left", ir_mask_left);
			cv::imshow("dMat_left", dMat_left);
			//cv::imshow("dMat_left", dMat_right);
			cv::imshow("disp", disp);
		}
		
		const int vis_size = vis_vertex.size();

		glPopMatrix();
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glEnable(GL_DEPTH_TEST);

		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		gluPerspective(60, window_width / (float)window_height, 0.1f, 10.0f);

		cv::Point3f center = cv::Point3f(0, 0, 0.5f);

		glTranslatef(handler.x_offset - center.x, -(handler.y_offset - center.y), -(handler.z_offset - center.z));
		glTranslatef(center.x, -center.y, -center.z);
		glRotated(handler.get_psi(), 1, 0, 0);
		glRotated(-handler.get_theta(), 0, 1, 0);
		glTranslatef(-center.x, +center.y, +center.z);

		glPointSize(2.0f);
		glBegin(GL_POINTS);
		for (int idx = 0; idx < vis_size; ++idx)
		{
			glColor3f(vis_color[idx].val[0], vis_color[idx].val[1], vis_color[idx].val[2]);
			glVertex3f(vis_vertex[idx].x, -vis_vertex[idx].y, -vis_vertex[idx].z);
		}
		glEnd();

		glPopMatrix();
		glPopMatrix();
		glPopAttrib();
		glPushMatrix();

		window.SwapBuffers();
		SDL_Delay(1);
		cv::waitKey(1);

	}


	cv::destroyAllWindows();

	return 0;
}