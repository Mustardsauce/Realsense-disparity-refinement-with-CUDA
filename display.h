#ifndef DISPLAY_H
#define DISPLAY_H

#include "SDL2\include\SDL.h"
#include "glew\include\GL\glew.h"

#include <algorithm>
#include <string>

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glu32.lib")
#pragma comment (lib, "./glew/lib/Release/x64/glew32.lib")
#pragma comment(lib, "./SDL2/lib/x64/SDL2.lib")
#pragma comment(lib, "./SDL2/lib/x64/SDL2main.lib")

class Display
{
public:
	Display(int width, int height, const std::string& title);

	void Clear(float r, float g, float b, float a);
	void SwapBuffers();

	int get_width() {
		return display_width;
	}
	int get_height() {
		return display_height;
	}

	virtual ~Display();
	SDL_Window* m_window;
protected:
private:
	void operator=(const Display& display) {}
	Display(const Display& display) {}

	int display_width;
	int display_height;

	//SDL_Window* m_window;
	SDL_GLContext m_glContext;
};

class sdl_event_state
{
public:

	sdl_event_state(int w, int h)
	: display_w(w), display_h(h)
		, x_offset(0.0f), y_offset(0.0f), z_offset(2.0f)
	 , point_size(2.0f)
	{
		previous_x_offset = x_offset;
		previous_y_offset = y_offset;
		previous_z_offset = z_offset;

		is_dragging = false;
		quit = false;
		theta = 0.0f;
		psi = 0.0f;


		r_key_toggle = true;
	}

	float x_offset;
	float y_offset;
	float z_offset;

	int display_w;
	int display_h;
	
	float point_size;

	bool quit;

	void event_handler()
	{
		SDL_Event e;
				
		bool window_focus = false;

		while (SDL_PollEvent(&e)) 
		{
			switch (e.type)
			{
				case SDL_QUIT:
					this->quit = true;
					break;

				case SDL_MOUSEWHEEL:
				{
					if (e.wheel.y && !window_focus)
					{
						float ratio = 0.2f;
						this->z_offset = std::max(this->z_offset - e.wheel.y*ratio, 0.2f);

						previous_x_offset = this->x_offset;
						previous_y_offset = this->y_offset;
						previous_z_offset = this->z_offset;
					}

					break;
				}					
					
				case SDL_MOUSEBUTTONDOWN:
					if (e.button.button == SDL_BUTTON_LEFT) {
						if (!window_focus)
						{
							is_dragging = true;
							drag_x_offset = e.button.x;
							drag_y_offset = e.button.y;
						}
					}
					break;
				case SDL_MOUSEBUTTONUP:
					if (e.button.button == SDL_BUTTON_LEFT) {
						is_dragging = false;

					}
					break;
				case SDL_MOUSEMOTION:
				{
					if (is_dragging && !window_focus)
					{
						theta += ((-e.button.x + drag_x_offset) / (display_w*0.5f))*60.0f;
						psi -= ((-e.button.y + drag_y_offset) / (display_h*0.5f))*60.0f;

						drag_x_offset = e.button.x;
						drag_y_offset = e.button.y;
					}
				}
				//case SDL_KEYDOWN:
				{
				case SDL_KEYUP:
				{
					int key = e.key.keysym.scancode;
					if (key == 21)	// r key
						r_key_toggle = !r_key_toggle;
				}
				}
			}
		}

	}


	float get_theta() const { return this->theta; }
	float get_psi() const { return this->psi; }

	bool get_r_key() const{ return this->r_key_toggle; }

private:
	bool is_dragging;
	int drag_x_offset;
	int drag_y_offset;

	float previous_x_offset;
	float previous_y_offset;
	float previous_z_offset;

	bool r_key_toggle;

	float theta;
	float psi;
	

};

#endif
