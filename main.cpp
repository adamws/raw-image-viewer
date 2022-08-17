#include <SDL2/SDL.h>
#include <SDL2/SDL_pixels.h>
#include <SDL2/SDL_surface.h>

#ifdef __EMSCRIPTEN__
  #include <emscripten.h>
#else
  #include <fstream>
  #include <iostream>
  #include <string>
#endif

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"

#define WINDOW_WIDTH    640
#define WINDOW_HEIGHT   480

SDL_Window *window = NULL;
SDL_Renderer *renderer = NULL;
SDL_Texture *texture = NULL;

cv::Mat input;
std::vector<uchar> pngBuffer;

// forward declarations of exported functions
extern "C" {
  uint8_t* create_buffer(int width, int height);
  void load_textures();
  uint8_t* get_png_data();
  size_t get_png_size();
}

bool init() {
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    return false;
  }

  SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);
  if (window == NULL | renderer == NULL) {
    return false;
  }

  SDL_EventState(SDL_TEXTINPUT, SDL_DISABLE);
  SDL_EventState(SDL_KEYDOWN, SDL_DISABLE);
  SDL_EventState(SDL_KEYUP, SDL_DISABLE);

  return true;
}

void render() {
  SDL_RenderClear(renderer);
  SDL_RenderCopy(renderer, texture, NULL, NULL);
  SDL_RenderPresent(renderer);
}

void destroy() {
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
}

uint8_t* create_buffer(int width, int height) {
  input = cv::Mat(height, width, CV_8UC1);
  return input.data;
}

void load_textures() {
  // if there is an existing texture we need to free it's memory
  if (texture != NULL) {
    SDL_DestroyTexture(texture);
  }

  cv::imencode(".png", input, pngBuffer);

  cv::Mat dst;
  cv::cvtColor(input, dst, CV_GRAY2RGB);

  int width = dst.cols;
  int height = dst.rows;
  SDL_Surface* surface = SDL_CreateRGBSurfaceFrom(
      dst.data, width, height, 24, width * 3, 0xFF0000, 0x00FF00, 0x0000FF, 0);
  if (!surface) {
      exit(EXIT_FAILURE);
  }
  texture = SDL_CreateTextureFromSurface(renderer, surface);
  SDL_FreeSurface(surface);

  input.release();
}

uint8_t* get_png_data() {
  return pngBuffer.data();
}

size_t get_png_size() {
  return pngBuffer.size();
}

int main(int argc, char** argv) {
#ifdef __EMSCRIPTEN__
  init();
  emscripten_set_main_loop(render, 0, 1);
#else
  if (argc != 4) {
    std::cout << "Usage:" << std::endl;
    std::cout << argv[0] << " FILENAME WIDTH HEIGHT" << std::endl;
    return EXIT_FAILURE;
  } else {
    auto filename = argv[1];
    auto width = std::stoi(argv[2]);
    auto height = std::stoi(argv[3]);
    auto data = create_buffer(width, height);
    std::ifstream fin(filename, std::ios::in | std::ios::binary);
    fin.read(reinterpret_cast<char *>(data), width * height);

    init();
    load_textures();
  }

  bool running = true;
  while(running) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      switch (event.type) {
        case SDL_QUIT:
          running = false;
          break;
        default:
          break;
      }
    }
    render();
  }
#endif
  destroy();
  return EXIT_SUCCESS;
}
