// ppm.h
// Minimal PPM (P6) reader/writer for 8-bit RGB images (maxval = 255).
// Designed for CUDA coursework: contiguous RGBRGB... byte layout.
//
// Supports:
//   - P6 binary PPM only
//   - Comments in header (# ...)
//   - Arbitrary whitespace between header tokens
//
// Does NOT support:
//   - P3 (ASCII) PPM
//   - maxval != 255
//   - grayscale PGM/PBM variants
//
// Build: compile and link ppm.cpp with your project.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ppm {

struct Image {
  int width  = 0;
  int height = 0;
  std::vector<std::uint8_t> data; // size = 3 * width * height

  std::size_t bytes() const { return data.size(); }
  bool empty() const { return width <= 0 || height <= 0 || data.empty(); }
};

// Throws std::runtime_error on parse/IO errors.
Image read(const std::string& filename);

// Throws std::runtime_error on IO/validation errors.
void write(const std::string& filename, const Image& img);

// Convenience: allocate an empty image with the correct buffer size.
Image make(int width, int height);

// Convenience: pointer access for CUDA copies, etc.
inline std::uint8_t*       pixels(Image& img)       { return img.data.data(); }
inline const std::uint8_t* pixels(const Image& img) { return img.data.data(); }

} // namespace ppm

