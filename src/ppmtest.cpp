// main.cpp
//
// PPM I/O smoke test.
// Usage:
//   ./ppm_test input.ppm output.ppm
//
// Behavior:
//   - Reads a binary PPM (P6)
//   - Prints basic image info
//   - Writes the image back out unchanged
//
// If output.ppm differs from input.ppm (byte-for-byte, ignoring header
// whitespace), the PPM I/O code is likely incorrect.

#include "ppm.h"

#include <cstdlib>
#include <exception>
#include <iostream>

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input.ppm> <output.ppm>\n";
    return EXIT_FAILURE;
  }

  const std::string input_file  = argv[1];
  const std::string output_file = argv[2];

  try {
    ppm::Image img = ppm::read(input_file);

    std::cout << "Loaded image:\n";
    std::cout << "  Width:  " << img.width << "\n";
    std::cout << "  Height: " << img.height << "\n";
    std::cout << "  Bytes:  " << img.bytes() << "\n";

    if (img.bytes() != static_cast<std::size_t>(3) * img.width * img.height) {
      std::cerr << "ERROR: unexpected buffer size\n";
      return EXIT_FAILURE;
    }

    ppm::write(output_file, img);

    std::cout << "Wrote image to '" << output_file << "'\n";
    std::cout << "PPM smoke test PASSED\n";
  }
  catch (const std::exception& e) {
    std::cerr << "PPM smoke test FAILED\n";
    std::cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}