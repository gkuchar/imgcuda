// ppm.cpp
#include "ppm.h"

#include <cctype>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ppm {

// --- helpers ---------------------------------------------------------------

static void throw_io_error(const std::string& filename, const std::string& what) {
  std::ostringstream oss;
  oss << "PPM IO error (" << filename << "): " << what;
  throw std::runtime_error(oss.str());
}

// Read the next non-comment token from the stream.
// Tokens are separated by whitespace. Lines beginning with '#' are comments
// (after optional leading whitespace) and are skipped.
static std::string next_token(std::istream& in) {
  while (true) {
    // Skip whitespace
    int c = in.peek();
    while (c != EOF && std::isspace(static_cast<unsigned char>(c))) {
      in.get();
      c = in.peek();
    }
    if (c == EOF) return {};

    // If we see a comment marker, skip the rest of that line.
    if (c == '#') {
      std::string dummy;
      std::getline(in, dummy);
      continue;
    }

    // Read a token
    std::string tok;
    while (true) {
      c = in.peek();
      if (c == EOF) break;
      if (std::isspace(static_cast<unsigned char>(c)) || c == '#') break;
      tok.push_back(static_cast<char>(in.get()));
    }
    if (!tok.empty()) return tok;

    // If token empty due to oddities, continue.
  }
}

static int parse_int_token(const std::string& tok, const char* field_name) {
  if (tok.empty()) {
    std::ostringstream oss;
    oss << "Missing token for " << field_name;
    throw std::runtime_error(oss.str());
  }
  // Use stoll with range checks
  std::size_t idx = 0;
  long long v = 0;
  try {
    v = std::stoll(tok, &idx, 10);
  } catch (...) {
    std::ostringstream oss;
    oss << "Invalid integer for " << field_name << ": '" << tok << "'";
    throw std::runtime_error(oss.str());
  }
  if (idx != tok.size()) {
    std::ostringstream oss;
    oss << "Invalid integer for " << field_name << ": '" << tok << "'";
    throw std::runtime_error(oss.str());
  }
  if (v < 0 || v > std::numeric_limits<int>::max()) {
    std::ostringstream oss;
    oss << "Out-of-range integer for " << field_name << ": '" << tok << "'";
    throw std::runtime_error(oss.str());
  }
  return static_cast<int>(v);
}

Image make(int width, int height) {
  if (width <= 0 || height <= 0) {
    throw std::runtime_error("ppm::make: width and height must be positive");
  }
  Image img;
  img.width = width;
  img.height = height;
  img.data.resize(static_cast<std::size_t>(3) * static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
  return img;
}

// --- public API ------------------------------------------------------------

Image read(const std::string& filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) {
    throw_io_error(filename, std::string("cannot open for reading: ") + std::strerror(errno));
  }

  const std::string magic = next_token(in);
  if (magic != "P6") {
    std::ostringstream oss;
    oss << "Unsupported format: expected 'P6', got '" << magic << "'";
    throw_io_error(filename, oss.str());
  }

  const int width  = parse_int_token(next_token(in), "width");
  const int height = parse_int_token(next_token(in), "height");
  const int maxval = parse_int_token(next_token(in), "maxval");

  if (width <= 0 || height <= 0) {
    throw_io_error(filename, "Invalid dimensions (width/height must be positive)");
  }
  if (maxval != 255) {
    std::ostringstream oss;
    oss << "Unsupported maxval: expected 255, got " << maxval;
    throw_io_error(filename, oss.str());
  }

  // Per PPM spec, after maxval there is a single whitespace (often '\n'),
  // then binary pixel data. Our token reader stops at whitespace but does not
  // consume the delimiter. Consume exactly one whitespace char if present.
  int c = in.peek();
  if (c != EOF && std::isspace(static_cast<unsigned char>(c))) {
    in.get();
  }

  const std::size_t nbytes =
      static_cast<std::size_t>(3) * static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

  Image img = make(width, height);

  in.read(reinterpret_cast<char*>(img.data.data()), static_cast<std::streamsize>(nbytes));
  if (!in) {
    std::ostringstream oss;
    oss << "Unexpected EOF while reading pixel data (expected " << nbytes << " bytes)";
    throw_io_error(filename, oss.str());
  }

  // Optional: ensure there isn't missing data; we don't require EOF afterward.
  return img;
}

void write(const std::string& filename, const Image& img) {
  if (img.width <= 0 || img.height <= 0) {
    throw_io_error(filename, "Invalid image dimensions (width/height must be positive)");
  }
  const std::size_t expected =
      static_cast<std::size_t>(3) * static_cast<std::size_t>(img.width) * static_cast<std::size_t>(img.height);
  if (img.data.size() != expected) {
    std::ostringstream oss;
    oss << "Image buffer has wrong size: expected " << expected << " bytes, got " << img.data.size();
    throw_io_error(filename, oss.str());
  }

  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    throw_io_error(filename, std::string("cannot open for writing: ") + std::strerror(errno));
  }

  // Header
  out << "P6\n" << img.width << " " << img.height << "\n255\n";
  if (!out) {
    throw_io_error(filename, "Failed while writing header");
  }

  // Binary pixel data
  out.write(reinterpret_cast<const char*>(img.data.data()), static_cast<std::streamsize>(img.data.size()));
  if (!out) {
    throw_io_error(filename, "Failed while writing pixel data");
  }
}

} // namespace ppm
