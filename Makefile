# Makefile — CUDA PPM Image Processing Library
# Location: project root
#
# Binaries:
#   bin/imgtool   - CUDA image processing CLI
#   bin/ppmtest   - PPM I/O test
#
# Directory layout:
#   src/
#     main.cpp        (CLI driver)
#     ppmtest.cpp     (PPM test)
#     ppm.h ppm.cpp
#     imgcuda.h imgcuda.cu
#     imgreference.h imgreference.cpp
#	tests/
#	  IMG_1528.jpeg IMG_1528.txt
#     griffin.ppm griffin.txt
#     lenna.ppm lenna.txt
#
# Build:
#   make
#   make ppmtest
#   make clean

CXX  := g++
NVCC := nvcc

# ---------------- configuration ----------------
CXXSTD := -std=c++17
OPT    := -O2
WARN   := -Wall -Wextra -Wpedantic -Werror

# Set CUDA architecture (override at command line if needed)
# Examples:
#   A100: 80   A40: 86   RTX 4090: 86
CUDA_ARCH ?= 80

CXXFLAGS  := $(OPT) $(CXXSTD) $(WARN)
NVCCFLAGS := $(OPT) --std=c++17 -arch=sm_$(CUDA_ARCH)

# ---------------- directories ------------------
SRC_DIR := src
OBJ_DIR := build
BIN_DIR := bin

# ---------------- targets ----------------------
IMGTOOL := $(BIN_DIR)/imgtool
PPMTEST := $(BIN_DIR)/ppmtest

# ---------------- source files -----------------
IMGTOOL_CPP := \
  $(SRC_DIR)/main.cpp \
  $(SRC_DIR)/ppm.cpp \
  $(SRC_DIR)/imgreference.cpp

IMGTOOL_CU := \
  $(SRC_DIR)/imgcuda.cu

PPMTEST_CPP := \
  $(SRC_DIR)/ppmtest.cpp \
  $(SRC_DIR)/ppm.cpp

# ---------------- object files -----------------
IMGTOOL_CPP_OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(IMGTOOL_CPP))
IMGTOOL_CU_OBJ  := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(IMGTOOL_CU))

PPMTEST_CPP_OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(PPMTEST_CPP))

# ---------------- default rule -----------------
.PHONY: all
all: $(IMGTOOL)

# ---------------- directory rules --------------
$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@

# ---------------- compile rules ----------------
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ---------------- link rules -------------------
# Link imgtool with nvcc to pull in CUDA runtime
$(IMGTOOL): $(IMGTOOL_CPP_OBJ) $(IMGTOOL_CU_OBJ) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# PPM test is CPU-only
$(PPMTEST): $(PPMTEST_CPP_OBJ) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# ---------------- convenience targets ----------
.PHONY: ppmtest
ppmtest: $(PPMTEST)

.PHONY: test
test: $(IMGTOOL)
	@echo "Example:"
	@echo "  ./bin/imgtool tests/in.ppm tests/out.ppm --filter grayscale --verify"

.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: vars
vars:
	@echo "CXX        = $(CXX)"
	@echo "NVCC       = $(NVCC)"
	@echo "CUDA_ARCH = $(CUDA_ARCH)"
	@echo "CXXFLAGS  = $(CXXFLAGS)"
	@echo "NVCCFLAGS = $(NVCCFLAGS)"