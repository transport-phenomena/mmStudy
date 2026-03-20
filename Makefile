CXX := g++
BUILD ?= release

WARNFLAGS := -Wall -Wextra -Wpedantic
STD_FLAGS := -std=c++17
CPPFLAGS := -I/opt/homebrew/include/eigen3 -Iinclude
LDFLAGS :=
LDLIBS :=

ifeq ($(BUILD),release)
OPTFLAGS := -O3 -DNDEBUG
BUILD_DIR := build/release
TARGET := bin/mmStudy_release
else ifeq ($(BUILD),debug)
OPTFLAGS := -O0 -g
BUILD_DIR := build/debug
TARGET := bin/mmStudy_debug
else
$(error Unsupported BUILD='$(BUILD)'. Use BUILD=debug or BUILD=release)
endif

CXXFLAGS := $(WARNFLAGS) $(STD_FLAGS) $(OPTFLAGS)

SRC_DIR := src
BIN_DIR := bin

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

.PHONY: all debug release clean

all: $(TARGET)

debug:
	$(MAKE) BUILD=debug

release:
	$(MAKE) BUILD=release

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) $(OBJS) $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf build/debug build/release bin/mmStudy_debug bin/mmStudy_release
