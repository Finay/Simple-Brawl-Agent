# make            build everything (viewer needs GLFW: `brew install glfw`)
# make sim        build only the headless library and tests (no OpenGL needed)
# make test       run the simulation tests
# make play       build and launch the viewer

CXX      ?= c++
CXXFLAGS ?= -O2
CXXFLAGS += -std=c++17 -Wall -Wextra -fPIC -Isim/include -MMD -MP

BUILD := build
UNAME := $(shell uname -s)

SIM_SRCS := sim/src/geometry.cpp sim/src/sim.cpp sim/src/bot.cpp sim/src/policy.cpp
SIM_OBJS := $(SIM_SRCS:%.cpp=$(BUILD)/%.o)
API_OBJS := $(BUILD)/sim/src/c_api.o
APP_OBJS := $(BUILD)/app/main.o

ifeq ($(UNAME),Darwin)
  SHLIB     := $(BUILD)/libpfenv.dylib
  GLFW_DIR  ?= $(shell brew --prefix glfw 2>/dev/null)
  APP_FLAGS := -I$(GLFW_DIR)/include -DGL_SILENCE_DEPRECATION
  APP_LIBS  := -L$(GLFW_DIR)/lib -lglfw -framework OpenGL
else
  SHLIB     := $(BUILD)/libpfenv.so
  APP_FLAGS := $(shell pkg-config --cflags glfw3 2>/dev/null)
  APP_LIBS  := $(shell pkg-config --libs glfw3 2>/dev/null || echo -lglfw) -lGL
endif

.PHONY: all sim test play clean

all: sim $(BUILD)/pf_play

sim: $(SHLIB) $(BUILD)/pf_tests

test: $(BUILD)/pf_tests
	$(BUILD)/pf_tests

play: $(BUILD)/pf_play
	$(BUILD)/pf_play

$(BUILD)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(APP_OBJS): CXXFLAGS += $(APP_FLAGS)

$(SHLIB): $(SIM_OBJS) $(API_OBJS)
	$(CXX) -shared $^ -o $@

$(BUILD)/pf_tests: $(BUILD)/tests/sim_tests.o $(SIM_OBJS) $(API_OBJS)
	$(CXX) $^ -o $@

$(BUILD)/pf_play: $(APP_OBJS) $(SIM_OBJS)
	$(CXX) $^ $(APP_LIBS) -o $@

clean:
	rm -rf $(BUILD)

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)
