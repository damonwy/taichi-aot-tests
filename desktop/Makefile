CPPFLAGS=-Iinclude -I$(TAICHI_REPO_DIR)/ -I$(TAICHI_REPO_DIR)/external/SPIRV-Tools/include/ -I$(TAICHI_REPO_DIR)/taichi/backends/vulkan/ -I$(TAICHI_REPO_DIR)/external/Vulkan-Headers/include/ -I$(TAICHI_REPO_DIR)/external/volk/ -I$(TAICHI_REPO_DIR)/external/spdlog/include/ -I$(TAICHI_REPO_DIR)/external/VulkanMemoryAllocator/include/ -I$(TAICHI_REPO_DIR)/external/eigen -DTI_ARCH_x64 -DTI_WITH_VULKAN -DTI_INCLUDED -std=c++17 -fsized-deallocation -Wall -DNDEBUG -g
LDFLAGS=$(TAICHI_REPO_DIR)/build_no_llvm/libtaichi_export_core.so -lpython3.8

SRCS=main.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

ifndef TAICHI_REPO_DIR
$(error TAICHI_REPO_DIR is not set)
endif

all: taichi-vk-launcher

taichi-vk-launcher: $(OBJS)
	clang++-10 $(OBJS) -o taichi-vk-launcher $(LDFLAGS)

%.o: %.cpp
	clang++-10 $(CPPFLAGS) -c $<

clean:
	$(RM) $(OBJS) taichi-vk-launcher
