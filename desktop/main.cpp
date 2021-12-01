#include <signal.h>
#include <iostream>

#include <taichi/backends/vulkan/vulkan_common.h>
#include <taichi/backends/vulkan/vulkan_program.h>
#include <taichi/backends/vulkan/vulkan_loader.h>
#include <taichi/program/context.h>
#include <taichi/gui/gui.h>

std::vector<std::string> get_required_instance_extensions() {
  uint32_t glfw_ext_count = 0;
  const char **glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

  std::vector<std::string> extensions;

  for (int i = 0; i < glfw_ext_count; ++i) {
    extensions.push_back(glfw_extensions[i]);
  }

  // VulkanDeviceCreator will check that these are supported
  extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
  return extensions;
}

std::vector<std::string> get_required_device_extensions() {
  static std::vector<std::string> extensions {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
  };

  return extensions;
}

int main() {
    // Initialize our Vulkan Program pipeline
    taichi::uint64 *result_buffer{nullptr};
    taichi::lang::RuntimeContext host_ctx;
    taichi::lang::CompileConfig config = taichi::lang::default_compile_config;
    config.arch = taichi::lang::Arch::vulkan;

    taichi::lang::VulkanProgramImpl program(config, "../mpm88");
    auto memory_pool = std::make_unique<taichi::lang::MemoryPool>(config.arch, nullptr);
    result_buffer = (taichi::uint64 *)memory_pool->allocate(sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);

    // Create Taichi Device for computation
    taichi::lang::vulkan::VulkanDeviceCreator::Params evd_params;
    evd_params.api_version = taichi::lang::vulkan::VulkanEnvSettings::kApiVersion();
    evd_params.additional_instance_extensions =
        get_required_instance_extensions();
    evd_params.additional_device_extensions = get_required_device_extensions();
    auto embedded_device = std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);

    taichi::lang::vulkan::VkRuntime::Params params;
    params.host_result_buffer = result_buffer;
    params.device = embedded_device->device();
    auto vulkan_runtime = std::make_unique<taichi::lang::vulkan::VkRuntime>(std::move(params));

    // Retrieve kernels/fields/etc from AOT module so we can initialize our
    // runtime
    taichi::lang::vulkan::VkRuntime::RegisterParams init_kernel, substep_kernel;
    bool ret = program.get_kernel("init", init_kernel);
    if (!ret) {
        printf("Cannot find 'init' kernel\n");
        return -1;
    }
    ret = program.get_kernel("substep", substep_kernel);
    if (!ret) {
        printf("Cannot find 'substep' kernel\n");
        return -1;
    }
    auto root_size = program.get_root_size();
    printf("root buffer size=%d\n", root_size);

    vulkan_runtime->add_root_buffer(root_size);
    auto init_kernel_handle = vulkan_runtime->register_taichi_kernel(init_kernel);
    auto substep_kernel_handle = vulkan_runtime->register_taichi_kernel(substep_kernel);

    //
    // Run MPM88 from AOT module similar to Python code
    //
    vulkan_runtime->launch_kernel(init_kernel_handle, &host_ctx);
    vulkan_runtime->synchronize();

    // Sanity check to make sure the shaders are running properly, we should have the same float
    // values as the python scripts

    int n_particles = 8192;
    float *x = new float[n_particles * 2];

    // Create a small GUI to later render our circles information calculated by the AOT module
    taichi::GUI gui("GUI Test", 512, 512, true, false, 0, false, false);
    auto canvas = *gui.canvas;
    uint32_t * np = nullptr;

    while (1) {
        canvas.clear(0x112F41);
        
        for (int i = 0; i < 50; i++) {
            vulkan_runtime->launch_kernel(substep_kernel_handle, &host_ctx);
        }
        vulkan_runtime->synchronize();

        vulkan_runtime->read_memory((uint8_t*) x, 0, n_particles * 2 * sizeof(taichi::float32));

        for (int i = 0; i < n_particles * 2; ++i) {
            x[i] *= 512;
        }

        canvas.circles_batched(n_particles, (std::size_t)x, 0x068587, (std::size_t)np, 1.5, (std::size_t)np);

        gui.update();
    }
    delete []x;

    return 0;
}
