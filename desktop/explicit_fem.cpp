#include <signal.h>
#include <iostream>

#include <taichi/backends/vulkan/vulkan_program.h>
#include <taichi/backends/vulkan/vulkan_common.h>
#include <taichi/backends/vulkan/vulkan_loader.h>
#include <taichi/backends/vulkan/aot_module_loader_impl.h>
#include <inttypes.h>

#include <taichi/gui/gui.h>
#include <taichi/ui/backends/vulkan/renderer.h>

#define NR_PARTICLES 512

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
#include <unistd.h>
int main() {
    // Init gl window
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(512, 512, "Taichi show", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Create a GGUI configuration
    taichi::ui::AppConfig app_config;
    app_config.name         = "Explicit_FEM";
    app_config.width        = 512;
    app_config.height       = 512;
    app_config.vsync        = true;
    app_config.show_window  = false;
    app_config.package_path = "../"; // make it flexible later
    app_config.ti_arch      = taichi::Arch::vulkan;

    // // Create GUI & renderer
    auto renderer  = std::make_unique<taichi::ui::vulkan::Renderer>();
    renderer->init(nullptr, window, app_config);

    // Initialize our Vulkan Program pipeline
    taichi::uint64 *result_buffer{nullptr};
    auto memory_pool = std::make_unique<taichi::lang::MemoryPool>(taichi::Arch::vulkan, nullptr);
    result_buffer = (taichi::uint64 *)memory_pool->allocate(sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);
    taichi::lang::vulkan::VkRuntime::Params params;
    params.host_result_buffer = result_buffer;
    params.device = &(renderer->app_context().device());
    auto vulkan_runtime = taichi::lang::vulkan::VkRuntime(std::move(params));

    // Retrieve kernels/fields/etc from AOT module so we can initialize our
    // runtime
    taichi::lang::vulkan::AotModuleParams aot_params{"../explicit_fem/", &vulkan_runtime};
    auto module = taichi::lang::vulkan::make_aot_module(aot_params);
    auto root_size = module->get_root_size();
    printf("root buffer size=%ld\n", root_size);
    vulkan_runtime.add_root_buffer(root_size);

    auto init_kernel = module->get_kernel("init");
    auto get_vertices_kernel = module->get_kernel("get_vertices");
    auto get_indices_kernel = module->get_kernel("get_indices");
    auto get_force_kernel = module->get_kernel("get_force");
    auto advect_kernel = module->get_kernel("advect");
    auto floor_bound_kernel = module->get_kernel("floor_bound");

    // Prepare Ndarray for model
    taichi::lang::Device::AllocParams alloc_params;
    alloc_params.size = NR_PARTICLES * 3 * sizeof(float);
    // x
    taichi::lang::DeviceAllocation devalloc_x = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params);
    // v
    taichi::lang::DeviceAllocation devalloc_v = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params);

    taichi::lang::RuntimeContext host_ctx;
    memset(&host_ctx, 0, sizeof(taichi::lang::RuntimeContext));
    // x
    host_ctx.set_arg(0, &devalloc_x);
    host_ctx.set_device_allocation(0, true);
    host_ctx.extra_args[0][0] = 512;
    host_ctx.extra_args[0][1] = 3;
    host_ctx.extra_args[0][2] = 1;
    // v
    host_ctx.set_arg(1, &devalloc_v);
    host_ctx.set_device_allocation(1, true);
    host_ctx.extra_args[1][0] = 512;
    host_ctx.extra_args[1][1] = 3;
    host_ctx.extra_args[1][2] = 1;

    // Create a GUI even though it's not used in our case (required to
    // render the renderer)
    auto gui = std::make_shared<taichi::ui::vulkan::Gui>(&renderer->app_context(), &renderer->swap_chain(), window);

    get_vertices_kernel->launch(&host_ctx);
    init_kernel->launch(&host_ctx);
    get_indices_kernel->launch(&host_ctx);
    vulkan_runtime.synchronize();

    // Describe information to render the circle with Vulkan
    taichi::ui::FieldInfo f_info;
    f_info.valid        = true;
    f_info.field_type   = taichi::ui::FieldType::Scalar;
    f_info.matrix_rows  = 1;
    f_info.matrix_cols  = 1;
    f_info.shape        = {NR_PARTICLES};
    f_info.field_source = taichi::ui::FieldSource::TaichiVulkan;
    f_info.dtype        = taichi::lang::PrimitiveType::f32;
    f_info.snode        = nullptr;
    f_info.dev_alloc    = devalloc_x;
    taichi::ui::CirclesInfo circles;
    circles.renderable_info.has_per_vertex_color = false;
    circles.renderable_info.vbo                  = f_info;
    circles.color                                = {0.8, 0.4, 0.1};
    circles.radius                               = 0.005f; // 0.0015f looks unclear on desktop

    renderer->set_background_color({0.6, 0.6, 0.6});

    while (!glfwWindowShouldClose(window)) {
        // Run 'substep' 40 times
        for (int i = 0; i < 40; i++) {
            get_force_kernel->launch(&host_ctx);
            advect_kernel->launch(&host_ctx);
        }
        floor_bound_kernel->launch(&host_ctx);
        vulkan_runtime.synchronize();

        // Render elements
        renderer->circles(circles);
        renderer->draw_frame(gui.get());
        renderer->swap_chain().surface().present_image();
        renderer->prepare_for_next_frame();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    return 0;
}
