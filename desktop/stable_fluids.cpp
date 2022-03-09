#include <signal.h>
#include <iostream>

#include <taichi/backends/vulkan/vulkan_program.h>
#include <taichi/backends/vulkan/vulkan_common.h>
#include <taichi/backends/vulkan/vulkan_loader.h>
#include <taichi/backends/vulkan/aot_module_loader_impl.h>
#include <inttypes.h>

#include <taichi/gui/gui.h>
#include <taichi/ui/backends/vulkan/renderer.h>

#define NX     512
#define NY     512
#define ITERS  400

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
    app_config.name         = "Stable Fluids";
    app_config.width        = 512;
    app_config.height       = 512;
    app_config.vsync        = true;
    app_config.show_window  = false;
    app_config.package_path = "../"; // make it flexible later
    app_config.ti_arch      = taichi::Arch::vulkan;

    // Create GUI & renderer
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
    taichi::lang::vulkan::AotModuleLoaderImpl aot_loader("../stable_fluids/");
    taichi::lang::vulkan::VkRuntime::RegisterParams advect_kernel, apply_impulse_kernel,
        divergence_kernel, pressure_jacobi_kernel, 
        subtract_gradient_kernel, generate_mouse_data_kernel,
        dye_to_image_kernel;

    bool ret = aot_loader.get_kernel("generate_mouse_data", generate_mouse_data_kernel);
    if (!ret) {
        printf("Cannot find 'generate_mouse_data' kernel\n");
    }

    ret = aot_loader.get_kernel("advect", advect_kernel);
    if (!ret) {
        printf("Cannot find 'advect' kernel\n");
    }

    ret = aot_loader.get_kernel("apply_impulse", apply_impulse_kernel);
    if (!ret) {
        printf("Cannot find 'apply_impulse' kernel\n");
    }

    ret = aot_loader.get_kernel("divergence", divergence_kernel);
    if (!ret) {
        printf("Cannot find 'divergence' kernel\n");
    }

    ret = aot_loader.get_kernel("pressure_jacobi", pressure_jacobi_kernel);
    if (!ret) {
        printf("Cannot find 'pressure_jacobi' kernel\n");
    }

    ret = aot_loader.get_kernel("subtract_gradient", subtract_gradient_kernel);
    if (!ret) {
        printf("Cannot find 'subtract_gradient' kernel\n");
    }

    ret = aot_loader.get_kernel("dye_to_image", dye_to_image_kernel);
    if (!ret) {
        printf("Cannot find 'dye_to_image' kernel\n");
    }

    auto root_size = aot_loader.get_root_size();
    printf("root buffer size=%ld\n", root_size);
    vulkan_runtime.add_root_buffer(root_size);

    auto generate_mouse_data_kernel_handle   = vulkan_runtime.register_taichi_kernel(generate_mouse_data_kernel);
    auto advect_kernel_handle                = vulkan_runtime.register_taichi_kernel(advect_kernel);
    auto apply_impulse_kernel_handle         = vulkan_runtime.register_taichi_kernel(apply_impulse_kernel);
    auto divergence_kernel_handle            = vulkan_runtime.register_taichi_kernel(divergence_kernel);
    auto pressure_jacobi_kernel_handle       = vulkan_runtime.register_taichi_kernel(pressure_jacobi_kernel);
    auto subtract_gradient_kernel_handle     = vulkan_runtime.register_taichi_kernel(subtract_gradient_kernel);
    auto dye_to_image_kernel_handle          = vulkan_runtime.register_taichi_kernel(dye_to_image_kernel);

    // Prepare Ndarray for model

    // Allocate memory for image buffer 
    taichi::lang::Device::AllocParams alloc_params_dye_image, alloc_params_dye, 
        alloc_params_mouse_data, alloc_params_new_dye,
        alloc_params_velocities, alloc_params_new_velocities,
        alloc_params_pressures, alloc_params_new_pressures,
        alloc_params_divs, alloc_params_curls,
        alloc_params_touch;

    alloc_params_touch.size             = 2 * sizeof(taichi::float32);    
    alloc_params_velocities.size        = NX * NY * sizeof(taichi::float32) * 2;
    alloc_params_new_velocities.size    = NX * NY * sizeof(taichi::float32) * 2;
    alloc_params_dye.size               = NX * NY * sizeof(taichi::float32) * 3;
    alloc_params_new_dye.size           = NX * NY * sizeof(taichi::float32) * 3;
    alloc_params_pressures.size         = NX * NY * sizeof(taichi::float32);
    alloc_params_new_pressures.size     = NX * NY * sizeof(taichi::float32);
    alloc_params_divs.size              = NX * NY * sizeof(taichi::float32);
    alloc_params_curls.size             = NX * NY * sizeof(taichi::float32);
    alloc_params_dye_image.size         = NX * NY * sizeof(taichi::float32) * 4;

    taichi::lang::DeviceAllocation dalloc_velocities, dalloc_new_velocities, 
        dalloc_dye, dalloc_new_dye,
        dalloc_pressures, dalloc_new_pressures,
        dalloc_divs, dalloc_curls,
        dalloc_touch, dalloc_dye_image;

    dalloc_touch            = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params_touch);
    dalloc_velocities       = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params_velocities);
    dalloc_new_velocities   = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params_new_velocities);
    dalloc_dye              = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params_dye);
    dalloc_new_dye          = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params_new_dye);
    dalloc_pressures        = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params_pressures);
    dalloc_new_pressures    = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params_new_pressures);
    dalloc_divs             = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params_divs);
    dalloc_curls            = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params_curls);
    dalloc_dye_image        = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params_dye_image);
    

    // Describe information to render the image with Vulkan
    taichi::ui::FieldInfo f_info;
    f_info.valid        = true;
    f_info.field_type   = taichi::ui::FieldType::Scalar;
    f_info.matrix_rows  = 1;
    f_info.matrix_cols  = 1;
    f_info.shape        = {NX, NY}; // Dimensions from taichi python kernels
    f_info.field_source = taichi::ui::FieldSource::TaichiVulkan;
    f_info.dtype        = taichi::lang::PrimitiveType::f32;
    f_info.snode        = nullptr;
    f_info.dev_alloc    = dalloc_dye_image;

    taichi::ui::SetImageInfo set_image_info;
    set_image_info.img = f_info;

    taichi::lang::RuntimeContext host_ctx_advect_velocities;
    taichi::lang::RuntimeContext host_ctx_advect_dye;
    taichi::lang::RuntimeContext host_ctx_apply_impulse;
    taichi::lang::RuntimeContext host_ctx_divergence;
    taichi::lang::RuntimeContext host_ctx_generate_mouse_data;
    taichi::lang::RuntimeContext host_ctx_pressure_jacobi;
    taichi::lang::RuntimeContext host_ctx_subtract_gradient;
    taichi::lang::RuntimeContext host_ctx_dye_to_image;

    memset(&host_ctx_generate_mouse_data, 0, sizeof(taichi::lang::RuntimeContext));
    memset(&host_ctx_advect_velocities, 0, sizeof(taichi::lang::RuntimeContext));
    memset(&host_ctx_advect_dye, 0, sizeof(taichi::lang::RuntimeContext));
    memset(&host_ctx_divergence, 0, sizeof(taichi::lang::RuntimeContext));
    memset(&host_ctx_apply_impulse, 0, sizeof(taichi::lang::RuntimeContext));
    memset(&host_ctx_subtract_gradient, 0, sizeof(taichi::lang::RuntimeContext));
    memset(&host_ctx_pressure_jacobi, 0, sizeof(taichi::lang::RuntimeContext));
    memset(&host_ctx_dye_to_image, 0, sizeof(taichi::lang::RuntimeContext));

    host_ctx_generate_mouse_data.set_arg(0, &dalloc_touch);
    host_ctx_generate_mouse_data.set_device_allocation(0, true);
    host_ctx_generate_mouse_data.extra_args[0][0] = 1;
    host_ctx_generate_mouse_data.extra_args[0][1] = 1;
    host_ctx_generate_mouse_data.extra_args[0][2] = 2;

    host_ctx_advect_velocities.set_arg(0, &dalloc_velocities);
    host_ctx_advect_velocities.set_arg(1, &dalloc_velocities);
    host_ctx_advect_velocities.set_arg(2, &dalloc_new_velocities);
    host_ctx_advect_velocities.set_device_allocation(0, true);
    host_ctx_advect_velocities.set_device_allocation(1, true);
    host_ctx_advect_velocities.set_device_allocation(2, true);
    host_ctx_advect_velocities.extra_args[0][0] = NX;
    host_ctx_advect_velocities.extra_args[0][1] = NY;
    host_ctx_advect_velocities.extra_args[0][2] = 2;
    host_ctx_advect_velocities.extra_args[1][0] = NX;
    host_ctx_advect_velocities.extra_args[1][1] = NY;
    host_ctx_advect_velocities.extra_args[1][2] = 2;
    host_ctx_advect_velocities.extra_args[2][0] = NX;
    host_ctx_advect_velocities.extra_args[2][1] = NY;
    host_ctx_advect_velocities.extra_args[2][2] = 2;

    host_ctx_advect_dye.set_arg(0, &dalloc_velocities);
    host_ctx_advect_dye.set_arg(1, &dalloc_dye);
    host_ctx_advect_dye.set_arg(2, &dalloc_new_dye);
    host_ctx_advect_dye.set_device_allocation(0, true);
    host_ctx_advect_dye.set_device_allocation(1, true);
    host_ctx_advect_dye.set_device_allocation(2, true);
    host_ctx_advect_dye.extra_args[0][0] = NX;
    host_ctx_advect_dye.extra_args[0][1] = NY;
    host_ctx_advect_dye.extra_args[0][2] = 2;
    host_ctx_advect_dye.extra_args[1][0] = NX;
    host_ctx_advect_dye.extra_args[1][1] = NY;
    host_ctx_advect_dye.extra_args[1][2] = 3;
    host_ctx_advect_dye.extra_args[2][0] = NX;
    host_ctx_advect_dye.extra_args[2][1] = NY;
    host_ctx_advect_dye.extra_args[2][2] = 3;

    host_ctx_apply_impulse.set_arg(0, &dalloc_velocities);
    host_ctx_apply_impulse.set_arg(1, &dalloc_dye);
    host_ctx_apply_impulse.set_device_allocation(0, true);
    host_ctx_apply_impulse.set_device_allocation(1, true);
    host_ctx_apply_impulse.extra_args[0][0] = NX;
    host_ctx_apply_impulse.extra_args[0][1] = NY;
    host_ctx_apply_impulse.extra_args[0][2] = 2;
    host_ctx_apply_impulse.extra_args[1][0] = NX;
    host_ctx_apply_impulse.extra_args[1][1] = NY;
    host_ctx_apply_impulse.extra_args[1][2] = 3;

    host_ctx_subtract_gradient.set_arg(0, &dalloc_velocities);
    host_ctx_subtract_gradient.set_arg(1, &dalloc_pressures);
    host_ctx_subtract_gradient.set_device_allocation(0, true);
    host_ctx_subtract_gradient.set_device_allocation(1, true);
    host_ctx_subtract_gradient.extra_args[0][0] = NX;
    host_ctx_subtract_gradient.extra_args[0][1] = NY;
    host_ctx_subtract_gradient.extra_args[0][2] = 2;
    host_ctx_subtract_gradient.extra_args[1][0] = NX;
    host_ctx_subtract_gradient.extra_args[1][1] = NY;
    host_ctx_subtract_gradient.extra_args[1][2] = 1;

    host_ctx_pressure_jacobi.set_arg(0, &dalloc_pressures);
    host_ctx_pressure_jacobi.set_arg(1, &dalloc_new_pressures);
    host_ctx_pressure_jacobi.set_arg(2, &dalloc_divs);
    host_ctx_pressure_jacobi.set_device_allocation(0, true);
    host_ctx_pressure_jacobi.set_device_allocation(1, true);
    host_ctx_pressure_jacobi.set_device_allocation(2, true);
    host_ctx_pressure_jacobi.extra_args[0][0] = NX;
    host_ctx_pressure_jacobi.extra_args[0][1] = NY;
    host_ctx_pressure_jacobi.extra_args[0][2] = 1;
    host_ctx_pressure_jacobi.extra_args[1][0] = NX;
    host_ctx_pressure_jacobi.extra_args[1][1] = NY;
    host_ctx_pressure_jacobi.extra_args[1][2] = 1;
    host_ctx_pressure_jacobi.extra_args[2][0] = NX;
    host_ctx_pressure_jacobi.extra_args[2][1] = NY;
    host_ctx_pressure_jacobi.extra_args[2][2] = 1;

    host_ctx_divergence.set_arg(0, &dalloc_velocities);
    host_ctx_divergence.set_arg(1, &dalloc_divs);
    host_ctx_divergence.set_device_allocation(0, true);
    host_ctx_divergence.set_device_allocation(1, true);
    host_ctx_divergence.extra_args[0][0] = NX;
    host_ctx_divergence.extra_args[0][1] = NY;
    host_ctx_divergence.extra_args[0][2] = 2;
    host_ctx_divergence.extra_args[1][0] = NX;
    host_ctx_divergence.extra_args[1][1] = NY;
    host_ctx_divergence.extra_args[1][2] = 1;

    host_ctx_dye_to_image.set_arg(0, &dalloc_dye);
    host_ctx_dye_to_image.set_arg(1, &dalloc_dye_image);
    host_ctx_dye_to_image.set_device_allocation(0, true);
    host_ctx_dye_to_image.set_device_allocation(1, true);
    host_ctx_dye_to_image.extra_args[0][0] = NX;
    host_ctx_dye_to_image.extra_args[0][1] = NY;
    host_ctx_dye_to_image.extra_args[0][2] = 3;
    host_ctx_dye_to_image.extra_args[1][0] = NX;
    host_ctx_dye_to_image.extra_args[1][1] = NY;
    host_ctx_dye_to_image.extra_args[1][2] = 4;

    // Create a GUI even though it's not used in our case (required to
    // render the renderer)
    auto gui = std::make_shared<taichi::ui::vulkan::Gui>(&renderer->app_context(), &renderer->swap_chain(), window);

    // Set flags 
    bool is_velocities = true;
    bool is_pressures = true;
    bool is_dye_buffer = true;

    renderer->set_background_color({0.6, 0.6, 0.6});

    while (!glfwWindowShouldClose(window)) {
        // Generate user inputs randomly
        float x_pos = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * NX;
        float y_pos = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * NY;

        taichi::float32 *touch_buffer = reinterpret_cast<taichi::float32*>(renderer->app_context().device().map(dalloc_touch));
        touch_buffer[0] = x_pos;
        touch_buffer[1] = y_pos;

        vulkan_runtime.launch_kernel(generate_mouse_data_kernel_handle, &host_ctx_generate_mouse_data);
        renderer->app_context().device().unmap(dalloc_touch);
        host_ctx_apply_impulse.set_arg(0, is_velocities? &dalloc_velocities : &dalloc_new_velocities);
        host_ctx_apply_impulse.set_arg(1, (is_dye_buffer) ? (&dalloc_dye) : (&dalloc_new_dye));
        vulkan_runtime.launch_kernel(apply_impulse_kernel_handle, &host_ctx_apply_impulse);   
        

        host_ctx_advect_velocities.set_arg(0, is_velocities? &dalloc_velocities : &dalloc_new_velocities);
        host_ctx_advect_velocities.set_arg(1, is_velocities? &dalloc_velocities : &dalloc_new_velocities);
        host_ctx_advect_velocities.set_arg(2, is_velocities? &dalloc_new_velocities : &dalloc_velocities);
        vulkan_runtime.launch_kernel(advect_kernel_handle, &host_ctx_advect_velocities);

        host_ctx_advect_dye.set_arg(1, is_dye_buffer? &dalloc_dye : &dalloc_new_dye);
        host_ctx_advect_dye.set_arg(2, is_dye_buffer? &dalloc_new_dye : &dalloc_dye);    
        vulkan_runtime.launch_kernel(advect_kernel_handle, &host_ctx_advect_dye);

        is_velocities = !is_velocities;
        is_dye_buffer = !is_dye_buffer;

        host_ctx_divergence.set_arg(0, is_velocities? &dalloc_velocities : &dalloc_new_velocities);
        vulkan_runtime.launch_kernel(divergence_kernel_handle, &host_ctx_divergence);

        for (int i = 0; i < ITERS; i++) {
            host_ctx_pressure_jacobi.set_arg(0, is_pressures? &dalloc_pressures : &dalloc_new_pressures);
            host_ctx_pressure_jacobi.set_arg(1, is_pressures? &dalloc_new_pressures : &dalloc_pressures);
            vulkan_runtime.launch_kernel(pressure_jacobi_kernel_handle, &host_ctx_pressure_jacobi);
            is_pressures = !is_pressures;
        }

        host_ctx_subtract_gradient.set_arg(0, is_velocities ? &dalloc_velocities : &dalloc_new_velocities);
        host_ctx_subtract_gradient.set_arg(1, is_pressures ? &dalloc_pressures : &dalloc_new_pressures);
        vulkan_runtime.launch_kernel(subtract_gradient_kernel_handle, &host_ctx_subtract_gradient);

        host_ctx_dye_to_image.set_arg(0, (is_dye_buffer) ? (&dalloc_dye) : (&dalloc_new_dye));
        vulkan_runtime.launch_kernel(dye_to_image_kernel_handle, &host_ctx_dye_to_image);

        vulkan_runtime.synchronize();

        renderer->set_image(set_image_info);
        renderer->draw_frame(gui.get());
        renderer->swap_chain().surface().present_image();
        renderer->prepare_for_next_frame();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    return 0;
}
