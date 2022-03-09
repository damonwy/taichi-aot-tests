#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <android/looper.h>
#include <android/native_window_jni.h>
#include <android/sensor.h>
#include <jni.h>

#include <taichi/backends/vulkan/vulkan_program.h>
#include <taichi/backends/vulkan/vulkan_common.h>
#include <taichi/backends/vulkan/vulkan_loader.h>
#include <taichi/backends/vulkan/aot_module_loader_impl.h>

#include <taichi/ui/backends/vulkan/renderer.h>

#include <assert.h>
#include <map>
#include <stdint.h>
#include <vector>

#define NX     512
#define NY     512
#define ITERS  400

#define ALOGI(fmt, ...)                                                  \
  ((void)__android_log_print(ANDROID_LOG_INFO, "TaichiTest", "%s: " fmt, \
                             __FUNCTION__, ##__VA_ARGS__))
#define ALOGE(fmt, ...)                                                   \
  ((void)__android_log_print(ANDROID_LOG_ERROR, "TaichiTest", "%s: " fmt, \
                             __FUNCTION__, ##__VA_ARGS__))

std::vector<std::string> get_required_instance_extensions() {
  std::vector<std::string> extensions;

  extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
  extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

  return extensions;
}

std::vector<std::string> get_required_device_extensions() {
  static std::vector<std::string> extensions{
      VK_KHR_SWAPCHAIN_EXTENSION_NAME,
  };

  return extensions;
}

#define NX     512
#define NY     512
#define ITERS  300

std::unique_ptr<taichi::lang::MemoryPool> memory_pool;
std::unique_ptr<taichi::lang::vulkan::VkRuntime> vulkan_runtime;

taichi::lang::vulkan::VkRuntime::KernelHandle generate_mouse_data_kernel_handle;
taichi::lang::vulkan::VkRuntime::KernelHandle advect_kernel_handle;
taichi::lang::vulkan::VkRuntime::KernelHandle apply_impulse_kernel_handle;
taichi::lang::vulkan::VkRuntime::KernelHandle divergence_kernel_handle;
taichi::lang::vulkan::VkRuntime::KernelHandle pressure_jacobi_kernel_handle;
taichi::lang::vulkan::VkRuntime::KernelHandle subtract_gradient_kernel_handle;
taichi::lang::vulkan::VkRuntime::KernelHandle dye_to_image_kernel_handle;

taichi::ui::vulkan::Renderer *renderer;
taichi::ui::vulkan::Gui *gui;

taichi::lang::DeviceAllocation dalloc_touch;
taichi::lang::DeviceAllocation dalloc_velocities;
taichi::lang::DeviceAllocation dalloc_new_velocities;
taichi::lang::DeviceAllocation dalloc_divs;
taichi::lang::DeviceAllocation dalloc_curls;
taichi::lang::DeviceAllocation dalloc_pressures;
taichi::lang::DeviceAllocation dalloc_new_pressures;
taichi::lang::DeviceAllocation dalloc_dye;
taichi::lang::DeviceAllocation dalloc_new_dye;
taichi::lang::DeviceAllocation dalloc_dye_image;
taichi::lang::DeviceAllocation dalloc_mouse_data;


taichi::lang::RuntimeContext host_ctx_advect_velocities;
taichi::lang::RuntimeContext host_ctx_advect_dye;
taichi::lang::RuntimeContext host_ctx_apply_impulse;
taichi::lang::RuntimeContext host_ctx_divergence;
taichi::lang::RuntimeContext host_ctx_generate_mouse_data;
taichi::lang::RuntimeContext host_ctx_pressure_jacobi;
taichi::lang::RuntimeContext host_ctx_subtract_gradient;
taichi::lang::RuntimeContext host_ctx_dye_to_image;

taichi::ui::SetImageInfo set_image_info;

bool is_velocities;
bool is_dye_buffer;
bool is_pressures;

extern "C" JNIEXPORT void JNICALL
Java_com_innopeaktech_naboo_taichi_1test_NativeLib_init(JNIEnv *env,
                                                        jclass,
                                                        jobject assets,
                                                        jobject surface) {
  ANativeWindow *native_window = ANativeWindow_fromSurface(env, surface);
  taichi::lang::vulkan::AotModuleLoaderImpl aot_loader("/data/local/tmp/stable_fluids");

  // Initialize our Vulkan Program pipeline
  taichi::uint64 *result_buffer{nullptr};

  // Create a memory pool to allocate GPU memory
  memory_pool = std::make_unique<taichi::lang::MemoryPool>(
      taichi::Arch::vulkan, nullptr);
  result_buffer = (taichi::uint64 *)memory_pool->allocate(
      sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);

  // Create a GGUI configuration
  taichi::ui::AppConfig app_config;
  app_config.name = "Stable Fluids";
  app_config.width = ANativeWindow_getWidth(native_window);
  app_config.height = ANativeWindow_getHeight(native_window);
  app_config.vsync = true;
  app_config.show_window = false;
  app_config.package_path = "/data/local/tmp/";  // Use CacheDir()
  app_config.ti_arch = taichi::Arch::vulkan;
  renderer = new taichi::ui::vulkan::Renderer();
  renderer->init(nullptr, native_window, app_config);

  // Create a GUI even though it's not used in our case (required to
  // render the renderer)
  gui = new taichi::ui::vulkan::Gui(&renderer->app_context(),
                                    &renderer->swap_chain(), native_window);

  // Create the Vk Runtime
  taichi::lang::vulkan::VkRuntime::Params params;
  params.host_result_buffer = result_buffer;
  params.device = &renderer->app_context().device();
  vulkan_runtime =
      std::make_unique<taichi::lang::vulkan::VkRuntime>(std::move(params));

  // Retrieve kernels/fields/etc from AOT module so we can initialize our
  // runtime
  taichi::lang::vulkan::VkRuntime::RegisterParams advect_kernel, apply_impulse_kernel,
      divergence_kernel, pressure_jacobi_kernel, 
      subtract_gradient_kernel, generate_mouse_data_kernel,
      dye_to_image_kernel;

  bool ret = aot_loader.get_kernel("generate_mouse_data", generate_mouse_data_kernel);
  if (!ret) {
      ALOGE("Cannot find 'generate_mouse_data' kernel\n");
      return;
  }

  ret = aot_loader.get_kernel("advect", advect_kernel);
  if (!ret) {
      ALOGE("Cannot find 'advect' kernel\n");
      return;
  }

  ret = aot_loader.get_kernel("apply_impulse", apply_impulse_kernel);
  if (!ret) {
      ALOGE("Cannot find 'apply_impulse' kernel\n");
      return;
  }

  ret = aot_loader.get_kernel("divergence", divergence_kernel);
  if (!ret) {
      ALOGE("Cannot find 'divergence' kernel\n");
      return;
  }

  ret = aot_loader.get_kernel("pressure_jacobi", pressure_jacobi_kernel);
  if (!ret) {
      ALOGE("Cannot find 'pressure_jacobi' kernel\n");
      return;
  }

  ret = aot_loader.get_kernel("subtract_gradient", subtract_gradient_kernel);
  if (!ret) {
      ALOGE("Cannot find 'subtract_gradient' kernel\n");
      return;
  }

  ret = aot_loader.get_kernel("dye_to_image", dye_to_image_kernel);
  if (!ret) {
      ALOGE("Cannot find 'dye_to_image' kernel\n");
      return;
  }

  auto root_size = aot_loader.get_root_size();
  ALOGI("root buffer size=%d\n", root_size);

  vulkan_runtime->add_root_buffer(root_size);

  generate_mouse_data_kernel_handle   = vulkan_runtime->register_taichi_kernel(generate_mouse_data_kernel);
  advect_kernel_handle                = vulkan_runtime->register_taichi_kernel(advect_kernel);
  apply_impulse_kernel_handle         = vulkan_runtime->register_taichi_kernel(apply_impulse_kernel);
  divergence_kernel_handle            = vulkan_runtime->register_taichi_kernel(divergence_kernel);
  pressure_jacobi_kernel_handle       = vulkan_runtime->register_taichi_kernel(pressure_jacobi_kernel);
  subtract_gradient_kernel_handle     = vulkan_runtime->register_taichi_kernel(subtract_gradient_kernel);
  dye_to_image_kernel_handle          = vulkan_runtime->register_taichi_kernel(dye_to_image_kernel);


  // Allocate memory for Circles position
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

  dalloc_touch            = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params_touch);
  dalloc_velocities       = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params_velocities);
  dalloc_new_velocities   = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params_new_velocities);
  dalloc_dye              = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params_dye);
  dalloc_new_dye          = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params_new_dye);
  dalloc_pressures        = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params_pressures);
  dalloc_new_pressures    = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params_new_pressures);
  dalloc_divs             = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params_divs);
  dalloc_curls            = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params_curls);
  dalloc_dye_image        = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params_dye_image);

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
  
  set_image_info.img = f_info;

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

  // Clear the background
  renderer->set_background_color({0.6, 0.6, 0.6});
  // Set flags 
  is_velocities = true;
  is_pressures = true;
  is_dye_buffer = true; 

#if 0
  // Sanity check to make sure the shaders are running properly, we should have
  // the same float values as the python scripts aot->get_field("x");
  float x[10];
  vulkan_runtime->synchronize();
  vulkan_runtime->read_memory((uint8_t *)x, 0, 5 * 2 * sizeof(taichi::float32));

  for (int i = 0; i < 10; i += 2) {
    ALOGI("[%f, %f]\n", x[i], x[i + 1]);
  }
#endif
}

extern "C" JNIEXPORT void JNICALL
Java_com_innopeaktech_naboo_taichi_1test_NativeLib_destroy(JNIEnv *env,
                                                           jclass,
                                                           jobject surface) {
  renderer->app_context().device().dealloc_memory(dalloc_dye);
  renderer->app_context().device().dealloc_memory(dalloc_new_dye);
  renderer->app_context().device().dealloc_memory(dalloc_velocities);
  renderer->app_context().device().dealloc_memory(dalloc_new_velocities);
  renderer->app_context().device().dealloc_memory(dalloc_pressures);
  renderer->app_context().device().dealloc_memory(dalloc_new_pressures);
  renderer->app_context().device().dealloc_memory(dalloc_divs);
  renderer->app_context().device().dealloc_memory(dalloc_curls);
  renderer->app_context().device().dealloc_memory(dalloc_dye_image);
  renderer->app_context().device().dealloc_memory(dalloc_mouse_data);
}

extern "C" JNIEXPORT void JNICALL
Java_com_innopeaktech_naboo_taichi_1test_NativeLib_pause(JNIEnv *env,
                                                         jclass,
                                                         jobject surface) {
}

extern "C" JNIEXPORT void JNICALL
Java_com_innopeaktech_naboo_taichi_1test_NativeLib_resume(JNIEnv *env,
                                                          jclass,
                                                          jobject surface) {
}

extern "C" JNIEXPORT void JNICALL
Java_com_innopeaktech_naboo_taichi_1test_NativeLib_resize(JNIEnv *,
                                                          jclass,
                                                          jobject,
                                                          jint width,
                                                          jint height) {
  ALOGI("Resize requested for %dx%d", width, height);
}

extern "C" JNIEXPORT void JNICALL
Java_com_innopeaktech_naboo_taichi_1test_NativeLib_render(JNIEnv *env,
                                                          jclass,
                                                          jobject surface) {
  // Generate user inputs randomly
  float x_pos = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * NX;
  float y_pos = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * NY;

  taichi::float32 *touch_buffer = reinterpret_cast<taichi::float32*>(renderer->app_context().device().map(dalloc_touch));
  touch_buffer[0] = x_pos;
  touch_buffer[1] = y_pos;

  vulkan_runtime->launch_kernel(generate_mouse_data_kernel_handle, &host_ctx_generate_mouse_data);
  renderer->app_context().device().unmap(dalloc_touch);
  host_ctx_apply_impulse.set_arg(0, is_velocities? &dalloc_velocities : &dalloc_new_velocities);
  host_ctx_apply_impulse.set_arg(1, (is_dye_buffer) ? (&dalloc_dye) : (&dalloc_new_dye));
  vulkan_runtime->launch_kernel(apply_impulse_kernel_handle, &host_ctx_apply_impulse);   

  host_ctx_advect_velocities.set_arg(0, is_velocities? &dalloc_velocities : &dalloc_new_velocities);
  host_ctx_advect_velocities.set_arg(1, is_velocities? &dalloc_velocities : &dalloc_new_velocities);
  host_ctx_advect_velocities.set_arg(2, is_velocities? &dalloc_new_velocities : &dalloc_velocities);
  vulkan_runtime->launch_kernel(advect_kernel_handle, &host_ctx_advect_velocities);

  host_ctx_advect_dye.set_arg(1, is_dye_buffer? &dalloc_dye : &dalloc_new_dye);
  host_ctx_advect_dye.set_arg(2, is_dye_buffer? &dalloc_new_dye : &dalloc_dye);    
  vulkan_runtime->launch_kernel(advect_kernel_handle, &host_ctx_advect_dye);

  is_velocities = !is_velocities;
  is_dye_buffer = !is_dye_buffer;

  host_ctx_divergence.set_arg(0, is_velocities? &dalloc_velocities : &dalloc_new_velocities);
  vulkan_runtime->launch_kernel(divergence_kernel_handle, &host_ctx_divergence);

  for (int i = 0; i < ITERS; i++) {
      host_ctx_pressure_jacobi.set_arg(0, is_pressures? &dalloc_pressures : &dalloc_new_pressures);
      host_ctx_pressure_jacobi.set_arg(1, is_pressures? &dalloc_new_pressures : &dalloc_pressures);
      vulkan_runtime->launch_kernel(pressure_jacobi_kernel_handle, &host_ctx_pressure_jacobi);
      is_pressures = !is_pressures;
  }

  host_ctx_subtract_gradient.set_arg(0, is_velocities ? &dalloc_velocities : &dalloc_new_velocities);
  host_ctx_subtract_gradient.set_arg(1, is_pressures ? &dalloc_pressures : &dalloc_new_pressures);
  vulkan_runtime->launch_kernel(subtract_gradient_kernel_handle, &host_ctx_subtract_gradient);

  host_ctx_dye_to_image.set_arg(0, (is_dye_buffer) ? (&dalloc_dye) : (&dalloc_new_dye));
  vulkan_runtime->launch_kernel(dye_to_image_kernel_handle, &host_ctx_dye_to_image);

  vulkan_runtime->synchronize();

  renderer->set_image(set_image_info);
  renderer->draw_frame(gui);
  renderer->swap_chain().surface().present_image();
  renderer->prepare_for_next_frame();
}
