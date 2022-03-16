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

#define NR_PARTICLES 512

std::unique_ptr<taichi::lang::MemoryPool> memory_pool;
std::unique_ptr<taichi::lang::vulkan::VkRuntime> vulkan_runtime;
std::unique_ptr<taichi::lang::aot::Kernel> init_kernel;
std::unique_ptr<taichi::lang::aot::Kernel> get_vertices_kernel;
std::unique_ptr<taichi::lang::aot::Kernel> get_indices_kernel;
std::unique_ptr<taichi::lang::aot::Kernel> get_force_kernel;
std::unique_ptr<taichi::lang::aot::Kernel> advect_kernel;
std::unique_ptr<taichi::lang::aot::Kernel> floor_bound_kernel;
taichi::ui::vulkan::Renderer *renderer;
taichi::ui::vulkan::Gui *gui;
taichi::lang::DeviceAllocation dalloc_circles;
taichi::ui::CirclesInfo circles;
taichi::lang::RuntimeContext host_ctx;

extern "C" JNIEXPORT void JNICALL
Java_com_innopeaktech_naboo_taichi_1test_NativeLib_init(JNIEnv *env,
                                                        jclass,
                                                        jobject assets,
                                                        jobject surface) {
  ANativeWindow *native_window = ANativeWindow_fromSurface(env, surface);

  // Initialize our Vulkan Program pipeline
  taichi::uint64 *result_buffer{nullptr};

  // Create a memory pool to allocate GPU memory
  memory_pool = std::make_unique<taichi::lang::MemoryPool>(
      taichi::Arch::vulkan, nullptr);
  result_buffer = (taichi::uint64 *)memory_pool->allocate(
      sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);
  ALOGI("Created a memory pool");

  // Create a GGUI configuration
  taichi::ui::AppConfig app_config;
  app_config.name = "Explicit_FEM";
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
  ALOGI("Created a gui");

  // Create the Vk Runtime
  taichi::lang::vulkan::VkRuntime::Params params;
  params.host_result_buffer = result_buffer;
  params.device = &renderer->app_context().device();
  vulkan_runtime =
      std::make_unique<taichi::lang::vulkan::VkRuntime>(std::move(params));

  taichi::lang::vulkan::AotModuleParams aot_params{"/data/local/tmp/explicit_fem", vulkan_runtime.get()};
  auto module = taichi::lang::vulkan::make_aot_module(aot_params);
  // Retrieve kernels/fields/etc from AOT module so we can initialize our
  // runtime
  auto root_size = module->get_root_size();
  ALOGI("root buffer size=%d\n", root_size);

  vulkan_runtime->add_root_buffer(root_size);

  init_kernel = module->get_kernel("init");
  get_vertices_kernel = module->get_kernel("get_vertices");
  get_indices_kernel = module->get_kernel("get_indices");
  get_force_kernel = module->get_kernel("get_force");
  advect_kernel = module->get_kernel("advect");
  floor_bound_kernel = module->get_kernel("floor_bound");


  ALOGI("Register kernels");

  // Allocate memory for Circles position
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.size = NR_PARTICLES * sizeof(taichi::ui::Vertex);
  dalloc_circles =
    vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
  ALOGI("Allocate memory");

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
  f_info.dev_alloc    = dalloc_circles;

  circles.renderable_info.has_per_vertex_color = false;
  circles.renderable_info.vbo                  = f_info;
  circles.color                                = {0.8, 0.4, 0.1};
  circles.radius                               = 0.005f;


  host_ctx.set_arg(0, &dalloc_circles);
  host_ctx.set_device_allocation(0, true);
  host_ctx.extra_args[0][0] = 512;
  host_ctx.extra_args[0][1] = 3;
  host_ctx.extra_args[0][2] = 1;


  get_vertices_kernel->launch(&host_ctx);
  init_kernel->launch(&host_ctx);
  get_indices_kernel->launch(&host_ctx);
  vulkan_runtime->synchronize();
  ALOGI("launch kernel init");

  // Clear the background
  renderer->set_background_color({0.6, 0.6, 0.6});

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

  // timer starts before launch kernel
  auto start = std::chrono  ::steady_clock::now();
  ALOGI("before launch kernel substep");

  // Run 'substep' 40 times
  for (int i = 0; i < 40; i++) {
//      ALOGI("before launch force substep");
    get_force_kernel->launch(&host_ctx);
//    ALOGI("after launch force substep");
    advect_kernel->launch(&host_ctx);
  }
  floor_bound_kernel->launch(&host_ctx);
  ALOGI("launch kernel substep");

  // Make sure to sync the GPU memory so we can read the latest update from CPU
  // And read the 'x' calculated on GPU to our local variable
  // @TODO: Skip this with support of NdArray as we will be able to specify 'dalloc_circles'
  // in the host_ctx before running the kernel, allowing the data to be updated automatically
  vulkan_runtime->synchronize();

  // timer ends after synchronization
  auto end = std::chrono::steady_clock::now();
  auto cpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  ALOGI("Execution time is %" PRId64 "ns\n", cpu_time);

  // Render the UI
  renderer->circles(circles);
  renderer->draw_frame(gui);
  renderer->swap_chain().surface().present_image();
  renderer->prepare_for_next_frame();
}
