#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <android/looper.h>
#include <android/native_window_jni.h>
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
#include <chrono>

// Uncomment this line to show explicit fem deomo
//#define USE_EXPLICIT


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

void set_ctx_arg_devalloc(taichi::lang::RuntimeContext &host_ctx, int arg_id, taichi::lang::DeviceAllocation& alloc, int n=512, int m=3) {
  host_ctx.set_arg(arg_id, &alloc);
  host_ctx.set_device_allocation(arg_id, true);
  // This is hack since our ndarrays happen to have exactly the same size in implicit_fem demo.
  host_ctx.extra_args[arg_id][0] = n;
  host_ctx.extra_args[arg_id][1] = m;
}

// TODO: provide a proper API from taichi
void set_ctx_arg_float(taichi::lang::RuntimeContext &host_ctx, int arg_id, float x) {
  host_ctx.set_arg(arg_id, x);
  host_ctx.set_device_allocation(arg_id, false);
}

#define NR_PARTICLES 512

std::unique_ptr<taichi::lang::MemoryPool> memory_pool;
std::unique_ptr<taichi::lang::vulkan::VkRuntime> vulkan_runtime;
taichi::lang::aot::Kernel* init_kernel;
taichi::lang::aot::Kernel* get_vertices_kernel;
taichi::lang::aot::Kernel* get_indices_kernel;
taichi::lang::aot::Kernel* get_force_kernel;
taichi::lang::aot::Kernel* advect_kernel;
taichi::lang::aot::Kernel* floor_bound_kernel;
taichi::lang::aot::Kernel* get_b_kernel;
taichi::lang::aot::Kernel* matmul_cell_kernel;
taichi::lang::aot::Kernel* ndarray_to_ndarray_kernel;
taichi::lang::aot::Kernel* fill_ndarray_kernel;
taichi::lang::aot::Kernel* add_ndarray_kernel;
taichi::lang::aot::Kernel* dot_kernel;
taichi::lang::aot::Kernel* add_kernel;
taichi::lang::aot::Kernel* update_alpha_kernel;
taichi::lang::aot::Kernel* update_beta_r_2_kernel;
taichi::lang::aot::Kernel* add_hack_kernel;
taichi::lang::aot::Kernel* dot2scalar_kernel;
taichi::lang::aot::Kernel* init_r_2_kernel;

std::unique_ptr<taichi::lang::aot::Module> module;
taichi::ui::vulkan::Renderer *renderer;
taichi::ui::vulkan::Gui *gui;
taichi::lang::DeviceAllocation dalloc_circles;
taichi::lang::DeviceAllocation dalloc_v;
taichi::lang::DeviceAllocation dalloc_f;
taichi::lang::DeviceAllocation dalloc_b;
taichi::lang::DeviceAllocation dalloc_mul_ans;
taichi::lang::DeviceAllocation dalloc_r0;
taichi::lang::DeviceAllocation dalloc_p0;
taichi::lang::DeviceAllocation dalloc_alpha;
taichi::lang::DeviceAllocation dalloc_beta;
taichi::lang::DeviceAllocation dalloc_ox;
//taichi::ui::CirclesInfo circles;
taichi::ui::RenderableInfo r_info;
taichi::lang::RuntimeContext host_ctx;
std::unique_ptr<taichi::ui::SceneBase> scene;
taichi::ui::ParticlesInfo p_info;
taichi::ui::Camera camera;
ANativeWindow *native_window;

extern "C" JNIEXPORT void JNICALL
Java_com_innopeaktech_naboo_taichi_1test_NativeLib_init(JNIEnv *env,
                                                        jclass,
                                                        jobject assets,
                                                        jobject surface) {
  native_window = ANativeWindow_fromSurface(env, surface);

  // Initialize our Vulkan Program pipeline
  taichi::uint64 *result_buffer{nullptr};

  // Create a memory pool to allocate GPU memory
  memory_pool = std::make_unique<taichi::lang::MemoryPool>(
      taichi::Arch::vulkan, nullptr);
  result_buffer = (taichi::uint64 *)memory_pool->allocate(
      sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);
  ALOGI("Created a memory pool");

  memset(&host_ctx, 0, sizeof(taichi::lang::RuntimeContext));
  host_ctx.result_buffer = result_buffer;
  // Create a GGUI configuration
  taichi::ui::AppConfig app_config;
  app_config.name = "FEM";
  app_config.width = ANativeWindow_getWidth(native_window);
  app_config.height = ANativeWindow_getHeight(native_window);
  app_config.vsync = true;
  app_config.show_window = false;
  app_config.package_path = "/data/local/tmp/";  // Use CacheDir()
  app_config.ti_arch = taichi::Arch::vulkan;
  renderer = new taichi::ui::vulkan::Renderer();
  renderer->init(nullptr, native_window, app_config);
  ALOGI("width %d height %d", app_config.width, app_config.height);

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

#ifdef USE_EXPLICIT
    std::string shader_source = "/data/local/tmp/explicit_fem";
#else
    std::string shader_source = "/data/local/tmp/implicit_fem";
#endif

  taichi::lang::vulkan::AotModuleParams aot_params{shader_source, vulkan_runtime.get()};
  module = taichi::lang::aot::Module::load(taichi::Arch::vulkan, aot_params);
  // Retrieve kernels/fields/etc from AOT module so we can initialize our
  // runtime
  auto root_size = module->get_root_size();
  ALOGI("root buffer size=%d\n", root_size);

  vulkan_runtime->add_root_buffer(root_size);
  get_vertices_kernel = module->get_kernel("get_vertices");
  init_kernel = module->get_kernel("init");
  get_indices_kernel = module->get_kernel("get_indices");

  get_force_kernel = module->get_kernel("get_force");
  advect_kernel = module->get_kernel("advect");
  floor_bound_kernel = module->get_kernel("floor_bound");

  get_b_kernel = module->get_kernel("get_b");
  matmul_cell_kernel = module->get_kernel("matmul_cell");
  add_kernel = module->get_kernel("add");
  ndarray_to_ndarray_kernel = module->get_kernel("ndarray_to_ndarray");
  dot_kernel = module->get_kernel("dot");
  add_ndarray_kernel = module->get_kernel("add_ndarray");
  fill_ndarray_kernel = module->get_kernel("fill_ndarray");
  init_r_2_kernel = module->get_kernel("init_r_2");
  update_alpha_kernel = module->get_kernel("update_alpha");
  update_beta_r_2_kernel = module->get_kernel("update_beta_r_2");
  add_hack_kernel = module->get_kernel("add_hack");
  dot2scalar_kernel = module->get_kernel("dot2scalar");


  ALOGI("Register kernels");

  // Allocate memory for Circles position
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.size = NR_PARTICLES * 3 * sizeof(float);
  ALOGI("allocated %d", alloc_params.size);
  dalloc_circles = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
  dalloc_v = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
  dalloc_f = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
  dalloc_b = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
  dalloc_mul_ans = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
  dalloc_r0 = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
  dalloc_p0 = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
  dalloc_ox = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);

  alloc_params.size = sizeof(float);
  dalloc_alpha = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
  dalloc_beta = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);

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

  r_info.vbo = f_info;
  r_info.has_per_vertex_color = false;
  r_info.vbo_attrs = taichi::ui::VertexAttributes::kPos;
  p_info.renderable_info = r_info;
  p_info.color = {1.0, 1.0, 1.0};
  p_info.radius = 0.008;
  camera.position = {4,0,0};
  camera.lookat = {0,0,0};
  camera.up = {0, 1, 0};
  scene = std::make_unique<taichi::ui::SceneBase>();
  renderer->set_background_color({0.6, 0.6, 0.6});


  // get_vertices(ox)
  set_ctx_arg_devalloc(host_ctx, 0, dalloc_ox);
  get_vertices_kernel->launch(&host_ctx);

  set_ctx_arg_devalloc(host_ctx, 0, dalloc_circles);
  set_ctx_arg_devalloc(host_ctx, 1, dalloc_v);
  set_ctx_arg_devalloc(host_ctx, 2, dalloc_f);
  set_ctx_arg_devalloc(host_ctx, 3, dalloc_ox);
  // init(x, v, f, ox)
  init_kernel->launch(&host_ctx);
  // get_indices(x)
  get_indices_kernel->launch(&host_ctx);

  vulkan_runtime->synchronize();
  ALOGI("launch kernel init");

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
                                                          jobject surface,
                                                          float g_x,
                                                          float g_y,
                                                          float g_z
                                                          ) {
  // timer starts before launch kernel
  auto start = std::chrono  ::steady_clock::now();

  float dt = 1e-2;
#ifdef USE_EXPLICIT
  // Run 'substep' 40 times
  for (int i = 0; i < 40; i++) {
    //get_force(x, f)
    set_ctx_arg_devalloc(host_ctx, 0, dalloc_circles);
    set_ctx_arg_devalloc(host_ctx, 1, dalloc_f);
    get_force_kernel->launch(&host_ctx);
    // advect(x, v, f)
    set_ctx_arg_devalloc(host_ctx, 0, dalloc_circles);
    set_ctx_arg_devalloc(host_ctx, 1, dalloc_v);
    set_ctx_arg_devalloc(host_ctx, 2, dalloc_f);
    advect_kernel->launch(&host_ctx);
  }
#else
  // get_force(x, f)
  ALOGI("Acceleration: g_x = %f, g_y = %f, g_z = %f", g_x, g_y, g_z);
  float a_x = g_z;
  float a_y = g_y > 2 || g_y < -2 ? -g_y : 0;
  float a_z = g_x > 2 || g_x < -2 ? g_x * 4 : 0;
  set_ctx_arg_devalloc(host_ctx, 0, dalloc_circles);
  set_ctx_arg_devalloc(host_ctx, 1, dalloc_f);
  set_ctx_arg_float(host_ctx, 2, a_x);
  set_ctx_arg_float(host_ctx, 3, a_y);
  set_ctx_arg_float(host_ctx, 4, a_z);
  get_force_kernel->launch(&host_ctx);
  // get_b(v, b, f)
  set_ctx_arg_devalloc(host_ctx, 0, dalloc_v);
  set_ctx_arg_devalloc(host_ctx, 1, dalloc_b);
  set_ctx_arg_devalloc(host_ctx, 2, dalloc_f);
  get_b_kernel->launch(&host_ctx);

  // matmul_cell(v, mul_ans)
  set_ctx_arg_devalloc(host_ctx, 0, dalloc_v);
  set_ctx_arg_devalloc(host_ctx, 1, dalloc_mul_ans);
  matmul_cell_kernel->launch(&host_ctx);

  // add(r0, b, -1, mul_ans)
  set_ctx_arg_devalloc(host_ctx, 0, dalloc_r0);
  set_ctx_arg_devalloc(host_ctx, 1, dalloc_b);
  set_ctx_arg_float(host_ctx, 2, -1);
  set_ctx_arg_devalloc(host_ctx, 3, dalloc_mul_ans);
  add_kernel->launch(&host_ctx);

  // ndarray_to_ndarray(p0, r0)
  set_ctx_arg_devalloc(host_ctx, 0, dalloc_p0);
  set_ctx_arg_devalloc(host_ctx, 1, dalloc_r0);
  ndarray_to_ndarray_kernel->launch(&host_ctx);

  // dot2scalar(r0, r0)
  set_ctx_arg_devalloc(host_ctx, 0, dalloc_r0);
  set_ctx_arg_devalloc(host_ctx, 1, dalloc_r0);
  dot2scalar_kernel->launch(&host_ctx);

  // init_r_2()
  init_r_2_kernel->launch(&host_ctx);

  int n_iter = 8;

  for (int i = 0; i < n_iter; ++i) {
    // matmul_cell(p0, mul_ans)
    set_ctx_arg_devalloc(host_ctx, 0, dalloc_p0);
    set_ctx_arg_devalloc(host_ctx, 1, dalloc_mul_ans);
    matmul_cell_kernel->launch(&host_ctx);

    set_ctx_arg_devalloc(host_ctx, 0, dalloc_p0);
    set_ctx_arg_devalloc(host_ctx, 1, dalloc_mul_ans);
    dot2scalar_kernel->launch(&host_ctx);

    set_ctx_arg_devalloc(host_ctx, 0, dalloc_alpha, 1, 1);
    update_alpha_kernel->launch(&host_ctx);

    set_ctx_arg_devalloc(host_ctx, 0, dalloc_v);
    set_ctx_arg_devalloc(host_ctx, 1, dalloc_v);
    set_ctx_arg_float(host_ctx, 2, 1);
    set_ctx_arg_devalloc(host_ctx, 3, dalloc_alpha, 1, 1);
    set_ctx_arg_devalloc(host_ctx, 4, dalloc_p0);
    add_hack_kernel->launch(&host_ctx);

    set_ctx_arg_devalloc(host_ctx, 0, dalloc_r0);
    set_ctx_arg_devalloc(host_ctx, 1, dalloc_r0);
    set_ctx_arg_float(host_ctx, 2, -1);
    set_ctx_arg_devalloc(host_ctx, 3, dalloc_alpha, 1, 1);
    set_ctx_arg_devalloc(host_ctx, 4, dalloc_mul_ans);
    add_hack_kernel->launch(&host_ctx);

    set_ctx_arg_devalloc(host_ctx, 0, dalloc_r0);
    set_ctx_arg_devalloc(host_ctx, 1, dalloc_r0);
    dot2scalar_kernel->launch(&host_ctx);


    set_ctx_arg_devalloc(host_ctx, 0, dalloc_beta, 1, 1);
    update_beta_r_2_kernel->launch(&host_ctx);

    set_ctx_arg_devalloc(host_ctx, 0, dalloc_p0);
    set_ctx_arg_devalloc(host_ctx, 1, dalloc_r0);
    set_ctx_arg_float(host_ctx, 2, 1);
    set_ctx_arg_devalloc(host_ctx, 3, dalloc_beta, 1, 1);
    set_ctx_arg_devalloc(host_ctx, 4, dalloc_p0);
    add_hack_kernel->launch(&host_ctx);

  }
  // fill_ndarray(f, 0)
  set_ctx_arg_devalloc(host_ctx, 0, dalloc_f);
  set_ctx_arg_float(host_ctx, 1, 0);
  fill_ndarray_kernel->launch(&host_ctx);

  // add(x, x, dt, v)
  set_ctx_arg_devalloc(host_ctx, 0, dalloc_circles);
  set_ctx_arg_devalloc(host_ctx, 1, dalloc_circles);
  set_ctx_arg_float(host_ctx, 2, dt);
  set_ctx_arg_devalloc(host_ctx, 3, dalloc_v);
  add_kernel->launch(&host_ctx);
#endif
  set_ctx_arg_devalloc(host_ctx, 0, dalloc_circles);
  set_ctx_arg_devalloc(host_ctx, 1, dalloc_v);
  floor_bound_kernel->launch(&host_ctx);
  //ALOGI("launch kernel floor_bound");
  // Make sure to sync the GPU memory so we can read the latest update from CPU
  // And read the 'x' calculated on GPU to our local variable
  // @TODO: Skip this with support of NdArray as we will be able to specify 'dalloc_circles'
  // in the host_ctx before running the kernel, allowing the data to be updated automatically
  vulkan_runtime->synchronize();

  // timer ends after synchronization
  auto end = std::chrono::steady_clock::now();
  auto cpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  //ALOGI("Execution time is %" PRId64 "ns\n", cpu_time);

  // Render the UI
  scene->set_camera(camera);
  scene->particles(p_info);
  scene->ambient_light({1.0, 1.0, 1.0});
  scene->point_light({0, 0, 0}, {0, 0, 0});
  renderer->scene(static_cast<taichi::ui::vulkan::Scene*>(scene.get()));
  renderer->draw_frame(gui);
  renderer->swap_chain().surface().present_image();
  renderer->prepare_for_next_frame();
}
