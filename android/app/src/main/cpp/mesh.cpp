#include <android/asset_manager.h>
#include "mesh_data.h"
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

#define ALOGI(fmt, ...)                                                  \
  ((void)__android_log_print(ANDROID_LOG_INFO, "TaichiTest", "%s: " fmt, \
                             __FUNCTION__, ##__VA_ARGS__))
#define ALOGE(fmt, ...)                                                   \
  ((void)__android_log_print(ANDROID_LOG_ERROR, "TaichiTest", "%s: " fmt, \
                             __FUNCTION__, ##__VA_ARGS__))

#define ONLY_INIT

void set_ctx_arg_devalloc(taichi::lang::RuntimeContext &host_ctx, int arg_id, taichi::lang::DeviceAllocation& alloc, int x, int y, int z) {
  host_ctx.set_arg(arg_id, &alloc);
  host_ctx.set_device_allocation(arg_id, true);
  // This is hack since our ndarrays happen to have exactly the same size in implicit_fem demo.
  host_ctx.extra_args[arg_id][0] = x;
  host_ctx.extra_args[arg_id][1] = y;
  host_ctx.extra_args[arg_id][2] = z;
}

// TODO: provide a proper API from taichi
void set_ctx_arg_float(taichi::lang::RuntimeContext &host_ctx, int arg_id, float x) {
  host_ctx.set_arg(arg_id, x);
  host_ctx.set_device_allocation(arg_id, false);
}

float *map(taichi::lang::vulkan::VkRuntime &vulkan_runtime, taichi::lang::DeviceAllocation &alloc) {
  float *device_arr_ptr = reinterpret_cast<float *> (vulkan_runtime.get_ti_device()->map(alloc));
  return device_arr_ptr;
}

void unmap(taichi::lang::vulkan::VkRuntime &vulkan_runtime, taichi::lang::DeviceAllocation &alloc) {
  vulkan_runtime.get_ti_device()->unmap(alloc);
}

void print_debug(taichi::lang::vulkan::VkRuntime *vulkan_runtime, taichi::lang::DeviceAllocation &alloc, int it, bool use_int = false) {
    vulkan_runtime->synchronize();
    auto ptr = map(*vulkan_runtime, alloc);
    if (!use_int) ALOGI("%d %.10f %.10f %.10f\n", it, ptr[0], ptr[1], ptr[2]);
    else
    {
        auto p = reinterpret_cast<int*>(ptr);
        ALOGI("%d %d %d %d\n", it, p[0], p[1], p[2]);
    }
    unmap(*vulkan_runtime, alloc);
}

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
taichi::lang::aot::Kernel* get_matrix_kernel;
taichi::lang::aot::Kernel* clear_field_kernel;
taichi::lang::aot::Kernel* matmul_edge_kernel;

std::unique_ptr<taichi::lang::aot::Module> module;
taichi::ui::vulkan::Renderer *renderer;
taichi::ui::vulkan::Gui *gui;
taichi::lang::DeviceAllocation dalloc_x;
taichi::lang::DeviceAllocation dalloc_v;
taichi::lang::DeviceAllocation dalloc_f;
taichi::lang::DeviceAllocation dalloc_b;
taichi::lang::DeviceAllocation dalloc_mul_ans;
taichi::lang::DeviceAllocation dalloc_r0;
taichi::lang::DeviceAllocation dalloc_p0;
taichi::lang::DeviceAllocation dalloc_c2e;
taichi::lang::DeviceAllocation dalloc_indices;
taichi::lang::DeviceAllocation dalloc_vertices;
taichi::lang::DeviceAllocation dalloc_edges;
taichi::lang::DeviceAllocation dalloc_alpha;
taichi::lang::DeviceAllocation dalloc_beta;
taichi::lang::DeviceAllocation dalloc_ox;

taichi::lang::RuntimeContext host_ctx;

taichi::ui::FieldInfo f_info;
taichi::ui::FieldInfo i_info;
taichi::ui::RenderableInfo r_info;
taichi::ui::MeshInfo m_info;
std::unique_ptr<taichi::ui::SceneBase> scene;
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

  std::string shader_source = "/data/local/tmp/implicit_mesh_fem";

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
  get_matrix_kernel = module->get_kernel("get_matrix");
  matmul_edge_kernel = module->get_kernel("matmul_edge");

  get_force_kernel = module->get_kernel("get_force");
  advect_kernel = module->get_kernel("advect");
  floor_bound_kernel = module->get_kernel("floor_bound");

  get_b_kernel = module->get_kernel("get_b");
  matmul_cell_kernel = module->get_kernel("matmul_cell");
  add_kernel = module->get_kernel("add");
  clear_field_kernel = module->get_kernel("clear_field");
  ndarray_to_ndarray_kernel = module->get_kernel("ndarray_to_ndarray");
  dot_kernel = module->get_kernel("dot");
  add_ndarray_kernel = module->get_kernel("add_ndarray");
  fill_ndarray_kernel = module->get_kernel("fill_ndarray");
  init_r_2_kernel = module->get_kernel("init_r_2");
  update_alpha_kernel = module->get_kernel("update_alpha");
  update_beta_r_2_kernel = module->get_kernel("update_beta_r_2");
  add_hack_kernel = module->get_kernel("add_hack");
  dot2scalar_kernel = module->get_kernel("dot2scalar");


  // Allocate memory for Circles position
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.host_write = true;
  // x
  alloc_params.size = N_VERTS * 3 * sizeof(float);
  dalloc_x = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
  dalloc_v = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
  dalloc_f = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
  dalloc_mul_ans = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);

  // c2e
  alloc_params.size = N_CELLS * 6 * sizeof(int);
  dalloc_c2e = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
  // b
  alloc_params.size = N_VERTS * 3 * sizeof(float);
  dalloc_b = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
  dalloc_r0 = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
  dalloc_p0 = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);

  // indices
  alloc_params.size = N_FACES * 3 * sizeof(int);
  dalloc_indices = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);

  // vertices
  alloc_params.size = N_CELLS * 4 * sizeof(int);
  dalloc_vertices = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);

  // edges
  alloc_params.size = N_EDGES * 2 * sizeof(int);
  dalloc_edges = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);

  // ox
  alloc_params.size = N_VERTS * 3 * sizeof(float);
  dalloc_ox = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);

  {
      char *const device_arr_ptr = reinterpret_cast<char *>(vulkan_runtime->get_ti_device()->map(dalloc_indices));
      std::memcpy(device_arr_ptr, (void *)indices_data, sizeof(indices_data));
      vulkan_runtime->get_ti_device()->unmap(dalloc_indices);
  }
  {
      char *const device_arr_ptr = reinterpret_cast<char *>(vulkan_runtime->get_ti_device()->map(dalloc_c2e));
      std::memcpy(device_arr_ptr, (void *)c2e_data, sizeof(c2e_data));
      vulkan_runtime->get_ti_device()->unmap(dalloc_c2e);
  }
  {
      char *const device_arr_ptr = reinterpret_cast<char *>(vulkan_runtime->get_ti_device()->map(dalloc_vertices));
      std::memcpy(device_arr_ptr, (void *)vertices_data, sizeof(vertices_data));
      vulkan_runtime->get_ti_device()->unmap(dalloc_vertices);
  }
  {
      char *const device_arr_ptr = reinterpret_cast<char *>(vulkan_runtime->get_ti_device()->map(dalloc_ox));
      std::memcpy(device_arr_ptr, (void *)ox_data, sizeof(ox_data));
      vulkan_runtime->get_ti_device()->unmap(dalloc_ox);
  }
  {
      char *const device_arr_ptr = reinterpret_cast<char *>(vulkan_runtime->get_ti_device()->map(dalloc_edges));
      std::memcpy(device_arr_ptr, (void *)edges_data, sizeof(edges_data));
      vulkan_runtime->get_ti_device()->unmap(dalloc_edges);
  }

  //alloc_params.size = sizeof(float);
  //dalloc_alpha = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
  //dalloc_beta = vulkan_runtime->get_ti_device()->allocate_memory(alloc_params);
  clear_field_kernel->launch(&host_ctx);

  // init(x, v, f, ox)
  set_ctx_arg_devalloc(host_ctx, 0, dalloc_x, N_VERTS, 3, 1);
  set_ctx_arg_devalloc(host_ctx, 1, dalloc_v, N_VERTS, 3, 1);
  set_ctx_arg_devalloc(host_ctx, 2, dalloc_f, N_VERTS, 3, 1);
  set_ctx_arg_devalloc(host_ctx, 3, dalloc_ox, N_VERTS, 3, 1);
  set_ctx_arg_devalloc(host_ctx, 4, dalloc_vertices, N_CELLS, 4, 1);
  init_kernel->launch(&host_ctx);

  // get_matrix(c2e, vertices)
  set_ctx_arg_devalloc(host_ctx, 0, dalloc_c2e, N_CELLS, 6, 1);
  set_ctx_arg_devalloc(host_ctx, 1, dalloc_vertices, N_CELLS, 4, 1);
  get_matrix_kernel->launch(&host_ctx);
  vulkan_runtime->synchronize();

  // Describe information to render the circle with Vulkan
  f_info.valid        = true;
  f_info.field_type   = taichi::ui::FieldType::Matrix;
  f_info.matrix_rows  = 3;
  f_info.matrix_cols  = 1;
  f_info.shape        = {N_VERTS};
  f_info.field_source = taichi::ui::FieldSource::TaichiVulkan;
  f_info.dtype        = taichi::lang::PrimitiveType::f32;
  f_info.snode        = nullptr;
  f_info.dev_alloc    = dalloc_x;

  i_info.valid        = true;
  i_info.field_type   = taichi::ui::FieldType::Matrix;
  i_info.matrix_rows  = 3;
  i_info.matrix_cols  = 1;
  i_info.shape        = {N_FACES};
  i_info.field_source = taichi::ui::FieldSource::TaichiVulkan;
  i_info.dtype        = taichi::lang::PrimitiveType::i32;
  i_info.snode        = nullptr;
  i_info.dev_alloc    = dalloc_indices;


  r_info.vbo = f_info;
  r_info.has_per_vertex_color = false;
  r_info.indices = i_info;
  r_info.vbo_attrs = taichi::ui::VertexAttributes::kPos;

  m_info.renderable_info = r_info;
  m_info.color = {0.73, 0.33, 0.23};
  m_info.two_sided = false;

  camera.position = {0, 1.5, 2.95};
  camera.lookat = {0, 0, 0};
  camera.up = {0, 1, 0};
  camera.fov = 55.0f;
  scene = std::make_unique<taichi::ui::SceneBase>();
  renderer->set_background_color({0., 0., 0.});


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
  ALOGI("Acceleration: g_x = %f, g_y = %f, g_z = %f", g_x, g_y, g_z);
  float a_x = g_z;
  float a_y = g_y > 2 || g_y < -2 ? -g_y : 0;
  float a_z = g_x > 2 || g_x < -2 ? g_x * 4 : 0;

  // timer starts before launch kernel
  auto start = std::chrono  ::steady_clock::now();

  float dt = 2.5e-3;
#if 0
  for (int i = 0; i < 4; i++) {
      // get_force(x, f, vertices)
      set_ctx_arg_devalloc(host_ctx, 0, dalloc_x, N_VERTS, 3, 1);
      set_ctx_arg_devalloc(host_ctx, 1, dalloc_f, N_VERTS, 3, 1);
      set_ctx_arg_devalloc(host_ctx, 2, dalloc_vertices, N_CELLS, 4, 1);
      get_force_kernel->launch(&host_ctx);
      // get_b(v, b, f)
      set_ctx_arg_devalloc(host_ctx, 0, dalloc_v, N_VERTS, 3, 1);
      set_ctx_arg_devalloc(host_ctx, 1, dalloc_b, N_VERTS, 3, 1);
      set_ctx_arg_devalloc(host_ctx, 2, dalloc_f, N_VERTS, 3, 1);
      get_b_kernel->launch(&host_ctx);

      // matmul_edge(mul_ans, v, edges)
      // matmul_edge(mul_ans, v, edges)
      set_ctx_arg_devalloc(host_ctx, 0, dalloc_mul_ans, N_VERTS, 3, 1);
      set_ctx_arg_devalloc(host_ctx, 1, dalloc_v, N_VERTS, 3, 1);
      set_ctx_arg_devalloc(host_ctx, 2, dalloc_edges, N_EDGES, 2, 1);
      matmul_edge_kernel->launch(&host_ctx);
      matmul_edge_kernel->launch(&host_ctx);
      // add(r0, b, -1, mul_ans)
      set_ctx_arg_devalloc(host_ctx, 0, dalloc_r0, N_VERTS, 3, 1);
      set_ctx_arg_devalloc(host_ctx, 1, dalloc_b, N_VERTS, 3, 1);
      set_ctx_arg_float(host_ctx, 2, -1.0f);
      set_ctx_arg_devalloc(host_ctx, 3, dalloc_mul_ans, N_VERTS, 3, 1);
      add_kernel->launch(&host_ctx);
      // ndarray_to_ndarray(p0, r0)
      set_ctx_arg_devalloc(host_ctx, 0, dalloc_p0, N_VERTS, 3, 1);
      set_ctx_arg_devalloc(host_ctx, 1, dalloc_r0, N_VERTS, 3, 1);
      ndarray_to_ndarray_kernel->launch(&host_ctx);
      // r_2 = dot(r0, r0)
      set_ctx_arg_devalloc(host_ctx, 0, dalloc_r0, N_VERTS, 3, 1);
      set_ctx_arg_devalloc(host_ctx, 1, dalloc_r0, N_VERTS, 3, 1);
      dot_kernel->launch(&host_ctx);
      float r_2 = host_ctx.get_ret<float>(0);

      int n_iter = 10;
      float epsilon = 1e-6;
      float r_2_init = r_2;
      float r_2_new = r_2;

      for (int i = 0; i < n_iter; i++) {
          // matmul_edge(mul_ans, p0, edges);
          set_ctx_arg_devalloc(host_ctx, 0, dalloc_mul_ans, N_VERTS, 3, 1);
          set_ctx_arg_devalloc(host_ctx, 1, dalloc_p0, N_VERTS, 3, 1);
          set_ctx_arg_devalloc(host_ctx, 2, dalloc_edges, N_EDGES, 2, 1);
          matmul_edge_kernel->launch(&host_ctx);
          // alpha = r_2_new / dot(p0, mul_ans)
          set_ctx_arg_devalloc(host_ctx, 0, dalloc_p0, N_VERTS, 3, 1);
          set_ctx_arg_devalloc(host_ctx, 1, dalloc_mul_ans, N_VERTS, 3, 1);
          dot_kernel->launch(&host_ctx);
          vulkan_runtime->synchronize();
          float alpha = r_2_new / host_ctx.get_ret<float>(0);
    		// add(v, v, alpha, p0)
    		set_ctx_arg_devalloc(host_ctx, 0, dalloc_v, N_VERTS, 3, 1);
    		set_ctx_arg_devalloc(host_ctx, 1, dalloc_v, N_VERTS, 3, 1);
    		set_ctx_arg_float(host_ctx, 2, alpha);
    		set_ctx_arg_devalloc(host_ctx, 3, dalloc_p0, N_VERTS, 3, 1);
    		add_kernel->launch(&host_ctx);
  	    // add(r0, r0, -alpha, mul_ans)
          set_ctx_arg_devalloc(host_ctx, 0, dalloc_r0, N_VERTS, 3, 1);
          set_ctx_arg_devalloc(host_ctx, 1, dalloc_r0, N_VERTS, 3, 1);
          set_ctx_arg_float(host_ctx, 2, -alpha);
          set_ctx_arg_devalloc(host_ctx, 3, dalloc_mul_ans, N_VERTS, 3, 1);
          add_kernel->launch(&host_ctx);

          r_2 = r_2_new;
          // r_2_new = dot(r0, r0)
          set_ctx_arg_devalloc(host_ctx, 0, dalloc_r0, N_VERTS, 3, 1);
          set_ctx_arg_devalloc(host_ctx, 1, dalloc_r0, N_VERTS, 3, 1);
          dot_kernel->launch(&host_ctx);
          vulkan_runtime->synchronize();
          r_2_new = host_ctx.get_ret<float>(0);

          if (r_2_new <= r_2_init * epsilon * epsilon) {break;}
          float beta = r_2_new / r_2;

          // add(p0, r0, beta, p0)
          set_ctx_arg_devalloc(host_ctx, 0, dalloc_p0, N_VERTS, 3, 1);
          set_ctx_arg_devalloc(host_ctx, 1, dalloc_r0, N_VERTS, 3, 1);
          set_ctx_arg_float(host_ctx, 2, beta);
          set_ctx_arg_devalloc(host_ctx, 3, dalloc_p0, N_VERTS, 3, 1);
          add_kernel->launch(&host_ctx);
      }

      // fill_ndarray(f, 0)
      set_ctx_arg_devalloc(host_ctx, 0, dalloc_f, N_VERTS, 3, 1);
      set_ctx_arg_float(host_ctx, 1, 0);
      fill_ndarray_kernel->launch(&host_ctx);

      // add(x, x, dt, v)
      set_ctx_arg_devalloc(host_ctx, 0, dalloc_x, N_VERTS, 3, 1);
      set_ctx_arg_devalloc(host_ctx, 1, dalloc_x, N_VERTS, 3, 1);
      set_ctx_arg_float(host_ctx, 2, dt);
      set_ctx_arg_devalloc(host_ctx, 3, dalloc_v, N_VERTS, 3, 1);
      add_kernel->launch(&host_ctx);
  }
  // floor_bound(x, v)
  set_ctx_arg_devalloc(host_ctx, 0, dalloc_x, N_VERTS, 3, 1);
  set_ctx_arg_devalloc(host_ctx, 1, dalloc_v, N_VERTS, 3, 1);
  floor_bound_kernel->launch(&host_ctx);
  vulkan_runtime->synchronize();
#endif
#ifdef ONLY_INIT
  set_ctx_arg_devalloc(host_ctx, 0, dalloc_x, N_VERTS, 3, 1);
  set_ctx_arg_devalloc(host_ctx, 1, dalloc_v, N_VERTS, 3, 1);
  set_ctx_arg_devalloc(host_ctx, 2, dalloc_f, N_VERTS, 3, 1);
  set_ctx_arg_devalloc(host_ctx, 3, dalloc_ox, N_VERTS, 3, 1);
  set_ctx_arg_devalloc(host_ctx, 4, dalloc_vertices, N_CELLS, 4, 1);
  init_kernel->launch(&host_ctx);
  vulkan_runtime->synchronize();

  print_debug(vulkan_runtime.get(), dalloc_x, 0);
  print_debug(vulkan_runtime.get(), dalloc_indices, 1, true);
#endif

  // timer ends after synchronization
  auto end = std::chrono::steady_clock::now();
  auto cpu_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  ALOGI("Execution time is %" PRId64 "ns\n", cpu_time);

  // Render elements
  scene->set_camera(camera);
  scene->mesh(m_info);
  scene->ambient_light({0.1f, 0.1f, 0.1f});
  scene->point_light({.5f, 10.0f, 0.5f}, {0.5f, 0.5f, 0.5f});
  scene->point_light({10.0f, 10.0f, 10.0f}, {0.5f, 0.5f, 0.5f});
  renderer->scene(static_cast<taichi::ui::vulkan::Scene*>(scene.get()));
  renderer->draw_frame(gui);
  renderer->swap_chain().surface().present_image();
  renderer->prepare_for_next_frame();
  vulkan_runtime->synchronize();
}
