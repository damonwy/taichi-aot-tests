#include "mesh_data.h"
#include <signal.h>
#include <iostream>

#include <taichi/backends/vulkan/vulkan_program.h>
#include <taichi/backends/vulkan/vulkan_common.h>
#include <taichi/backends/vulkan/vulkan_loader.h>
#include <taichi/backends/vulkan/aot_module_loader_impl.h>
#include <taichi/inc/constants.h>
#include <inttypes.h>

#include <taichi/gui/gui.h>
#include <taichi/ui/backends/vulkan/renderer.h>

// Uncomment this line to show explicit fem deomo
//#define USE_EXPLICIT

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
    float *device_arr_ptr = reinterpret_cast<float *>(vulkan_runtime.get_ti_device()->map(alloc));
    return device_arr_ptr;
}

void unmap(taichi::lang::vulkan::VkRuntime &vulkan_runtime, taichi::lang::DeviceAllocation &alloc) {
    vulkan_runtime.get_ti_device()->unmap(alloc);
}

void print_debug(taichi::lang::vulkan::VkRuntime &vulkan_runtime, taichi::lang::DeviceAllocation &alloc, int it, bool use_int = false) {
    vulkan_runtime.synchronize();
    auto ptr = map(vulkan_runtime, alloc);
    if (!use_int) printf("%d %.10f %.10f %.10f\n", it, ptr[0], ptr[1], ptr[2]);
    else
    {
        auto p = reinterpret_cast<int*>(ptr);
        printf("%d %d %d %d\n", it, p[0], p[1], p[2]);
    }
    unmap(vulkan_runtime, alloc);
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

    float dt = 2.5e-3;

    // Create a GGUI configuration
    taichi::ui::AppConfig app_config;
    app_config.name         = "FEM";
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
#ifdef USE_EXPLICIT
    std::string shader_source = "../explicit_fem";
#else
    std::string shader_source = "../implicit_mesh_fem";
#endif
    taichi::lang::vulkan::AotModuleParams aot_params{shader_source, &vulkan_runtime};
    auto module = taichi::lang::aot::Module::load(taichi::Arch::vulkan, aot_params);
    auto root_size = module->get_root_size();
    printf("root buffer size=%ld\n", root_size);
    vulkan_runtime.add_root_buffer(root_size);

    auto get_force_kernel = module->get_kernel("get_force");
    auto init_kernel = module->get_kernel("init");
    auto floor_bound_kernel = module->get_kernel("floor_bound");
    auto get_matrix_kernel = module->get_kernel("get_matrix");
    auto matmul_edge_kernel = module->get_kernel("matmul_edge");
    auto add_kernel = module->get_kernel("add");
    auto dot_kernel = module->get_kernel("dot");
    auto get_b_kernel = module->get_kernel("get_b");
    auto ndarray_to_ndarray_kernel = module->get_kernel("ndarray_to_ndarray");
    auto fill_ndarray_kernel = module->get_kernel("fill_ndarray");
    auto clear_field_kernel = module->get_kernel("clear_field");
    
    /*
    auto init_kernel = module->get_kernel("init");
    auto get_vertices_kernel = module->get_kernel("get_vertices");
    auto get_indices_kernel = module->get_kernel("get_indices");
    auto get_force_kernel = module->get_kernel("get_force");
    auto advect_kernel = module->get_kernel("advect");
    auto floor_bound_kernel = module->get_kernel("floor_bound");
    auto get_b_kernel = module->get_kernel("get_b");
    auto matmul_cell_kernel = module->get_kernel("matmul_cell");
    auto add_kernel = module->get_kernel("add");
    auto ndarray_to_ndarray_kernel = module->get_kernel("ndarray_to_ndarray");
    auto dot_kernel = module->get_kernel("dot");
    auto fill_ndarray_kernel = module->get_kernel("fill_ndarray");
    */

    // Prepare Ndarray for model
    taichi::lang::Device::AllocParams alloc_params;
    alloc_params.host_write = true;
    // x
    alloc_params.size = N_VERTS * 3 * sizeof(float);
    taichi::lang::DeviceAllocation devalloc_x = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params);
    // v
    taichi::lang::DeviceAllocation devalloc_v = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params);
    // f
    taichi::lang::DeviceAllocation devalloc_f = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params);
    // mul_ans
    taichi::lang::DeviceAllocation devalloc_mul_ans = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params);
    // c2e
    alloc_params.size = N_CELLS * 6 * sizeof(int);
    taichi::lang::DeviceAllocation devalloc_c2e = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params);
    // b
    alloc_params.size = N_VERTS * 3 * sizeof(float);
    taichi::lang::DeviceAllocation devalloc_b = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params);
    // r0
    taichi::lang::DeviceAllocation devalloc_r0 = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params);
    // p0
    taichi::lang::DeviceAllocation devalloc_p0 = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params);
    // indices
    alloc_params.size = N_FACES * 3 * sizeof(int);
    taichi::lang::DeviceAllocation devalloc_indices = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params);
    // vertices
    alloc_params.size = N_CELLS * 4 * sizeof(int);
    taichi::lang::DeviceAllocation devalloc_vertices = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params);
    // edges
    alloc_params.size = N_EDGES * 2 * sizeof(int);
    taichi::lang::DeviceAllocation devalloc_edges = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params);
    // ox
    alloc_params.size = N_VERTS * 3 * sizeof(float);
    taichi::lang::DeviceAllocation devalloc_ox = vulkan_runtime.get_ti_device()->allocate_memory(alloc_params);

    {
        char *const device_arr_ptr = reinterpret_cast<char *>(vulkan_runtime.get_ti_device()->map(devalloc_indices));
        std::memcpy(device_arr_ptr, (void *)indices_data, sizeof(indices_data));
        vulkan_runtime.get_ti_device()->unmap(devalloc_indices);
    }
    {
        char *const device_arr_ptr = reinterpret_cast<char *>(vulkan_runtime.get_ti_device()->map(devalloc_c2e));
        std::memcpy(device_arr_ptr, (void *)c2e_data, sizeof(c2e_data));
        vulkan_runtime.get_ti_device()->unmap(devalloc_c2e);
    }
    {
        char *const device_arr_ptr = reinterpret_cast<char *>(vulkan_runtime.get_ti_device()->map(devalloc_vertices));
        std::memcpy(device_arr_ptr, (void *)vertices_data, sizeof(vertices_data));
        vulkan_runtime.get_ti_device()->unmap(devalloc_vertices);
    }
    {
        char *const device_arr_ptr = reinterpret_cast<char *>(vulkan_runtime.get_ti_device()->map(devalloc_ox));
        std::memcpy(device_arr_ptr, (void *)ox_data, sizeof(ox_data));
        vulkan_runtime.get_ti_device()->unmap(devalloc_ox);
    }
    {
        char *const device_arr_ptr = reinterpret_cast<char *>(vulkan_runtime.get_ti_device()->map(devalloc_edges));
        std::memcpy(device_arr_ptr, (void *)edges_data, sizeof(edges_data));
        vulkan_runtime.get_ti_device()->unmap(devalloc_edges);
    }

    taichi::lang::RuntimeContext host_ctx;
    memset(&host_ctx, 0, sizeof(taichi::lang::RuntimeContext));
    host_ctx.result_buffer = result_buffer;

    // Create a GUI even though it's not used in our case (required to
    // render the renderer)
    auto gui = std::make_shared<taichi::ui::vulkan::Gui>(&renderer->app_context(), &renderer->swap_chain(), window);
    clear_field_kernel->launch(&host_ctx);
    
    set_ctx_arg_devalloc(host_ctx, 0, devalloc_x, N_VERTS, 3, 1);
    set_ctx_arg_devalloc(host_ctx, 1, devalloc_v, N_VERTS, 3, 1);
    set_ctx_arg_devalloc(host_ctx, 2, devalloc_f, N_VERTS, 3, 1);
    set_ctx_arg_devalloc(host_ctx, 3, devalloc_ox, N_VERTS, 3, 1);
    set_ctx_arg_devalloc(host_ctx, 4, devalloc_vertices, N_CELLS, 4, 1);
    // init(x, v, f, ox, vertices)
    init_kernel->launch(&host_ctx);
    // get_matrix(c2e, vertices)
    set_ctx_arg_devalloc(host_ctx, 0, devalloc_c2e, N_CELLS, 6, 1);
    set_ctx_arg_devalloc(host_ctx, 1, devalloc_vertices, N_CELLS, 4, 1);
    get_matrix_kernel->launch(&host_ctx);
    vulkan_runtime.synchronize();

    // Describe information to render the circle with Vulkan
    taichi::ui::FieldInfo f_info;
    f_info.valid        = true;
    f_info.field_type   = taichi::ui::FieldType::Matrix;
    f_info.matrix_rows  = 3;
    f_info.matrix_cols  = 1;
    f_info.shape        = {N_VERTS};
    f_info.field_source = taichi::ui::FieldSource::TaichiVulkan;
    f_info.dtype        = taichi::lang::PrimitiveType::f32;
    f_info.snode        = nullptr;
    f_info.dev_alloc    = devalloc_x;
    taichi::ui::FieldInfo i_info;
    i_info.valid = true;
    i_info.field_type = taichi::ui::FieldType::Matrix;
    i_info.matrix_rows = 3;
    i_info.matrix_cols = 1;
    i_info.shape = {N_FACES};
    i_info.field_source = taichi::ui::FieldSource::TaichiVulkan;
    i_info.dtype = taichi::lang::PrimitiveType::i32;
    i_info.snode = nullptr;
    i_info.dev_alloc = devalloc_indices;
    taichi::ui::RenderableInfo r_info;
    r_info.vbo = f_info;
    r_info.has_per_vertex_color = false;
    r_info.indices = i_info;
    r_info.vbo_attrs = taichi::ui::VertexAttributes::kPos;
    taichi::ui::MeshInfo m_info;
    m_info.renderable_info = r_info;
    m_info.color = glm::vec3(0.73, 0.33, 0.23);
    m_info.two_sided = false;
    taichi::ui::Camera camera;
    camera.position = glm::vec3(0.0, 1.5, 2.95);
    camera.lookat = glm::vec3(0, 0, 0);
    camera.up = glm::vec3(0, 1, 0);
    camera.fov = 55.0f;
    auto scene = std::make_unique<taichi::ui::SceneBase>();

    renderer->set_background_color({0.0, 0.0, 0.0});

    while (!glfwWindowShouldClose(window)) {
        // Run 'substep' 40 times
        // Explicit FEM
#ifdef USE_EXPLICIT
        for (int i = 0; i < 40; i++) {
            // get_force(x, f)
            set_ctx_arg_devalloc(host_ctx, 0, devalloc_x);
            set_ctx_arg_devalloc(host_ctx, 1, devalloc_f);
            get_force_kernel->launch(&host_ctx);
            // advect(x, v, f)
            set_ctx_arg_devalloc(host_ctx, 0, devalloc_x);
            set_ctx_arg_devalloc(host_ctx, 1, devalloc_v);
            set_ctx_arg_devalloc(host_ctx, 2, devalloc_f);
            advect_kernel->launch(&host_ctx);
        }
#else
        for (int i = 0; i < 4; i++) {
            // get_force(x, f, vertices)
            set_ctx_arg_devalloc(host_ctx, 0, devalloc_x, N_VERTS, 3, 1);
            set_ctx_arg_devalloc(host_ctx, 1, devalloc_f, N_VERTS, 3, 1);
            set_ctx_arg_devalloc(host_ctx, 2, devalloc_vertices, N_CELLS, 4, 1);
            get_force_kernel->launch(&host_ctx);
            // get_b(v, b, f)
            set_ctx_arg_devalloc(host_ctx, 0, devalloc_v, N_VERTS, 3, 1);
            set_ctx_arg_devalloc(host_ctx, 1, devalloc_b, N_VERTS, 3, 1);
            set_ctx_arg_devalloc(host_ctx, 2, devalloc_f, N_VERTS, 3, 1);
            get_b_kernel->launch(&host_ctx);
            
            // matmul_edge(mul_ans, v, edges)
            // matmul_edge(mul_ans, v, edges)
            set_ctx_arg_devalloc(host_ctx, 0, devalloc_mul_ans, N_VERTS, 3, 1);
            set_ctx_arg_devalloc(host_ctx, 1, devalloc_v, N_VERTS, 3, 1);
            set_ctx_arg_devalloc(host_ctx, 2, devalloc_edges, N_EDGES, 2, 1);
            matmul_edge_kernel->launch(&host_ctx);
            matmul_edge_kernel->launch(&host_ctx);
            // add(r0, b, -1, mul_ans)
            set_ctx_arg_devalloc(host_ctx, 0, devalloc_r0, N_VERTS, 3, 1);
            set_ctx_arg_devalloc(host_ctx, 1, devalloc_b, N_VERTS, 3, 1);
            set_ctx_arg_float(host_ctx, 2, -1.0f);
            set_ctx_arg_devalloc(host_ctx, 3, devalloc_mul_ans, N_VERTS, 3, 1);
            add_kernel->launch(&host_ctx);
            // ndarray_to_ndarray(p0, r0)
            set_ctx_arg_devalloc(host_ctx, 0, devalloc_p0, N_VERTS, 3, 1);
            set_ctx_arg_devalloc(host_ctx, 1, devalloc_r0, N_VERTS, 3, 1);
            ndarray_to_ndarray_kernel->launch(&host_ctx);
            // r_2 = dot(r0, r0)
            set_ctx_arg_devalloc(host_ctx, 0, devalloc_r0, N_VERTS, 3, 1);
            set_ctx_arg_devalloc(host_ctx, 1, devalloc_r0, N_VERTS, 3, 1);
            dot_kernel->launch(&host_ctx);
            float r_2 = host_ctx.get_ret<float>(0);

            int n_iter = 10;
            float epsilon = 1e-6;
            float r_2_init = r_2;
            float r_2_new = r_2;

            for (int i = 0; i < n_iter; i++) {
                // matmul_edge(mul_ans, p0, edges);
                set_ctx_arg_devalloc(host_ctx, 0, devalloc_mul_ans, N_VERTS, 3, 1);
                set_ctx_arg_devalloc(host_ctx, 1, devalloc_p0, N_VERTS, 3, 1);
                set_ctx_arg_devalloc(host_ctx, 2, devalloc_edges, N_EDGES, 2, 1);
                matmul_edge_kernel->launch(&host_ctx);
                // alpha = r_2_new / dot(p0, mul_ans)
                set_ctx_arg_devalloc(host_ctx, 0, devalloc_p0, N_VERTS, 3, 1);
                set_ctx_arg_devalloc(host_ctx, 1, devalloc_mul_ans, N_VERTS, 3, 1);
                dot_kernel->launch(&host_ctx);
                vulkan_runtime.synchronize();
                float alpha = r_2_new / host_ctx.get_ret<float>(0);
          		// add(v, v, alpha, p0)
          		set_ctx_arg_devalloc(host_ctx, 0, devalloc_v, N_VERTS, 3, 1);
          		set_ctx_arg_devalloc(host_ctx, 1, devalloc_v, N_VERTS, 3, 1);
          		set_ctx_arg_float(host_ctx, 2, alpha);
          		set_ctx_arg_devalloc(host_ctx, 3, devalloc_p0, N_VERTS, 3, 1);
          		add_kernel->launch(&host_ctx);
			    // add(r0, r0, -alpha, mul_ans)
                set_ctx_arg_devalloc(host_ctx, 0, devalloc_r0, N_VERTS, 3, 1);
                set_ctx_arg_devalloc(host_ctx, 1, devalloc_r0, N_VERTS, 3, 1);
                set_ctx_arg_float(host_ctx, 2, -alpha);
                set_ctx_arg_devalloc(host_ctx, 3, devalloc_mul_ans, N_VERTS, 3, 1);
                add_kernel->launch(&host_ctx);

                r_2 = r_2_new;
                // r_2_new = dot(r0, r0)
                set_ctx_arg_devalloc(host_ctx, 0, devalloc_r0, N_VERTS, 3, 1);
                set_ctx_arg_devalloc(host_ctx, 1, devalloc_r0, N_VERTS, 3, 1);
                dot_kernel->launch(&host_ctx);
                vulkan_runtime.synchronize();
                r_2_new = host_ctx.get_ret<float>(0);

                if (r_2_new <= r_2_init * epsilon * epsilon) {break;}
                float beta = r_2_new / r_2;

                // add(p0, r0, beta, p0)
                set_ctx_arg_devalloc(host_ctx, 0, devalloc_p0, N_VERTS, 3, 1);
                set_ctx_arg_devalloc(host_ctx, 1, devalloc_r0, N_VERTS, 3, 1);
                set_ctx_arg_float(host_ctx, 2, beta);
                set_ctx_arg_devalloc(host_ctx, 3, devalloc_p0, N_VERTS, 3, 1);
                add_kernel->launch(&host_ctx);
            }
            
            // fill_ndarray(f, 0)
            set_ctx_arg_devalloc(host_ctx, 0, devalloc_f, N_VERTS, 3, 1);
            set_ctx_arg_float(host_ctx, 1, 0);
            fill_ndarray_kernel->launch(&host_ctx);

            // add(x, x, dt, v)
            set_ctx_arg_devalloc(host_ctx, 0, devalloc_x, N_VERTS, 3, 1);
            set_ctx_arg_devalloc(host_ctx, 1, devalloc_x, N_VERTS, 3, 1);
            set_ctx_arg_float(host_ctx, 2, dt);
            set_ctx_arg_devalloc(host_ctx, 3, devalloc_v, N_VERTS, 3, 1);
            add_kernel->launch(&host_ctx);
        }
#endif
        // floor_bound(x, v)
        set_ctx_arg_devalloc(host_ctx, 0, devalloc_x, N_VERTS, 3, 1);
        set_ctx_arg_devalloc(host_ctx, 1, devalloc_v, N_VERTS, 3, 1);
        floor_bound_kernel->launch(&host_ctx);
        vulkan_runtime.synchronize();
#ifdef ONLY_INIT
        set_ctx_arg_devalloc(host_ctx, 0, devalloc_x, N_VERTS, 3, 1);
        set_ctx_arg_devalloc(host_ctx, 1, devalloc_v, N_VERTS, 3, 1);
        set_ctx_arg_devalloc(host_ctx, 2, devalloc_f, N_VERTS, 3, 1);
        set_ctx_arg_devalloc(host_ctx, 3, devalloc_ox, N_VERTS, 3, 1);
        set_ctx_arg_devalloc(host_ctx, 4, devalloc_vertices, N_CELLS, 4, 1);
        init_kernel->launch(&host_ctx);
        print_debug(vulkan_runtime, devalloc_x, 0);
#endif
        // Render elements
        scene->set_camera(camera);
        scene->mesh(m_info);
        scene->ambient_light(glm::vec3(0.1f, 0.1f, 0.1f));
        scene->point_light(glm::vec3(0.5f, 10.0f, 0.5f), glm::vec3(0.5f, 0.5f, 0.5f));
        scene->point_light(glm::vec3(10.0f, 10.0f, 10.0f), glm::vec3(0.5f, 0.5f, 0.5f));
        renderer->scene(static_cast<taichi::ui::vulkan::Scene*>(scene.get()));
        renderer->draw_frame(gui.get());
        renderer->swap_chain().surface().present_image();
        renderer->prepare_for_next_frame();
        vulkan_runtime.synchronize();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    return 0;
}
