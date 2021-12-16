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


#include <assert.h>
#include <map>
#include <stdint.h>
#include <vector>

#define ALOGI(fmt, ...)                                                        \
    ((void)__android_log_print(ANDROID_LOG_INFO, "TaichiTest", "%s: " fmt,     \
                               __FUNCTION__, ##__VA_ARGS__))
#define ALOGE(fmt, ...)                                                        \
    ((void)__android_log_print(ANDROID_LOG_ERROR, "TaichiTest", "%s: " fmt,    \
                               __FUNCTION__, ##__VA_ARGS__))

std::vector<std::string> get_required_instance_extensions() {
    std::vector<std::string> extensions;

    extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
    extensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
    extensions.push_back(
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    return extensions;
}

std::vector<std::string> get_required_device_extensions() {
    static std::vector<std::string> extensions{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };

    return extensions;
}

extern "C" JNIEXPORT void JNICALL
Java_com_innopeaktech_naboo_taichi_1test_NativeLib_init(JNIEnv *env, jclass,
                                                        jobject assets,
                                                        jobject surface) {
    ANativeWindow *native_window = ANativeWindow_fromSurface(env, surface);
    taichi::lang::vulkan::AotModuleLoaderImpl aot_loader("/data/local/tmp/mpm88");

    // Initialize our Vulkan Program pipeline
    taichi::uint64 *result_buffer{nullptr};
    taichi::lang::RuntimeContext host_ctx;

    auto memory_pool =
        std::make_unique<taichi::lang::MemoryPool>(taichi::lang::Arch::vulkan, nullptr);
    result_buffer = (taichi::uint64 *)memory_pool->allocate(
        sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);

    // Create Taichi Device for computation
    taichi::lang::vulkan::VulkanDeviceCreator::Params evd_params;
    evd_params.api_version =
        taichi::lang::vulkan::VulkanEnvSettings::kApiVersion();
    evd_params.additional_instance_extensions =
        get_required_instance_extensions();
    evd_params.additional_device_extensions = get_required_device_extensions();
    auto embedded_device =
        std::make_unique<taichi::lang::vulkan::VulkanDeviceCreator>(evd_params);

    taichi::lang::vulkan::VkRuntime::Params params;
    params.host_result_buffer = result_buffer;
    params.device = embedded_device->device();
    auto vulkan_runtime =
        std::make_unique<taichi::lang::vulkan::VkRuntime>(std::move(params));

    // Retrieve kernels/fields/etc from AOT module so we can initialize our
    // runtime
    taichi::lang::vulkan::VkRuntime::RegisterParams init_kernel, substep_kernel;
    bool ret = aot_loader.get_kernel("init", init_kernel);
    if (!ret) {
        ALOGE("Cannot find 'init' kernel\n");
        return;
    }
    ret = aot_loader.get_kernel("substep", substep_kernel);
    if (!ret) {
        ALOGE("Cannot find 'substep' kernel\n");
        return;
    }
    auto root_size = aot_loader.get_root_size();
    ALOGI("root buffer size=%d\n", root_size);

    vulkan_runtime->add_root_buffer(root_size);
    auto init_kernel_handle =
        vulkan_runtime->register_taichi_kernel(init_kernel);
    auto substep_kernel_handle =
        vulkan_runtime->register_taichi_kernel(substep_kernel);

    //
    // Run MPM88 from AOT module similar to Python code
    //
    vulkan_runtime->launch_kernel(init_kernel_handle, &host_ctx);
    vulkan_runtime->synchronize();

    // Sanity check to make sure the shaders are running properly, we should have the same float
    // values as the python scripts
    // aot->get_field("x");
    float x[10];
    vulkan_runtime->read_memory((uint8_t*) x, 0, 5 * 2 * sizeof(taichi::float32));

    for (int i = 0; i < 10; i += 2) {
        ALOGI("[%f, %f]\n", x[i], x[i+1]);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_innopeaktech_naboo_taichi_1test_NativeLib_destroy(JNIEnv *env, jclass,
                                                           jobject surface) {}

extern "C" JNIEXPORT void JNICALL
Java_com_innopeaktech_naboo_taichi_1test_NativeLib_pause(JNIEnv *env, jclass,
                                                         jobject surface) {}

extern "C" JNIEXPORT void JNICALL
Java_com_innopeaktech_naboo_taichi_1test_NativeLib_resume(JNIEnv *env, jclass,
                                                          jobject surface) {}

extern "C" JNIEXPORT void JNICALL
Java_com_innopeaktech_naboo_taichi_1test_NativeLib_resize(JNIEnv *, jclass,
                                                          jobject, jint width,
                                                          jint height) {
    ALOGI("Resize requested for %dx%d", width, height);
}

extern "C" JNIEXPORT void JNICALL
Java_com_innopeaktech_naboo_taichi_1test_NativeLib_render(JNIEnv *env, jclass,
                                                          jobject surface) {}
