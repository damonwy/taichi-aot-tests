#include <android/asset_manager.h>
#include "mesh_data.h"
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <android/looper.h>
#include <android/native_window_jni.h>
#include <jni.h>

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

//#define ONLY_INIT

ANativeWindow *native_window;

extern "C" JNIEXPORT void JNICALL
Java_com_innopeaktech_naboo_taichi_1test_NativeLib_init(JNIEnv *env,
                                                        jclass,
                                                        jobject assets,
                                                        jobject surface) {
  native_window = ANativeWindow_fromSurface(env, surface);

  run_init(
    /*width=*/ANativeWindow_getWidth(native_window),
    /*height=*/ANativeWindow_getHeight(native_window),
    "/data/local/tmp/",
    native_window);
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
  float a_x = g_x > 2 || g_x < -2 ? -g_x * 4 : 0;
  float a_y = g_y > 2 || g_y < -2 ? -g_y * 4 : 0;
  float a_z = g_z > 2 || g_z < -2 ? -g_z * 4 : 0;

  run_render_loop(a_x, a_y, a_z);
}
