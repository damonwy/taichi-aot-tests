# taichi-vk-launcher

Make sure to apply the patch `0002-WIP-Allow-temporary-reading-root-buffer-memory-for-t.patch` 
`0003-WIP-Add-stable-fluid-ndarray-aot.patch` `0004-WIP-Modify-set-image-class-for-t.patch`in your Taichi repo:

```
git apply 0002-WIP-Allow-temporary-reading-root-buffer-memory-for-t.patch
git apply 0003-WIP-Add-stable-fluid-ndarray-aot.patch
git apply 0004-WIP-Modify-set-image-class-for-t.patch
```

## Desktop Build
```
export TAICHI_REPO_DIR=/path/github/taichi/
cd desktop
make
```

Taichi built with
```
TAICHI_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_CUDA:BOOL=OFF -DTI_WITH_OPENGL:BOOL=OFF -DTI_WITH_LLVM:BOOL=OFF -DTI_EXPORT_CORE:BOOL=ON -DTI_WITH_LLVM:BOOL=OFF" python3 setup.py build_ext
```

## Android Build
If you are building Taichi with custom changes, make sure to override the prebuilt binaries in: `app/src/main/jniLibs/arm64-v8a/`
```
export TAICHI_REPO_DIR=/path/github/taichi/
cd android
./gradlew assembleDebug
adb install ./app/build/outputs/apk/debug/app-debug.apk
adb push ../mpm88 /data/local/tmp/
adb shell chmod -R 777 /data/local/tmp/mpm88/
adb push $TAICHI_REPO_DIR/python/taichi/shaders /data/local/tmp/
```

Taichi built with
```
TAICHI_CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=${ANDROID_SDK_ROOT}/ndk/22.1.7171670/build/cmake/android.toolchain.cmake -DANDROID_NATIVE_API_LEVEL=29 -DANDROID_ABI=arm64-v8a -DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_CUDA:BOOL=OFF -DTI_WITH_OPENGL:BOOL=OFF -DTI_WITH_LLVM:BOOL=OFF -DTI_EXPORT_CORE:BOOL=ON" python3 setup.py build_ext
```

## Android Monitor Logs

```
adb logcat -s "TaichiTest" "*:F" "AndroidRuntime" vulkan VALIDATION
```
