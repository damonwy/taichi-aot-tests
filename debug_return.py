import taichi as ti

ti.init(ti.vulkan)

@ti.kernel
def func1() -> ti.f32:
    return 1.0

@ti.kernel
def func2() -> ti.f32:
    return -2.0


mod = ti.aot.Module(ti.vulkan)

mod.add_kernel(func1)
mod.add_kernel(func2)
mod.save('debug', '')
