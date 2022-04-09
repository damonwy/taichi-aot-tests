from model_data import *
import argparse, re, pathlib, shutil, os

import numpy as np
from taichi._lib import core as _ti_core

import taichi as ti

parser = argparse.ArgumentParser()
parser.add_argument('--exp',
                    choices=['implicit', 'explicit'],
                    default='implicit')
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--gui', choices=['auto', 'ggui', 'cpu'], default='auto')
#parser.add_argument('--mesh', default=None)
#parser.add_argument('--mesh', default='/media/hdd/model/scale/bunny/bunny0.1.node')
#parser.add_argument('--mesh', default='./armadillo0.1.node')
parser.add_argument('--mesh', default='./fox.1.node')
parser.add_argument('--aot', default=False, action='store_true')
parser.add_argument('place_holder', nargs='*')
args = parser.parse_args()

ti.init(arch=ti.vulkan, dynamic_index=False)

if args.gui == 'auto':
    if _ti_core.GGUI_AVAILABLE:
        args.gui = 'ggui'
    else:
        args.gui = 'cpu'

if 'ele_str' in locals():
    def read_np(str, dim, face_flag=False, node_flag=False):
        data = str.split('\n')
        n = int(re.findall(r'\S+', data[0])[0])
        array_np = []
        for line in data[1: n + 1]:
            xs = re.findall(r'\S+', line)[1: dim + 1]
            #if face_flag: xs[0], xs[1] = xs[1], xs[0] # flip the face
            if node_flag: array_np.append([float(i) for i in xs])
            else : array_np.append([int(i) for i in xs])
        return [n, np.array(array_np)]

    n_verts, array_np = read_np(node_str, 3, node_flag=True)
    ma = 0
    for i in range(3):
        array_np[:, i] -= array_np[:, i].min()
        ma = max(ma, array_np[:, i].max())
    array_np /= ma
    array_np[:, 2] *= -1
    ox_np = array_np

    n_cells, vertices_np = read_np(ele_str, 4)

    n_faces, indices_np = read_np(face_str, 3, face_flag=True)

    edges_np = set()
    for i in vertices_np:
        for u in i:
            for v in i:
                if u < v:
                    edges_np.add((u, v))
    edges_np = np.array(list(sorted(edges_np)))
    n_edges = len(edges_np)

    map = {}
    for i, j in enumerate(edges_np):
        map[tuple(j)] = i
    c2e_np = []
    for i in vertices_np:
        tmp = []
        for u in i:
            for u1 in i:
                if u < u1:
                    tmp.append(map[(u, u1)])
        c2e_np.append(tmp)
    c2e_np = np.array(c2e_np)

    def serialize():
        def write_array(x, t, name, f):
            if t is float:
                f.write('float {}'.format(name))
            else:
                f.write('int {}'.format(name))
            if len(x.shape) == 2:
                f.write('[{}][{}] = '.format(x.shape[0], x.shape[1]))
                f.write('{')
                for i in range(x.shape[0]):
                    f.write('{')
                    for j in range(x.shape[1]):
                        f.write(str(t(x[i, j])) + ',')
                    f.write('},')
                f.write('};\n')
            elif len(x.shape) == 1:
                f.write('[{}] = '.format(x.shape[0]))
                f.write('{')
                for i in range(x.shape[0]):
                    f.write(str(t(x[i])) + ',')
                f.write('};\n')
        data_header_path = '../include/data.h'
        os.remove(data_header_path)
        with open(data_header_path, 'w') as f:
            f.write('constexpr int N_VERTS = {};\n'.format(n_verts))
            f.write('constexpr int N_CELLS = {};\n'.format(n_cells))
            f.write('constexpr int N_FACES = {};\n'.format(n_faces))
            f.write('constexpr int N_EDGES = {};\n'.format(n_edges))
            write_array(ox_np, float, 'ox_data', f)
            write_array(vertices_np, int, 'vertices_data', f)
            write_array(indices_np.reshape(-1), int, 'indices_data', f)
            write_array(edges_np, int, 'edges_data', f)
            write_array(c2e_np, int, 'c2e_data', f)
    serialize()
# else: # read mesh from tetgen output file
#     res = re.search(r'^(.+)\.(\w+)$', args.mesh)
#     if res is not None and res.group(2) in ['node', 'ele', 'face']:
#         args.mesh = res.group(1)
#     def read_np(filename, dim):
#         with open(filename, 'r') as fi:
#             data = fi.readlines()
#             n = int(re.findall(r'\S+', data[0])[0])
#             array_np = []
#             for line in data[1: n + 1]:
#                 xs = re.findall(r'\S+', line)[1: dim + 1]
#                 if filename[-4:] == 'face': xs[0], xs[1] = xs[1], xs[0] # flip the face
#                 if filename[-4:] == 'node': array_np.append([float(i) for i in xs])
#                 else : array_np.append([int(i) for i in xs])
#             return [n, np.array(array_np)]

#     n_verts, array_np = read_np(f'{args.mesh}.node', 3)
#     ma = 0
#     for i in range(3):
#         array_np[:, i] -= array_np[:, i].min()
#         ma = max(ma, array_np[:, i].max())
#     array_np /= ma
#     ox = ti.Vector.field(args.dim, dtype=ti.f32, shape=n_verts)
#     ox.from_numpy(array_np)

#     n_cells, array_np = read_np(f'{args.mesh}.ele', 4)
#     vertices = ti.Vector.field(4, dtype=ti.i32, shape=n_cells)
#     vertices.from_numpy(array_np)

#     n_faces, array_np = read_np(f'{args.mesh}.face', 3)
#     indices = ti.field(ti.i32, shape=n_faces * 3)
#     indices.from_numpy(array_np.reshape(-1))


E, nu = 5e4, 0.0
mu, la = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # lambda = 0
density = 1000.0
dt_explicit = 2e-4
#dt_implicit = 1e-2
dt_implicit = 2.5e-3
#num_substep = int(dt_implicit / dt_explicit + 0.5)

if args.exp == 'implicit':
    dt = dt_implicit

if args.exp == 'explicit':
    dt = dt_explicit

num_substep = int(1e-2 / dt + 0.5)

x = ti.Vector.ndarray(args.dim, dtype=ti.f32, shape=n_verts)
v = ti.Vector.ndarray(args.dim, dtype=ti.f32, shape=n_verts)
f = ti.Vector.ndarray(args.dim, dtype=ti.f32, shape=n_verts)
mul_ans = ti.Vector.ndarray(args.dim, dtype=ti.f32, shape=n_verts)
m = ti.field(dtype=ti.f32, shape=n_verts)

B = ti.Matrix.field(args.dim, args.dim, dtype=ti.f32, shape=n_cells)
W = ti.field(dtype=ti.f32, shape=n_cells)

gravity = ti.Vector.field(3, dtype=ti.f32, shape=())

ox = ti.Vector.ndarray(args.dim, dtype=ti.f32, shape=n_verts)
vertices = ti.Vector.ndarray(4, dtype=ti.i32, shape=n_cells)
indices = ti.field(ti.i32, shape=n_faces * 3)
edges = ti.Vector.ndarray(2, dtype=ti.i32, shape=n_edges)
c2e = ti.Vector.ndarray(6, dtype=ti.i32, shape=n_cells)

hes_edge = ti.field(dtype=ti.f32, shape=edges.shape)
hes_vert = ti.field(dtype=ti.f32, shape=ox.shape)

b = ti.Vector.ndarray(3, dtype=ti.f32, shape=n_verts)
r0 = ti.Vector.ndarray(3, dtype=ti.f32, shape=n_verts)
p0 = ti.Vector.ndarray(3, dtype=ti.f32, shape=n_verts)
alpha_scalar = ti.ndarray(ti.f32, shape=())
beta_scalar = ti.ndarray(ti.f32, shape=())

dot_ans = ti.field(ti.f32, shape=())
r_2_scalar = ti.field(ti.f32, shape=())

ox.from_numpy(ox_np)
vertices.from_numpy(vertices_np)
indices.from_numpy(indices_np.reshape(-1))

edges.from_numpy(np.array(list(edges_np)))
c2e.from_numpy(c2e_np)


@ti.kernel
def clear_field():
    for I in ti.grouped(hes_edge):
        hes_edge[I] = 0
    for I in ti.grouped(hes_vert):
        hes_vert[I] = 0

@ti.func
def Ds(verts, x: ti.template()):
    return ti.Matrix.cols([x[verts[i]] - x[verts[3]] for i in range(3)])


@ti.func
def ssvd(F):
    U, sig, V = ti.svd(F)
    if U.determinant() < 0:
        for i in ti.static(range(3)):
            U[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    if V.determinant() < 0:
        for i in ti.static(range(3)):
            V[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    return U, sig, V


@ti.func
def get_force_func(c, verts, x: ti.template(), f: ti.template()):
    F = Ds(verts, x) @ B[c]
    P = ti.Matrix.zero(ti.f32, 3, 3)
    U, sig, V = ssvd(F)
    P = 2 * mu * (F - U @ V.transpose())
    H = -W[c] * P @ B[c].transpose()
    for i in ti.static(range(3)):
        force = ti.Vector([H[j, i] for j in range(3)])
        f[verts[i]] += force
        f[verts[3]] -= force


@ti.kernel
def get_force(x: ti.any_arr(), f: ti.any_arr(), vertices: ti.any_arr(), g_x: ti.f32, g_y: ti.f32, g_z: ti.f32):
    for c in vertices:
        get_force_func(c, vertices[c], x, f)
    for u in f:
        g = ti.Vector([g_x, g_y, g_z])
        f[u] += g * m[u]


@ti.kernel
def get_matrix(c2e: ti.any_arr(), vertices: ti.any_arr()):
    for c in vertices:
        verts = vertices[c]
        W_c = W[c]
        B_c = B[c]
        hes = ti.Matrix.zero(ti.f32, 12, 12)
        for u in ti.static(range(4)):
            for d in ti.static(range(3)):
                dD = ti.Matrix.zero(ti.f32, 3, 3)
                if ti.static(u == 3):
                    for j in ti.static(range(3)):
                        dD[d, j] = -1
                else:
                    dD[d, u] = 1
                dF = dD @ B_c
                dP = 2.0 * mu * dF
                dH = -W_c * dP @ B_c.transpose()
                for i in ti.static(range(3)):
                    for j in ti.static(range(3)):
                        hes[i * 3 + j, u * 3 + d] = -dt**2 * dH[j, i]
                        hes[3 * 3 + j, u * 3 + d] += dt**2 * dH[j, i]

        z = 0
        for u_i in ti.static(range(4)):
            u = verts[u_i]
            for v_i in ti.static(range(4)):
                v = verts[v_i]
                if u < v:
                    hes_edge[c2e[c][z]] += hes[u_i * 3, v_i * 3]
                    z += 1
        for zz in ti.static(range(4)):
            hes_vert[verts[zz]] += hes[zz * 3, zz * 3]

# @ti.kernel
# def matmul_cell(ret: ti.template(), vel: ti.template()):
#     for i in ret:
#         ret[i] = vel[i] * m[i]
#     for c in vertices:
#         verts = vertices[c]
#         W_c = W[c]
#         B_c = B[c]
#         for u in range(4):
#             for d in range(3):
#                 dD = ti.Matrix.zero(ti.f32, 3, 3)
#                 if u == 3:
#                     for j in range(3):
#                         dD[d, j] = -1
#                 else:
#                     dD[d, u] = 1
#                 dF = dD @ B_c
#                 dP = 2.0 * mu * dF
#                 dH = -W_c * dP @ B_c.transpose()
#                 for i in range(3):
#                     for j in range(3):
#                         tmp = (vel[verts[i]][j] - vel[verts[3]][j])
#                         ret[verts[u]][d] += -dt**2 * dH[j, i] * tmp

@ti.kernel
def matmul_edge(ret: ti.any_arr(), vel: ti.any_arr(), edges: ti.any_arr()):
    for i in ret:
        ret[i] = vel[i] * m[i] + hes_vert[i] * vel[i]
    for e in edges:
        u = edges[e][0]
        v = edges[e][1]
        ret[u] += hes_edge[e] * vel[v]
        ret[v] += hes_edge[e] * vel[u]


@ti.kernel
def add(ans: ti.any_arr(), a: ti.any_arr(), k: ti.f32, b: ti.any_arr()):
    for i in ans:
        ans[i] = a[i] + k * b[i]

@ti.kernel
def add_hack(ans: ti.any_arr(), a: ti.any_arr(), k: ti.f32, scalar: ti.any_arr(), b: ti.any_arr()):
    for i in ans:
        ans[i] = a[i] + k * scalar[None] * b[i]


@ti.kernel
def dot2scalar(a: ti.any_arr(), b: ti.any_arr()):
    dot_ans[None] = 0.0
    for i in a:
        dot_ans[None] += a[i].dot(b[i])


@ti.kernel
def get_b(v: ti.any_arr(), b: ti.any_arr(), f: ti.any_arr()):
    for i in b:
        b[i] = m[i] * v[i] + dt * f[i]


@ti.kernel
def ndarray_to_ndarray(ndarray: ti.any_arr(), other: ti.any_arr()):
    for I in ti.grouped(ndarray):
        ndarray[I] = other[I]


@ti.kernel
def fill_ndarray(ndarray: ti.any_arr(), val: ti.f32):
    for I in ti.grouped(ndarray):
        ndarray[I] = [val, val, val]

@ti.kernel
def init_r_2():
    r_2_scalar[None] = dot_ans[None]

@ti.kernel
def update_alpha(alpha_scalar: ti.any_arr()):
    alpha_scalar[None] = r_2_scalar[None] / dot_ans[None]

@ti.kernel
def update_beta_r_2(beta_scalar: ti.any_arr()):
    beta_scalar[None] = dot_ans[None] / r_2_scalar[None]
    r_2_scalar[None] = dot_ans[None]


def cg(it):
    get_force(x, f, vertices, 3, -9.8, 0)
    get_b(v, b, f)
    matmul_edge(mul_ans, v, edges)
    matmul_edge(mul_ans, v, edges)
    add(r0, b, -1, mul_ans)

    ndarray_to_ndarray(p0, r0)
    dot2scalar(r0, r0)
    init_r_2()
    n_iter = 10
    for iter in range(n_iter):
        matmul_edge(mul_ans, p0, edges)
        dot2scalar(p0, mul_ans)
        update_alpha(alpha_scalar)
        add_hack(v, v, 1, alpha_scalar, p0)
        add_hack(r0, r0, -1, alpha_scalar, mul_ans)
        dot2scalar(r0, r0)
        update_beta_r_2(beta_scalar)
        add_hack(p0, r0, 1, beta_scalar, p0)
    fill_ndarray(f, 0)
    add(x, x, dt, v)


@ti.kernel
def advect():
    for p in x:
        v[p] += dt * (f[p] / m[p])
        x[p] += dt * v[p]
        f[p] = ti.Vector([0, 0, 0])


@ti.kernel
def init(x: ti.any_arr(), v: ti.any_arr(), f: ti.any_arr(), ox: ti.any_arr(), vertices: ti.any_arr()):
    gravity[None] = [0, -9.8, 0]
    for u in x:
        x[u] = ox[u]
        v[u] = [0.0] * 3
        f[u] = [0.0] * 3
        m[u] = 0.0
    for c in vertices:
        F = Ds(vertices[c], x)
        B[c] = F.inverse()
        W[c] = ti.abs(F.determinant()) / 6
        for i in ti.static(range(4)):
            m[vertices[c][i]] += W[c] / 4 * density

@ti.kernel
def floor_bound(x: ti.any_arr(), v: ti.any_arr()):
    for u in x:
        for i in ti.static(range(3)):
            if x[u][i] < -1:
                x[u][i] = -1
                if v[u][i] < 0:
                    v[u][i] = 0
            if x[u][i] > 1:
                x[u][i] = 1
                if v[u][i] > 0:
                    v[u][i] = 0




def substep():
    if args.exp == 'explicit':
        for i in range(num_substep):
            get_force(x, f)
            advect()
    else:
        for i in range(num_substep):
            cg(i)
    floor_bound(x, v)


def run_aot():
    dir_name = '../shaders/aot/implicit_fem'
    shutil.rmtree(dir_name, ignore_errors=True)
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=False)

    mod = ti.aot.Module(ti.metal)
    mod.add_kernel(get_force, (x, f, vertices))
    mod.add_kernel(init, (x, v, f, ox, vertices))
    mod.add_kernel(floor_bound, (x, v))
    mod.add_kernel(get_matrix, (c2e, vertices))
    mod.add_kernel(matmul_edge, (mul_ans, x, edges))
    mod.add_kernel(add, (x, x, v))
    mod.add_kernel(add_hack, (x, x, alpha_scalar, v))
    mod.add_kernel(dot2scalar, (r0, r0))
    mod.add_kernel(get_b, (v, b, f))
    mod.add_kernel(ndarray_to_ndarray, (p0, r0))
    mod.add_kernel(fill_ndarray, (f, ))
    mod.add_kernel(clear_field)
    mod.add_kernel(init_r_2)
    mod.add_kernel(update_alpha, (alpha_scalar, ))
    mod.add_kernel(update_beta_r_2, (beta_scalar, ))
    mod.save(dir_name, '')
    print('AOT done')


@ti.kernel
def convert_to_field(x: ti.any_arr(), y: ti.template()):
    for I in ti.grouped(x):
        y[I] = x[I]


if __name__ == '__main__':
    if args.aot:
        run_aot()
    else:
        clear_field()
        init(x, v, f, ox, vertices)
        get_matrix(c2e, vertices)

        def handle_interaction(window):
            #print(window.event)
            gravity[None] = [0, -9.8, 0]
            if window.is_pressed('i'):
                gravity[None] = [0, 0, -9.8]
            if window.is_pressed('k'):
                gravity[None] = [0, 0, 9.8]
            if window.is_pressed('o'):
                gravity[None] = [0, 9.8, 0]
            if window.is_pressed('u'):
                gravity[None] = [0, -9.8, 0]
            if window.is_pressed('l'):
                gravity[None] = [9.8, 0, 0]
            if window.is_pressed('j'):
                gravity[None] = [-9.8, 0, 0]


        if args.gui == 'ggui':
            res = (800, 600)
            window = ti.ui.Window("Implicit FEM", res, vsync=True)

            frame_id = 0
            canvas = window.get_canvas()
            scene = ti.ui.Scene()
            camera = ti.ui.make_camera()
            camera.position(0.0, 1.5, 2.95)
            camera.lookat(0.0, 0.0, 0.0)
            camera.fov(55)

            x_field = ti.Vector.field(3, dtype=ti.f32, shape=n_verts)

            def render():
                convert_to_field(x, x_field)
                camera.track_user_inputs(window,
                                         movement_speed=0.03,
                                         hold_key=ti.ui.RMB)
                scene.set_camera(camera)

                scene.ambient_light((0.1, ) * 3)

                scene.point_light(pos=(0.5, 10.0, 0.5), color=(0.5, 0.5, 0.5))
                scene.point_light(pos=(10.0, 10.0, 10.0), color=(0.5, 0.5, 0.5))

                scene.mesh(x_field, indices, color=(0.73, 0.33, 0.23))

                canvas.scene(scene)

            while window.running:
                frame_id += 1
                frame_id = frame_id % 256
                substep()
                print(x[0][0], x[0][1], x[0][2])
                if window.is_pressed('r'):
                    init(x, v, f)
                if window.is_pressed(ti.GUI.ESCAPE):
                    break
                handle_interaction(window)

                render()

                window.show()

        else:

            def T(a):

                phi, theta = np.radians(28), np.radians(32)

                a = a - 0.2
                x, y, z = a[:, 0], a[:, 1], a[:, 2]
                c, s = np.cos(phi), np.sin(phi)
                C, S = np.cos(theta), np.sin(theta)
                x, z = x * c + z * s, z * c - x * s
                u, v = x, y * C + z * S
                return np.array([u, v]).swapaxes(0, 1) + 0.5

            gui = ti.GUI('Implicit FEM')
            while gui.running:
                substep()
                if gui.get_event(ti.GUI.PRESS):
                    if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]: break
                if gui.is_pressed('r'):
                    init(x, v, f)
                handle_interaction(gui)
                gui.clear(0x000000)
                gui.circles(T(x.to_numpy() / 3), radius=1.5, color=(1.0, 1.0, 1.0))
                gui.show()
