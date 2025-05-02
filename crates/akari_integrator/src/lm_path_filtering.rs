use std::{
    cell, f32::consts::FRAC_1_PI, fs::File, io::BufWriter, rc::Rc, sync::Arc, time::Instant,
};

use akari_render::svm::ShaderRef;
use luisa::{rtx::offset_ray_origin, BindGroup};

use super::{Integrator, RenderSession};
use crate::pt::{BsdfDenoisingFeatures, DirectLighting, SurfaceHit, SurfaceHitComps};
use crate::{
    color::*,
    geometry::*,
    interaction::SurfaceInteraction,
    light::LightEvalContext,
    sampler::*,
    svm::surface::{diffuse::DiffuseBsdf, *},
    *,
};

#[tracked(crate = "luisa")]
fn hash(p: Expr<Uint3>) -> Expr<u64> {
    (p.x.as_u64() * 73856093) ^ (p.y.as_u64() * 19349663) ^ (p.z.as_u64() * 83492791)
}
#[derive(Clone, Copy, Debug, Value)]
#[luisa(crate = "luisa")]
#[repr(C)]
pub struct GridState {
    pub p_min: Float3,
    pub p_max: Float3,
    pub hash_table_size: u32,
    pub point_count: u32,
    pub res: Uint3,
    pub radius: f32,
}
#[derive(BindGroup)]
#[luisa(crate = "luisa")]
pub struct HashGrid<T: Value> {
    pub pos: Buffer<Float3>,
    pub data: Buffer<T>,
    pub next: Buffer<u32>,
    pub cell_head: Buffer<u32>,
    pub state: Buffer<GridState>,
    pub cnt: Buffer<u32>,
}
impl<T: Value> HashGridVar<T> {
    #[tracked(crate = "luisa")]
    pub fn get_cell(&self, p: Expr<Float3>) -> Expr<Uint3> {
        let state = self.state.read(0);
        let res = state.res;
        let p = (p - state.p_min) / (state.p_max - state.p_min);
        let p = p.clamp(Float3::expr(0.0, 0.0, 0.0), Float3::expr(1.0, 1.0, 1.0));
        let p = p * state.res.cast_f32();
        p.cast_u32().min_(res - 1)
    }
    #[tracked(crate = "luisa")]
    pub fn cell_pos_to_index(&self, cell: Expr<Uint3>) -> Expr<u32> {
        let state = self.state.read(0);
        (hash(cell) % state.hash_table_size.as_u64()).cast_u32()
    }
    #[tracked(crate = "luisa")]
    pub fn for_each_neighbor(
        &self,
        p: Expr<Float3>,
        f: impl Fn(Expr<Float3>, Expr<T>) -> Expr<bool>,
    ) {
        let cell = self.get_cell(p).cast_i32();
        // let cnt = 0u32.var();
        let should_break = false.var();
        // let visited = Var::<[u32; 27]>::zeroed();
        // for i in 0u32..27 {
        //     visited.write(i, u32::MAX);
        // }
        // let visited_cnt = 0u32.var();
        for dz in (-1i32.expr())..=1i32.expr() {
            if should_break {
                break;
            }
            for dy in (-1i32.expr())..=1.expr() {
                if should_break {
                    break;
                }
                for dx in (-1i32.expr())..=1.expr() {
                    if should_break {
                        break;
                    }
                    let new_cell = cell + Int3::expr(dx, dy, dz);
                    if (new_cell < Int3::expr(0, 0, 0)).any()
                        | (new_cell >= self.state.read(0).res.cast_i32()).any()
                    {
                        continue;
                    }

                    let cell_idx = self.cell_pos_to_index(new_cell.cast_u32());
                    // {
                    //     let already_visited = false.var();
                    //     for i in 0u32.expr()..**visited_cnt {
                    //         if visited.read(i) == cell_idx {
                    //             *already_visited = true;
                    //             break;
                    //         }
                    //     }
                    //     if already_visited {
                    //         continue;
                    //     }
                    //     visited.write(visited_cnt, cell_idx);
                    //     *visited_cnt += 1;
                    // }

                    let idx = self.cell_head.read(cell_idx).var();
                    while idx != u32::MAX {
                        let data = self.data.read(idx);
                        let continue_ = f(self.pos.read(idx), data);
                        if !continue_ {
                            *should_break = true;
                            break;
                        }
                        *idx = self.next.read(idx);
                    }
                }
            }
        }
        //    device_log!("cnt: {}", cnt.load());
    }
}
impl<T: Value> HashGrid<T> {
    pub fn new(
        device: &Device,
        aabb: (Float3, Float3),
        res: Uint3,
        radius: f32,
        hash_size: usize,
        point_count: usize,
    ) -> Self {
        let state = GridState {
            p_min: aabb.0,
            p_max: aabb.1,
            hash_table_size: hash_size as u32,
            point_count: point_count as u32,
            radius,
            res,
        };
        let pos = device.create_buffer::<Float3>(point_count);
        let data = device.create_buffer::<T>(point_count);
        let next = device.create_buffer_from_fn::<u32>(point_count, |_| u32::MAX);
        let cell_head = device.create_buffer_from_fn::<u32>(hash_size, |_| u32::MAX);
        let state = device.create_buffer_from_slice(&[state]);
        let cnt = device.create_buffer_from_slice::<u32>(&[0]);
        Self {
            pos,
            data,
            next,
            cell_head,
            state,
            cnt,
        }
    }
    pub fn mem_usage(&self) -> usize {
        self.pos.len() * std::mem::size_of::<Float3>()
            + self.data.len() * std::mem::size_of::<T>()
            + self.next.len() * std::mem::size_of::<u32>()
            + self.cell_head.len() * std::mem::size_of::<u32>()
            + self.state.len() * std::mem::size_of::<GridState>()
            + self.cnt.len() * std::mem::size_of::<u32>()
    }
    pub fn reset(&self, aabb: (Float3, Float3), res: Uint3, radius: f32) {
        let state = GridState {
            p_min: aabb.0,
            p_max: aabb.1,
            hash_table_size: self.cell_head.len() as u32,
            point_count: self.pos.len() as u32,
            radius,
            res,
        };
        self.state.copy_from(&[state]);
    }
    #[tracked(crate = "luisa")]
    pub fn sanity_check(&self, device: &Device) {
        let kernel = device.create_kernel::<fn(HashGrid<T>)>(&|grid: HashGridVar<T>| {
            let tid = dispatch_id().x;
            let pos = grid.pos.read(tid);
            let cell = grid.get_cell(pos);
            let cell_idx = grid.cell_pos_to_index(cell);
            let idx = grid.cell_head.read(cell_idx).var();
            let found = false.var();
            while idx != u32::MAX {
                let p = grid.pos.read(idx);
                if (pos - p).length() == 0.0 {
                    *found = true;
                    break;
                }
                *idx = grid.next.read(idx);
            }
            if !found {
                device_log!("not found: {}", pos);
            }
            // let radius = grid.state.read(0).radius;
            // let perturbed = pos + radius * 0.01;
            // let perturbed_cell = grid.get_cell(perturbed);
            // let diff = perturbed_cell.cast_i32() - cell.cast_i32();
            // let perturbed_cell_idx = grid.cell_pos_to_index(perturbed_cell);
            // device_log!("cell_idx: {}, perturbed_cell_idx: {}", cell_idx, perturbed_cell_idx);
            // lc_assert!(diff.abs().reduce_max().le(1));
        });
        kernel.dispatch([self.cnt.copy_to_vec()[0], 1, 1], self);
        let cnt = device.create_buffer_from_slice::<u32>(&[0]);
        let kernel = device.create_kernel::<fn(HashGrid<T>)>(&|grid: HashGridVar<T>| {
            let tid = dispatch_id().x;
            let count = 0u32.var();
            let idx = grid.cell_head.read(tid).var();
            while idx != u32::MAX {
                *count += 1;
                *idx = grid.next.read(idx);
            }
            cnt.atomic_fetch_add(0, count);
        });
        kernel.dispatch([self.cell_head.len() as u32, 1, 1], self);
        assert_eq!(cnt.copy_to_vec()[0], self.cnt.copy_to_vec()[0]);
    }
    #[tracked(crate = "luisa")]
    pub fn reset_kernel(device: &Device) -> Kernel<fn(HashGrid<T>)> {
        let kernel = device.create_kernel_async::<fn(HashGrid<T>)>(&|grid: HashGridVar<T>| {
            let tid = dispatch_id().x;
            if tid == 0 {
                grid.cnt.write(0, 0);
            }
            grid.cell_head.write(tid, u32::MAX);
        });
        kernel
    }
    #[tracked(crate = "luisa")]
    pub fn build_kernel(device: &Device) -> Kernel<fn(Buffer<Float3>, Buffer<T>, HashGrid<T>)> {
        let kernel = device.create_kernel_async::<fn(Buffer<Float3>, Buffer<T>, HashGrid<T>)>(
            &|pts: BufferVar<Float3>, data: BufferVar<T>, grid: HashGridVar<T>| {
                let tid = dispatch_id().x;
                let pt = pts.read(tid);
                let data = data.read(tid);
                let cell = grid.get_cell(pt);
                let cell_idx = grid.cell_pos_to_index(cell);
                let data_idx = grid.cnt.atomic_fetch_add(0, 1);
                loop {
                    let old = grid.cell_head.read(cell_idx);
                    let actual = grid
                        .cell_head
                        .atomic_compare_exchange(cell_idx, old, data_idx);
                    if actual == old {
                        grid.pos.write(data_idx, pt);
                        grid.data.write(data_idx, data);
                        grid.next.write(data_idx, old);
                        break;
                    }
                }
            },
        );
        kernel
    }
}

#[derive(Clone)]
pub struct PathFiltering {
    pub device: Device,
    config: Config,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(crate = "serde")]
#[serde(default)]
pub struct Config {
    pub max_depth: u32,
    pub repeats: u32,
    pub branch: u32,
    pub radius_scale: Vec<f32>,
    pub reuse: Vec<u32>,
    pub vram: usize, // in MB
    pub smis: bool,
    pub no_first_hit_stylization: bool,
    pub disable_flag_check: bool,
    pub cluster: bool,
    pub scene_bounds: Option<Vec<[f32; 3]>>,
    // pub final_gather: Option<u32>,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            max_depth: 7,
            repeats: 1,
            branch: 1,
            radius_scale: vec![0.1, 0.05],
            reuse: vec![128, 64],
            vram: 1024 * 2,
            smis: true,
            no_first_hit_stylization: false,
            disable_flag_check: false,
            cluster: true,
            scene_bounds: None,
            // final_gather: None,
        }
    }
}
impl PathFiltering {
    pub fn new(device: Device, config: Config) -> Self {
        Self { device, config }
    }
}

pub fn mis_weight(pdf_a: Expr<f32>, pdf_b: Expr<f32>, power: u32) -> Expr<f32> {
    let apply_power = |x: Expr<f32>| {
        let mut p = 1.0f32.expr();
        for _ in 0..power {
            p = track!(p * x);
        }
        p
    };
    let pdf_a = apply_power(pdf_a);
    let pdf_b = apply_power(pdf_b);
    track!(pdf_a / (pdf_a + pdf_b))
}
#[derive(Clone, Copy, Debug, Value)]
#[luisa(crate = "luisa")]
#[repr(C)]
pub struct IrradianceRecord {
    pub wi: [f32; 3],
    pub irradiance: [f32; 3],
    pub pdf: f32,
    pub mis: f32,
}
#[derive(Clone, Copy, Debug, Value)]
#[luisa(crate = "luisa")]
#[repr(C)]
pub struct Vertex {
    pub wo: [f32; 3],
    pub p: [f32; 3],
    pub hit: SurfaceHit,
    pub le: [f32; 3],
    pub mis: f32,
    pub direct: IrradianceRecord,
    pub indirect: IrradianceRecord,
    pub radiance: [f32; 3],
    pub bsdf_pdf: f32,
    pub flags: StylizationFlags,
    pub new_flags: StylizationFlags,
    pub sum_pdf: f32,
    pub cluster_id: u32,
    pub cluster_size: u32,
}
#[derive(Clone, Copy, Debug, Value)]
#[luisa(crate = "luisa")]
#[repr(C)]
pub struct PathState {
    pub depth: u32,
    pub px: Uint2,
    pub valid: bool,
    pub vertex_idx: u32,
    pub prev_p: Float3,
    pub prev_ng: Float3,
    pub prev_bsdf_pdf: f32,
    pub flags: StylizationFlags,
    pub terminated: bool,
}

struct RenderState {
    rng_buf: Buffer<Pcg32>,
    first_hit_radiance_buf: Buffer<Float3>,
    path_state_buf: Buffer<PathState>,
    reuses_per_depth: Buffer<u32>,
    counter: Buffer<u32>,
    neighbor_counter: Buffer<u64>,
}
impl PathFiltering {
    pub fn eval_context<'a>(
        &self,
        scene: &'a Scene,
        color_pipeline: ColorPipeline,
    ) -> (LightEvalContext<'a>, BsdfEvalContext) {
        let bsdf = BsdfEvalContext {
            color_repr: color_pipeline.color_repr,
            ad_mode: ADMode::None,
        };
        let light = LightEvalContext {
            svm: &scene.svm,
            meshes: &scene.meshes,
            surface_eval_ctx: bsdf,
            color_pipeline: color_pipeline,
        };
        (light, bsdf)
    }

    #[tracked(crate = "luisa")]
    pub fn handle_surface_light(
        &self,
        scene: &Scene,
        color_pipeline: ColorPipeline,
        swl: Var<SampledWavelengths>,
        si: SurfaceInteraction,
        path_depth: Expr<u32>,
        ray: Expr<Ray>,
        prev_ng: Expr<Float3>,
        prev_bsdf_pdf: Expr<f32>,
    ) -> (Color, Expr<f32>, Expr<bool>) {
        let instance = scene.meshes.mesh_instances().read(si.inst_id);

        if instance.light.valid() {
            let light_ctx = self.eval_context(scene, color_pipeline).0;
            let direct = scene.lights.le(ray, si, **swl, &light_ctx);
            // cpu_dbg!(direct.flatten());
            if path_depth == 0 {
                (direct, 1.0f32.expr(), true.expr())
            } else {
                let pn = {
                    let p = ray.o;
                    let n = prev_ng;
                    PointNormal::new_expr(p, n)
                };
                let light_pdf = scene.lights.pdf_direct(si, pn, &light_ctx);
                // device_log!("prev_bsdf_pdf: {}, light_pdf: {}", prev_bsdf_pdf, light_pdf);
                let w = mis_weight(prev_bsdf_pdf, light_pdf, 1);

                (direct, w, true.expr())
            }
        } else {
            (
                Color::zero(color_pipeline.color_repr),
                0.0f32.expr(),
                false.expr(),
            )
        }
    }

    #[tracked(crate = "luisa")]
    fn kernel_update_film(
        &self,
        state: &RenderState,
        scene: Arc<Scene>,
        vertex_buf: BufferVar<Vertex>,
        film: &Film,
        color_pipeline: ColorPipeline,
        scale: Expr<f32>,
    ) {
        let tid = dispatch_id().x;
        let ps = state.path_state_buf.read(tid);
        if !ps.valid {
            return;
        }
        let n_pixels = scene.camera.resolution().x * scene.camera.resolution().y;
        let px_idx = tid % n_pixels;
        let vertex_start_idx = ps.vertex_idx;
        let v = vertex_buf.read(vertex_start_idx);
        let si = scene.surface_interaction(v.hit.inst_id, v.hit.prim_id, v.hit.bary);
        let radiance = state.first_hit_radiance_buf.read(px_idx) * scale;
        let radiance = Color::from_flat(color_pipeline.color_repr, radiance.extend(0.0));
        let wo = Expr::<Float3>::from(v.wo);
        let stylized_radiance = if self.config.no_first_hit_stylization {
            radiance
        } else {
            scene.svm.dispatch_surface(
                si.surface,
                color_pipeline,
                si,
                SampledWavelengths::expr_zeroed(),
                |closure| {
                    closure.apply_stylization(
                        0u32.expr(),
                        si,
                        wo,
                        radiance,
                        &self.eval_context(&scene, color_pipeline).1,
                        v.flags,
                    )
                },
            )
        };
        film.add_sample(
            ps.px.cast_f32(),
            &stylized_radiance,
            SampledWavelengths::expr_zeroed(),
            1.0f32.expr(),
        );
    }

    #[tracked(crate = "luisa")]
    fn kernel_splat(&self, state: &RenderState, scene: &Scene, vertex_buf: BufferVar<Vertex>) {
        let tid = dispatch_id().x;
        let ps = state.path_state_buf.read(tid);
        if !ps.valid {
            return;
        }
        let vertex_start_idx = ps.vertex_idx;
        let v = vertex_buf.read(vertex_start_idx);
        let radiance = Expr::<Float3>::from(v.radiance);
        let px_idx = ps.px.y * scene.camera.resolution().x + ps.px.x;

        state
            .first_hit_radiance_buf
            .atomic_ref(px_idx)
            .x
            .fetch_add(radiance.x);
        state
            .first_hit_radiance_buf
            .atomic_ref(px_idx)
            .y
            .fetch_add(radiance.y);
        state
            .first_hit_radiance_buf
            .atomic_ref(px_idx)
            .z
            .fetch_add(radiance.z);
    }
    #[tracked(crate = "luisa")]
    fn kernel_gather_points(
        &self,
        state: &RenderState,
        scene: Arc<Scene>,
        path_depth: Expr<u32>,
        counter: BufferVar<u32>,
        pos: BufferVar<Float3>,
        indices: BufferVar<u32>,
        vertex_buf: BufferVar<Vertex>,
    ) {
        let tid = dispatch_id().x;
        let ps = state.path_state_buf.read(tid).var();
        if !ps.valid {
            return;
        }
        if path_depth > ps.depth {
            return;
        }
        let v = vertex_buf.read(ps.vertex_idx);
        let p = Expr::<Float3>::from(v.p);
        let i = counter.atomic_fetch_add(0, 1);
        pos.write(i, p);
        indices.write(i, tid);
    }
    #[tracked(crate = "luisa")]
    fn kernel_reset_cluster_id_buf(&self, cluster_id_buf: BufferVar<u32>) {
        let tid = dispatch_id().x;
        cluster_id_buf.write(tid, u32::MAX);
    }
    #[tracked(crate = "luisa")]
    fn kernel_init_cluster(
        &self,
        state: &RenderState,
        path_depth: Expr<u32>,
        vertex_buf: BufferVar<Vertex>,
        hash_grid: &HashGridVar<u32>,
        cluster_id_buf: BufferVar<u32>,
    ) {
        let tid = dispatch_id().x;
        let ps = state.path_state_buf.read(tid).var();
        if !ps.valid {
            return;
        }
        if path_depth > ps.depth {
            return;
        }
        let v = vertex_buf.read(ps.vertex_idx).var();
        if v.cluster_id != u32::MAX {
            return;
        }
        let p = Expr::<Float3>::from(**v.p);
        let cell_index = hash_grid.cell_pos_to_index(hash_grid.get_cell(p));
        if cluster_id_buf.atomic_compare_exchange(cell_index, u32::MAX, tid) == u32::MAX {
            *v.cluster_id = tid;
            // device_log!("cluster_id: {}, {}", tid, ps.vertex_idx);
            vertex_buf.write(ps.vertex_idx, v);
        }
    }
    #[tracked(crate = "luisa")]
    fn kernel_cluster(
        &self,
        state: &RenderState,
        path_depth: Expr<u32>,
        counter: BufferVar<u32>,
        vertex_buf: BufferVar<Vertex>,
        hash_grid: &HashGridVar<u32>,
    ) {
        let tid = dispatch_id().x;
        let ps = state.path_state_buf.read(tid).var();
        if !ps.valid {
            return;
        }
        if path_depth > ps.depth {
            return;
        }
        let v = vertex_buf.read(ps.vertex_idx).var();
        if v.cluster_id != u32::MAX {
            return;
        }
        let max_reuse = state.reuses_per_depth.read(path_depth);
        let p = Expr::<Float3>::from(**v.p);
        let pf_radius = hash_grid.state.read(0).radius;
        hash_grid.for_each_neighbor(p, |q, i| {
            let dist = (q - p).length();
            if dist <= pf_radius {
                let neighbor_vertex_idx = i;
                let neighbor_v = vertex_buf.read(neighbor_vertex_idx);
                if neighbor_v.cluster_id == i {
                    loop {
                        let cur_cnt = vertex_buf.read(neighbor_vertex_idx).cluster_size;
                        // device_log!("cur_cnt: {}", cur_cnt);
                        if cur_cnt >= max_reuse {
                            break;
                        }
                        let actual = vertex_buf
                            .atomic_ref(neighbor_vertex_idx)
                            .cluster_size
                            .compare_exchange(cur_cnt, cur_cnt + 1);
                        // device_log!("cur_cnt: {}, actual: {}", cur_cnt, actual);
                        if actual >= max_reuse {
                            break;
                        }
                        if actual == cur_cnt {
                            *v.cluster_id = neighbor_v.cluster_id;
                            // device_log!("cluster_id: {}, {}", neighbor_v.cluster_id, ps.vertex_idx);
                            vertex_buf.write(ps.vertex_idx, v);
                            return;
                        }
                    }
                }
            }
            true.expr()
        });
        // need to do clustering again
        counter.atomic_fetch_add(0, 1);
    }
    #[tracked(crate = "luisa")]
    fn kernel_evaluate_smis_weight(
        &self,
        state: &RenderState,
        scene: &Scene,
        color_pipeline: ColorPipeline,
        path_depth: Expr<u32>,
        ps_idx_offset: Expr<u32>,
        ps_indices: BufferVar<u32>,
        hash_grid: &HashGridVar<u32>,
        neighbor_buf: &BufferVar<u32>,
        vertex_buf: BufferVar<Vertex>,
    ) {
        let tid = dispatch_id().x;
        let ps_idx = ps_indices.read(ps_idx_offset + tid);
        let ps = state.path_state_buf.read(ps_idx).var();

        if !ps.valid {
            return;
        }
        if path_depth > ps.depth {
            return;
        }
        let pf_radius = hash_grid.state.read(0).radius;

        let v = vertex_buf.read(ps.vertex_idx).var();
        let si = scene.surface_interaction(**v.hit.inst_id, **v.hit.prim_id, **v.hit.bary);
        let p = si.p;
        let query_p = {
            let cluster_v = vertex_buf.read(**v.cluster_id);
            let center_p = Expr::<Float3>::from(cluster_v.p);
            let dist = (p - center_p).length();
            lc_assert!(dist.le(pf_radius));
            center_p
        };

        let sum_pdf = 0.0f32.var();
        let wi = Expr::<Float3>::from(**v.indirect.wi);
        if wi.length() == 0.0 {
            return;
        }
        let max_reuse = state.reuses_per_depth.read(path_depth);
        let neighbor_buf_offset = tid * max_reuse;
        let cluster_id = **v.cluster_id;
        let cnt = 0u32.var();
        hash_grid.for_each_neighbor(query_p, |q, neighbor_ps_idx| {
            let dist = (q - query_p).length();
            if dist <= pf_radius {
                let neighbor_ps = state.path_state_buf.read(neighbor_ps_idx);
                let valid = true.var();
                if (!neighbor_ps.valid) | (path_depth > neighbor_ps.depth) {
                    *valid = false;
                }
                if valid {
                    let neighbor_vertex_idx = neighbor_ps.vertex_idx;
                    let reuse_vertex = vertex_buf.read(neighbor_vertex_idx);
                    if self.config.cluster {
                        if reuse_vertex.cluster_id != cluster_id {
                            *valid = false;
                        }
                    }
                    if !self.config.disable_flag_check {
                        if !reuse_vertex.new_flags.is_equal(**v.new_flags) {
                            *valid = false;
                        }
                    }
                    if valid {
                        neighbor_buf.write(neighbor_buf_offset + cnt, neighbor_ps_idx);
                        *cnt += 1;
                    }
                }
            }
            true.expr()
        });
        for i in 0u32.expr()..**cnt {
            let neighbor_ps_idx = neighbor_buf.read(neighbor_buf_offset + i);
            let neighbor_ps = state.path_state_buf.read(neighbor_ps_idx);
            let valid = true.var();
            if (!neighbor_ps.valid) | (path_depth > neighbor_ps.depth) {
                *valid = false;
            }
            if valid {
                let neighbor_vertex_idx = neighbor_ps.vertex_idx;
                let reuse_vertex = vertex_buf.read(neighbor_vertex_idx);

                if reuse_vertex.cluster_id != v.cluster_id {
                    *valid = false;
                }

                if !self.config.disable_flag_check {
                    if !reuse_vertex.new_flags.is_equal(**v.new_flags) {
                        *valid = false;
                    }
                }
                if valid {
                    let alternative_neighbor_ps = state.path_state_buf.read(neighbor_ps_idx);
                    if (!alternative_neighbor_ps.valid)
                        | (path_depth > alternative_neighbor_ps.depth)
                    {
                        continue;
                    }
                    let alternative_neighbor_vertex_idx = alternative_neighbor_ps.vertex_idx;
                    let alternative_reuse_vertex = vertex_buf.read(alternative_neighbor_vertex_idx);
                    let alternative_si = scene.surface_interaction(
                        alternative_reuse_vertex.hit.inst_id,
                        alternative_reuse_vertex.hit.prim_id,
                        alternative_reuse_vertex.hit.bary,
                    );
                    let wo = Expr::<Float3>::from(alternative_reuse_vertex.wo);
                    let alternative_pdf = scene.svm.dispatch_surface(
                        alternative_si.surface,
                        color_pipeline,
                        alternative_si,
                        SampledWavelengths::expr_zeroed(),
                        |closure| {
                            closure
                                .evaluate(
                                    wo,
                                    wi,
                                    SampledWavelengths::expr_zeroed(),
                                    &self.eval_context(scene, color_pipeline).1,
                                )
                                .1
                        },
                    );
                    lc_assert!(alternative_pdf.ge(0.0));
                    *sum_pdf += alternative_pdf;
                }
            }
        }
        *v.sum_pdf = sum_pdf;
        vertex_buf.write(ps.vertex_idx, v);
    }
    #[tracked(crate = "luisa")]
    fn evaluate_filtered_radiance(
        &self,
        state: &RenderState,
        scene: &Scene,
        color_pipeline: ColorPipeline,
        rng: Var<Pcg32>,
        cluster_id: Expr<u32>,
        wo: Expr<Float3>,
        si: SurfaceInteraction,
        le: Expr<Float3>,
        path_depth: Expr<u32>,
        flags: Expr<StylizationFlags>,
        hash_grid: &HashGridVar<u32>,
        neighbor_buf: &BufferVar<u32>,
        vertex_buf: BufferVar<Vertex>,
    ) -> Expr<Float3> {
        let tid = dispatch_id().x;
        let indirect_radiance = Float3::var_zeroed();
        let direct_radiance = Float3::var_zeroed();
        let max_reuse = state.reuses_per_depth.read(path_depth);
        let neighbor_buf_offset = tid * max_reuse;
        let pf_radius = hash_grid.state.read(0).radius;
        let cnt = 0u32.var();
        let p = si.p;
        let query_p = if self.config.cluster {
            let cluster_v = vertex_buf.read(cluster_id);
            let center_p = Expr::<Float3>::from(cluster_v.p);
            let dist = (p - center_p).length();
            lc_assert!(dist.le(pf_radius));
            center_p
        } else {
            p
        };
        hash_grid.for_each_neighbor(query_p, |q, neighbor_ps_idx| {
            let dist = (q - query_p).length();
            if dist <= pf_radius {
                let neighbor_ps = state.path_state_buf.read(neighbor_ps_idx);
                let valid = true.var();
                if (!neighbor_ps.valid) | (path_depth > neighbor_ps.depth) {
                    *valid = false;
                }
                if valid {
                    let neighbor_vertex_idx = neighbor_ps.vertex_idx;
                    let reuse_vertex = vertex_buf.read(neighbor_vertex_idx);
                    if self.config.cluster {
                        if reuse_vertex.cluster_id != cluster_id {
                            *valid = false;
                        }
                    }
                    if !self.config.disable_flag_check {
                        if !reuse_vertex.new_flags.is_equal(flags) {
                            *valid = false;
                        }
                    }
                    if valid {
                        if cnt < max_reuse {
                            neighbor_buf.write(neighbor_buf_offset + cnt, neighbor_ps_idx);
                        } else {
                            let j = rng.gen_u32() % (cnt + 1);
                            if j < max_reuse {
                                neighbor_buf.write(neighbor_buf_offset + j, neighbor_ps_idx);
                            }
                        }
                        *cnt += 1;
                    }
                }
            }
            true.expr()
        });
        state.neighbor_counter.atomic_fetch_add(0, cnt.as_u64());
        let actual_cnt = cnt.min_(max_reuse);
        lc_assert!(actual_cnt.gt(0));
        for i in 0u32.expr()..actual_cnt {
            let neighbor_ps_idx = neighbor_buf.read(neighbor_buf_offset + i);
            let neighbor_ps = state.path_state_buf.read(neighbor_ps_idx);
            if (!neighbor_ps.valid) | (path_depth > neighbor_ps.depth) {
                continue;
            }
            let neighbor_vertex_idx = neighbor_ps.vertex_idx;
            let reuse_vertex = vertex_buf.read(neighbor_vertex_idx);
            // first compute direct
            let direct_record = reuse_vertex.direct;
            if direct_record.pdf > 0.0 {
                let wi = Expr::<Float3>::from(direct_record.wi);
                let (bsdf_f, bsdf_pdf) = scene.svm.dispatch_surface(
                    si.surface,
                    color_pipeline,
                    si,
                    SampledWavelengths::expr_zeroed(),
                    |closure| {
                        closure.evaluate(
                            wo,
                            wi,
                            SampledWavelengths::expr_zeroed(),
                            &self.eval_context(scene, color_pipeline).1,
                        )
                    },
                );
                let mis_weight = mis_weight(direct_record.pdf, bsdf_pdf, 1);
                *direct_radiance += bsdf_f.as_rgb()
                    * Expr::<Float3>::from(direct_record.irradiance)
                    * direct_record.mis
                    * mis_weight
                    / direct_record.pdf
                    / actual_cnt.as_f32();
            }
            let record = reuse_vertex.indirect;
            let wi = Expr::<Float3>::from(record.wi);
            if wi.length() == 0.0 {
                continue;
            }

            let (bsdf_f, bsdf_pdf) = scene.svm.dispatch_surface(
                si.surface,
                color_pipeline,
                si,
                SampledWavelengths::expr_zeroed(),
                |closure| {
                    closure.evaluate(
                        wo,
                        wi,
                        SampledWavelengths::expr_zeroed(),
                        &self.eval_context(scene, color_pipeline).1,
                    )
                },
            );
            let bsdf_f = bsdf_f.as_rgb();
            let reused_radiance = bsdf_f * Expr::<Float3>::from(record.irradiance) * record.mis;
            if self.config.smis {
                // now compute SMIS weight
                let sum_pdf = (0.0f32).var();
                if self.config.cluster && reuse_vertex.sum_pdf > 0.0 {
                    *sum_pdf = reuse_vertex.sum_pdf;
                } else {
                    for j in 0u32.expr()..actual_cnt {
                        let alternative_neighbor_ps_idx =
                            neighbor_buf.read(neighbor_buf_offset + j);
                        let alternative_neighbor_ps =
                            state.path_state_buf.read(alternative_neighbor_ps_idx);
                        if (!alternative_neighbor_ps.valid)
                            | (path_depth > alternative_neighbor_ps.depth)
                        {
                            continue;
                        }
                        let alternative_neighbor_vertex_idx = alternative_neighbor_ps.vertex_idx;
                        let alternative_reuse_vertex =
                            vertex_buf.read(alternative_neighbor_vertex_idx);
                        let alternative_si = scene.surface_interaction(
                            alternative_reuse_vertex.hit.inst_id,
                            alternative_reuse_vertex.hit.prim_id,
                            alternative_reuse_vertex.hit.bary,
                        );
                        let wo = Expr::<Float3>::from(alternative_reuse_vertex.wo);
                        let alternative_pdf = scene.svm.dispatch_surface(
                            alternative_si.surface,
                            color_pipeline,
                            alternative_si,
                            SampledWavelengths::expr_zeroed(),
                            |closure| {
                                closure
                                    .evaluate(
                                        wo,
                                        wi,
                                        SampledWavelengths::expr_zeroed(),
                                        &self.eval_context(scene, color_pipeline).1,
                                    )
                                    .1
                            },
                        );
                        lc_assert!(alternative_pdf.ge(0.0));
                        *sum_pdf += alternative_pdf;
                        // device_log!("alternative_pdf: {}", alternative_pdf);
                    }
                    if self.config.cluster {
                        vertex_buf
                            .atomic_ref(neighbor_vertex_idx)
                            .sum_pdf
                            .compare_exchange(0.0, sum_pdf);
                    }
                }
                lc_assert!(sum_pdf.gt(0.0));
                *indirect_radiance += reused_radiance / sum_pdf;
            } else {
                if bsdf_pdf > 0.0 {
                    *indirect_radiance += reused_radiance / bsdf_pdf / actual_cnt.as_f32();
                }
            }
        }

        let radiance = le + **direct_radiance + **indirect_radiance;
        let should_stylize = (path_depth > 0) | !self.config.no_first_hit_stylization;
        if should_stylize {
            scene
                .svm
                .dispatch_surface(
                    si.surface,
                    color_pipeline,
                    si,
                    SampledWavelengths::expr_zeroed(),
                    |closure| {
                        closure.apply_stylization(
                            path_depth,
                            si,
                            wo,
                            Color::from_flat(color_pipeline.color_repr, radiance.extend(0.0)),
                            &self.eval_context(&scene, color_pipeline).1,
                            flags,
                        )
                    },
                )
                .as_rgb()
        } else {
            radiance
        }
    }
    #[tracked(crate = "luisa")]
    fn kernel_propagate(
        &self,
        state: &RenderState,
        _scene: &Scene,
        ps_indices: BufferVar<u32>,
        path_depth: Expr<u32>,
        vertex_buf: BufferVar<Vertex>,
        record_buf: BufferVar<IrradianceRecord>,
    ) {
        let tid = dispatch_id().x;
        let ps_idx = ps_indices.read(tid);
        let ps = state.path_state_buf.read(ps_idx).var();

        if !ps.valid {
            return;
        }
        if path_depth > ps.depth {
            return;
        }
        let record = record_buf.read(ps_idx).var();
        let v = vertex_buf.read(ps.vertex_idx).var();
        *record.pdf = v.bsdf_pdf;
        *v.indirect = record;
        vertex_buf.write(ps.vertex_idx, v);
    }
    #[tracked(crate = "luisa")]
    fn kernel_filter(
        &self,
        state: &RenderState,
        scene: &Scene,
        color_pipeline: ColorPipeline,
        ps_idx_offset: Expr<u32>,
        ps_indices: BufferVar<u32>,
        path_depth: Expr<u32>,
        hash_grid: &HashGridVar<u32>,
        neighbor_buf: &BufferVar<u32>,
        vertex_buf: BufferVar<Vertex>,
        record_buf: BufferVar<IrradianceRecord>,
    ) {
        let tid = dispatch_id().x;
        let ps_idx = ps_indices.read(ps_idx_offset + tid);
        let ps = state.path_state_buf.read(ps_idx).var();
        let rng = state.rng_buf.read(tid).var();

        if !ps.valid {
            return;
        }
        if path_depth > ps.depth {
            return;
        }
        let vertex_start_idx = ps.vertex_idx;
        let v = vertex_buf.read(vertex_start_idx).var();
        let wo = Expr::<Float3>::from(**v.wo);
        let si = scene.surface_interaction(**v.hit.inst_id, **v.hit.prim_id, **v.hit.bary);
        let stylized_radiance = self.evaluate_filtered_radiance(
            state,
            scene,
            color_pipeline,
            rng,
            **v.cluster_id,
            wo,
            si,
            Expr::<Float3>::from(**v.le),
            path_depth,
            **v.new_flags,
            hash_grid,
            neighbor_buf,
            vertex_buf.clone(),
        );
        let new_record = IrradianceRecord::var_zeroed();
        *new_record.irradiance = Expr::<[f32; 3]>::from(stylized_radiance);
        *new_record.wi = Expr::<[f32; 3]>::from(-wo);

        *new_record.mis = v.mis;
        *v.radiance = new_record.irradiance;
        // device_log!(
        //     "le: {}, direct: {}, indirect: {}",
        //     le,
        //     direct_radiance,
        //     indirect_radiance
        // );
        if path_depth > 0 {
            record_buf.write(ps_idx, new_record);
        }
        vertex_buf.write(vertex_start_idx, v);
        state.rng_buf.write(tid, rng);
    }

    #[tracked(crate = "luisa")]
    fn kernel_mk_trace(
        &self,
        state: &RenderState,
        scene: Arc<Scene>,
        film: &Film,
        color_pipeline: ColorPipeline,
        px_idx: Expr<u32>,
        depth: Expr<u32>,
        max_trace_depth: Expr<u32>,
        vertex_buf: BufferVar<Vertex>,
        ray_buf: BufferVar<Ray>,
    ) {
        let tid = dispatch_id().x;

        let px_x = px_idx % scene.camera.resolution().x;
        let px_y = px_idx / scene.camera.resolution().x;

        let rng = state.rng_buf.read(tid).var();

        let sampler = IndependentSampler::from_pcg32(rng);
        let swl = SampledWavelengths::var_zeroed();

        let ps = PathState::var_zeroed();
        let vertex_idx = tid;
        let ray = if depth == 0 {
            *ps.depth = 0;
            *ps.px = Uint2::expr(px_x, px_y);
            let (ray, _) = scene.camera.generate_ray(
                &scene,
                film.filter(),
                Uint2::expr(px_x, px_y),
                &sampler,
                color_pipeline.color_repr,
                **swl,
            );
            *ps.prev_p = ray.o.var();
            *ps.prev_ng = ray.d.var();
            *ps.prev_bsdf_pdf = 0.0f32.var();
            *ps.vertex_idx = vertex_idx;
            *ps.terminated = false;
            ray
        } else {
            *ps = state.path_state_buf.read(tid);
            ray_buf.read(tid)
        };
        if ps.terminated {
            return;
        }

        let si = scene.intersect(ray);
        if si.valid {
            *ps.valid = true;
            *ps.depth = depth;
            let wo = -ray.d;
            let (le, mis, valid) = self.handle_surface_light(
                &scene,
                color_pipeline,
                swl,
                si,
                depth,
                ray,
                **ps.prev_ng,
                **ps.prev_bsdf_pdf,
            );

            let v = Vertex::var_zeroed();
            *v.cluster_id = u32::MAX;
            *v.hit = SurfaceHit::from_comps_expr(SurfaceHitComps {
                inst_id: si.inst_id,
                prim_id: si.prim_id,
                bary: si.bary,
            });
            *v.le = Expr::<[f32; 3]>::from(le.as_rgb());
            if !valid {
                *v.mis = 1.0;
            } else {
                *v.mis = mis;
            }
            *v.wo = Expr::<[f32; 3]>::from(-ray.d);
            *v.p = Expr::<[f32; 3]>::from(si.p);
            *v.flags = ps.flags;
            let new_flags =
                scene
                    .svm
                    .dispatch_surface(si.surface, color_pipeline, si, **swl, |closure| {
                        closure.update_stylization_flags(depth, si, wo, **ps.flags)
                    });
            *v.new_flags = new_flags;
            *ps.flags = new_flags;
            if depth >= max_trace_depth {
                vertex_buf.write(vertex_idx, v);
            } else {
                let depth = depth + 1;
                {
                    let u = sampler.next_3d();
                    let dl = self.sample_light(
                        scene.clone(),
                        color_pipeline,
                        swl,
                        si,
                        u,
                        depth,
                        new_flags,
                    );

                    let occluded = scene.occlude(dl.shadow_ray);
                    if !occluded {
                        let irradiance = dl.irradiance;
                        let record = IrradianceRecord::var_zeroed();
                        *record.irradiance = Expr::<[f32; 3]>::from(irradiance.as_rgb());
                        *record.wi = Expr::<[f32; 3]>::from(dl.wi);
                        *record.pdf = dl.pdf;
                        *record.mis = 1.0;
                        *v.direct = record;
                    }
                }

                let bs =
                    scene
                        .svm
                        .dispatch_surface(si.surface, color_pipeline, si, **swl, |closure| {
                            closure.sample(
                                wo,
                                sampler.next_1d(),
                                sampler.next_2d(),
                                swl,
                                &self.eval_context(&scene, color_pipeline).1,
                            )
                        });
                *v.bsdf_pdf = bs.pdf;
                vertex_buf.write(vertex_idx, v);
                if bs.pdf > 0.0 {
                    *ps.prev_bsdf_pdf = bs.pdf;
                    *ps.prev_ng = si.ng;
                    *ps.prev_p = si.p;
                    let ro = offset_ray_origin(si.p, face_forward(si.ng, bs.wi));
                    let new_ray = Ray::new_expr(
                        ro,
                        bs.wi,
                        0.0,
                        1e20,
                        Uint2::expr(si.inst_id, si.prim_id),
                        Uint2::expr(u32::MAX, u32::MAX),
                    );
                    ray_buf.write(tid, new_ray);
                } else {
                    *ps.terminated = true;
                }
            }
        } else {
            *ps.terminated = true;
        }

        state.path_state_buf.write(tid, ps);
        state.rng_buf.write(tid, rng);
    }

    #[tracked(crate = "luisa")]
    pub fn sample_surface_and_shade_direct(
        &self,
        scene: &Scene,
        color_pipeline: ColorPipeline,
        swl: Var<SampledWavelengths>,
        si: SurfaceInteraction,
        wo: Expr<Float3>,
        direct_lighting: DirectLighting,
        u_bsdf: Expr<Float3>,
    ) -> (BsdfSample, BsdfDenoisingFeatures, Color) {
        let ctx = self.eval_context(scene, color_pipeline).1;

        let sample_and_shade = |closure: &SurfaceClosure| {
            let direct = if direct_lighting.valid {
                let (f, pdf) = closure.evaluate(wo, direct_lighting.wi, **swl, &ctx);
                let w = mis_weight(direct_lighting.pdf, pdf, 1);
                direct_lighting.irradiance * f * w / direct_lighting.pdf
            } else {
                Color::zero(color_pipeline.color_repr)
            };

            let sample = closure.sample(wo, u_bsdf.x, u_bsdf.yz(), swl, &ctx);
            let albedo = closure.albedo(wo, **swl, &ctx) + closure.emission(wo, **swl, &ctx);
            let roughness = closure.avg_roughness(wo, **swl, &ctx);
            (
                sample,
                BsdfDenoisingFeatures {
                    albedo,
                    roughness,
                    roughness_interval: 0.0f32.expr(),
                },
                direct,
            )
        };
        scene
            .svm
            .dispatch_surface(si.surface, color_pipeline, si, **swl, &sample_and_shade)
    }

    #[tracked(crate = "luisa")]
    pub fn sample_light(
        &self,
        scene: Arc<Scene>,
        color_pipeline: ColorPipeline,
        swl: Var<SampledWavelengths>,
        si: SurfaceInteraction,
        u: Expr<Float3>,
        path_depth: Expr<u32>,
        flags: Expr<StylizationFlags>,
    ) -> DirectLighting {
        let p = si.p;
        let ng = si.ng;
        let pn = PointNormal::new_expr(p, ng);

        let sample = scene.lights.sample_direct(
            pn,
            u.x,
            u.yz(),
            **swl,
            &self.eval_context(&scene, color_pipeline).0,
        );
        if sample.valid {
            let wi = sample.wi;
            let shadow_ray = sample.shadow_ray.var();
            *shadow_ray.exclude0 = Uint2::expr(si.inst_id, si.prim_id);
            let stylized_li = if si.valid {
                scene.svm.dispatch_surface(
                    sample.si.surface,
                    color_pipeline,
                    sample.si,
                    **swl,
                    |closure| {
                        closure.apply_stylization(
                            path_depth,
                            sample.si,
                            -wi,
                            sample.li,
                            &self.eval_context(&scene, color_pipeline).1,
                            flags,
                        )
                    },
                )
            } else {
                sample.li
            };
            DirectLighting {
                irradiance: stylized_li,
                wi,
                pdf: sample.pdf,
                shadow_ray: shadow_ray.load(),
                valid: true.expr(),
            }
        } else {
            DirectLighting::invalid(color_pipeline)
        }
    }
}
impl Integrator for PathFiltering {
    fn render(
        &self,
        scene: Arc<Scene>,
        sampler_config: SamplerConfig,
        color_pipeline: ColorPipeline,
        film: &mut Film,
        session: &RenderSession,
    ) {
        let resolution = scene.camera.resolution();
        log::info!(
            "Resolution {}x{}\nconfig:{:#?}",
            resolution.x,
            resolution.y,
            &self.config
        );
        assert_eq!(resolution.x, film.resolution().x);
        assert_eq!(resolution.y, film.resolution().y);
        let n_pixels = resolution.x * resolution.y;
        let reuses_per_depth = (0..=self.config.max_depth)
            .map(|depth| {
                let n_direct_samples = self
                    .config
                    .reuse
                    .get(depth as usize)
                    .copied()
                    .unwrap_or(self.config.reuse.last().copied().unwrap());
                n_direct_samples
            })
            .collect::<Vec<_>>();
        let render_state = RenderState {
            rng_buf: init_pcg32_buffer_with_seed(
                self.device.clone(),
                n_pixels as usize * self.config.branch as usize,
                0,
            ),
            counter: self.device.create_buffer::<u32>(2),
            first_hit_radiance_buf: self.device.create_buffer::<Float3>(n_pixels as usize),
            path_state_buf: self
                .device
                .create_buffer::<PathState>(n_pixels as usize * self.config.branch as usize),
            reuses_per_depth: self.device.create_buffer_from_slice(&reuses_per_depth),
            neighbor_counter: self.device.create_buffer::<u64>(1),
        };
        render_state
            .first_hit_radiance_buf
            .fill(Float3::new(0.0, 0.0, 0.0));
        let kernel_reset_hash_grid = HashGrid::<u32>::reset_kernel(&self.device);
        let kernel_build_hash_grid = HashGrid::<u32>::build_kernel(&self.device);
        let kernel_trace = self
            .device
            .create_kernel_async::<fn(u32, Buffer<Vertex>, Buffer<Ray>)>(&track!(
                |depth: Expr<u32>, vertex_buf, ray_buf| {
                    let tid = dispatch_id().x;
                    let px_idx = tid % n_pixels;
                    self.kernel_mk_trace(
                        &render_state,
                        scene.clone(),
                        film,
                        color_pipeline,
                        px_idx,
                        depth,
                        self.config.max_depth.expr(),
                        vertex_buf,
                        ray_buf,
                    );
                }
            ));
        let kernel_propagate = self.device.create_kernel_async::<fn(
            Buffer<u32>,
            u32,
            Buffer<Vertex>,
            Buffer<IrradianceRecord>,
        )>(&track!(
            |ps_indices: BufferVar<u32>,
             path_depth: Expr<u32>,
             vertex_buf: BufferVar<Vertex>,
             record_buf| {
                self.kernel_propagate(
                    &render_state,
                    &scene,
                    ps_indices,
                    path_depth,
                    vertex_buf,
                    record_buf,
                );
            }
        ));
        let kernel_evaluate_smis = self.device.create_kernel_async::<fn(
            HashGrid<u32>,
            u32,
            Buffer<u32>,
            u32,
            Buffer<u32>,
            Buffer<Vertex>,
        )>(&track!(
            |hash_grid, ps_idx_offset, ps_indices, path_depth, neighbor_buf, vertex_buf| {
                self.kernel_evaluate_smis_weight(
                    &render_state,
                    &scene,
                    color_pipeline,
                    path_depth,
                    ps_idx_offset,
                    ps_indices,
                    &hash_grid,
                    &neighbor_buf,
                    vertex_buf,
                );
            }
        ));
        let kernel_filter = self.device.create_kernel_async::<fn(
            HashGrid<u32>,
            u32,
            Buffer<u32>,
            u32,
            Buffer<u32>,
            Buffer<Vertex>,
            Buffer<IrradianceRecord>,
        )>(&track!(
            |hash_grid: HashGridVar<u32>,
             ps_idx_offset: Expr<u32>,
             ps_indices: BufferVar<u32>,
             path_depth: Expr<u32>,
             neighbor_buf: BufferVar<u32>,
             vertex_buf: BufferVar<Vertex>,
             record_buf| {
                self.kernel_filter(
                    &render_state,
                    &scene,
                    color_pipeline,
                    ps_idx_offset,
                    ps_indices,
                    path_depth,
                    &hash_grid,
                    &neighbor_buf,
                    vertex_buf,
                    record_buf,
                );
            }
        ));
        let kernel_gather_points = self.device.create_kernel_async::<fn(
            u32,
            Buffer<u32>,
            Buffer<Float3>,
            Buffer<u32>,
            Buffer<Vertex>,
        )>(&track!(|path_depth: Expr<u32>,
                                                                  counter: BufferVar<u32>,
                                                                  pos: BufferVar<Float3>,
                                                                  indices: BufferVar<u32>,
                                                                  vertex_buf: BufferVar<
            Vertex,
        >| {
            self.kernel_gather_points(
                &render_state,
                scene.clone(),
                path_depth,
                counter,
                pos,
                indices,
                vertex_buf,
            );
        }));
        let kernel_splat = self
            .device
            .create_kernel_async::<fn(Buffer<Vertex>)>(&track!(|vertex_buf| {
                self.kernel_splat(&render_state, &scene, vertex_buf);
            }));
        let kernel_update_film =
            self.device
                .create_kernel_async::<fn(Buffer<Vertex>, f32)>(&track!(|vertex_buf, scale| {
                    self.kernel_update_film(
                        &render_state,
                        scene.clone(),
                        vertex_buf,
                        film,
                        color_pipeline,
                        scale,
                    );
                }));
        let kernel_reset_radiance_buf =
            self.device
                .create_kernel::<fn(Buffer<Float3>)>(&track!(|buf: BufferVar<Float3>| {
                    let tid = dispatch_id().x;
                    buf.write(tid, Float3::expr(0.0, 0.0, 0.0));
                }));
        let kernel_init_cluster = self
            .device
            .create_kernel_async::<fn(u32, Buffer<Vertex>, HashGrid<u32>, Buffer<u32>)>(&track!(
                |path_depth, vertex_buf, hash_grid, cluster_id_buf| {
                    self.kernel_init_cluster(
                        &render_state,
                        path_depth,
                        vertex_buf,
                        &hash_grid,
                        cluster_id_buf,
                    );
                }
            ));
        let kernel_cluster = self
            .device
            .create_kernel_async::<fn(Buffer<u32>, u32, Buffer<Vertex>, HashGrid<u32>)>(&track!(
                |counter, path_depth, vertex_buf, hash_grid| {
                    self.kernel_cluster(&render_state, path_depth, counter, vertex_buf, &hash_grid);
                }
            ));
        let kernel_reset_cluster_id_buf =
            self.device
                .create_kernel::<fn(Buffer<u32>)>(&track!(|cluster_id_buf: BufferVar<u32>| {
                    self.kernel_reset_cluster_id_buf(cluster_id_buf);
                }));
        kernel_reset_hash_grid.wait_for_compile();
        kernel_build_hash_grid.wait_for_compile();
        kernel_trace.wait_for_compile();
        kernel_filter.wait_for_compile();
        kernel_gather_points.wait_for_compile();
        kernel_splat.wait_for_compile();
        kernel_update_film.wait_for_compile();
        kernel_reset_radiance_buf.wait_for_compile();
        kernel_propagate.wait_for_compile();
        kernel_init_cluster.wait_for_compile();
        kernel_cluster.wait_for_compile();
        kernel_reset_cluster_id_buf.wait_for_compile();
        kernel_evaluate_smis.wait_for_compile();
        if session.dry_run {
            return;
        }
        log::info!("Rendering started");
        let mut acc_time = 0.0;
        let scene_aabb = if let Some(aabb) = &self.config.scene_bounds {
            (aabb[0].into(), aabb[1].into())
        } else {
            scene.meshes.aabb
        };
        let aabb_extent = scene_aabb.1 - scene_aabb.0;
        let aabb_size_avg = (aabb_extent.x + aabb_extent.y + aabb_extent.z) / 3.0;
        log::info!("Scene AABB: {:?}", scene_aabb);
        let pt_counter = self.device.create_buffer::<u32>(1);
        let pts = self
            .device
            .create_buffer::<Float3>(n_pixels as usize * self.config.branch as usize);
        let pt_indices = self
            .device
            .create_buffer::<u32>(n_pixels as usize * self.config.branch as usize);

        let build_hash_grid =
            |hash_grid: &HashGrid<u32>, d: u32, vertex_buf: &Buffer<Vertex>| -> u32 {
                pt_counter.copy_from(&[0]);
                kernel_gather_points.dispatch(
                    [n_pixels * self.config.branch, 1, 1],
                    &d,
                    &pt_counter,
                    &pts,
                    &pt_indices,
                    vertex_buf,
                );
                let pt_count = pt_counter.copy_to_vec()[0];
                kernel_reset_hash_grid
                    .dispatch([hash_grid.cell_head.len() as u32, 1, 1], hash_grid);

                {
                    let radius = aabb_size_avg
                        * 0.01
                        * self
                            .config
                            .radius_scale
                            .get(d as usize)
                            .copied()
                            .unwrap_or(self.config.radius_scale.last().copied().unwrap());
                    let res = (aabb_extent / radius).floor().as_uvec3();
                    log::info!("Hash grid resolution: {:?}, radius: {}", res, radius);
                    hash_grid.reset(
                        (scene_aabb.0.into(), scene_aabb.1.into()),
                        res.into(),
                        radius,
                    );
                }
                log::info!("Building hash grid for depth {} of {} points", d, pt_count);
                kernel_build_hash_grid.dispatch([pt_count, 1, 1], &pts, &pt_indices, hash_grid);
                pt_count
            };
        let neighbor_buf = self
            .device
            .create_buffer::<u32>(self.config.vram * 1024 * 1024 / 4 + 1);
        let vertex_buf = self
            .device
            .create_buffer::<Vertex>(n_pixels as usize * self.config.branch as usize);

        let mut per_depth_vertex_buffers: Vec<Vec<Vertex>> = (0..=self.config.max_depth)
            .map(|_| {
                let mut v = Vec::with_capacity(n_pixels as usize * self.config.branch as usize);
                unsafe {
                    v.set_len(n_pixels as usize * self.config.branch as usize);
                }
                v
            })
            .collect();
        let mut vram = 0;
        let mut peak_vram = 0;
        {
            vram += render_state.counter.len() * 4;
            vram += render_state.first_hit_radiance_buf.len() * std::mem::size_of::<Float3>();
            vram += render_state.path_state_buf.len() * std::mem::size_of::<PathState>();
            vram += vertex_buf.len() * std::mem::size_of::<Vertex>();
            vram += render_state.reuses_per_depth.len() * 4;
            vram += render_state.neighbor_counter.len() * 8;
            vram += neighbor_buf.len() * 4;
        };
        peak_vram = peak_vram.max(vram);
        log::info!("VRAM usage: {}MB", vram / 1024 / 1024);

        for r in 0..self.config.repeats {
            log::info!("Iteration {}/{}", r + 1, self.config.repeats);
            let update = || {
                if let Some(channel) = &session.display {
                    film.copy_to_rgba_image(channel.screen_tex(), false);
                    channel.notify_update();
                }
            };
            let tic = Instant::now();
            log::info!("Tracing paths");
            render_state.counter.copy_from(&[0, 0]);
            let ray_buf = self
                .device
                .create_buffer::<Ray>(n_pixels as usize * self.config.branch as usize);
            vram += ray_buf.len() * std::mem::size_of::<Ray>();
            peak_vram = peak_vram.max(vram);
            log::info!("VRAM usage: {}MB", vram / 1024 / 1024);
            for d in 0..=self.config.max_depth {
                kernel_trace.dispatch(
                    [n_pixels * self.config.branch, 1, 1],
                    &d,
                    &vertex_buf,
                    &ray_buf,
                );
                vertex_buf.copy_to(&mut per_depth_vertex_buffers[d as usize]);
            }
            vram -= ray_buf.len() * std::mem::size_of::<Ray>();
            std::mem::drop(ray_buf);
            let record_buf = self
                .device
                .create_buffer::<IrradianceRecord>(n_pixels as usize * self.config.branch as usize);

            vram += record_buf.len() * std::mem::size_of::<Ray>();
            let hash_grid = HashGrid::<u32>::new(
                &self.device,
                (scene_aabb.0.into(), scene_aabb.1.into()),
                Uint3::new(16, 16, 16),
                1.0,
                1024 * 1024 * 64,
                n_pixels as usize * self.config.branch as usize,
            );
            vram += hash_grid.mem_usage();
            peak_vram = peak_vram.max(vram);
            log::info!("VRAM usage: {}MB", vram / 1024 / 1024);
            for d in (0..=self.config.max_depth).rev() {
                let pt_count = build_hash_grid(&hash_grid, d, &vertex_buf);
                if self.config.cluster {
                    let counter = self.device.create_buffer::<u32>(1);
                    let mut it = 0;
                    let cluster_id_buf =
                        self.device.create_buffer::<u32>(hash_grid.cell_head.len());
                    loop {
                        it += 1;
                        counter.copy_from(&[0]);
                        kernel_reset_cluster_id_buf
                            .dispatch([cluster_id_buf.len() as u32, 1, 1], &cluster_id_buf);
                        kernel_init_cluster.dispatch(
                            [(n_pixels * self.config.branch as u32), 1, 1],
                            &d,
                            &vertex_buf,
                            &hash_grid,
                            &cluster_id_buf,
                        );
                        kernel_cluster.dispatch(
                            [n_pixels * self.config.branch, 1, 1],
                            &counter,
                            &d,
                            &vertex_buf,
                            &hash_grid,
                        );
                        let remaining = counter.copy_to_vec()[0];
                        if remaining == 0 {
                            break;
                        }
                        log::info!(
                            "Clustering depth {} iterations {}, remaining {}",
                            d,
                            it,
                            remaining
                        );
                    }
                }
                {
                    let max_reuse = reuses_per_depth[d as usize];

                    let batch_size =
                        (self.config.vram * 1024 * 1024 / (max_reuse as usize * 4)).max(1) as u32;
                    log::info!(
                        "Filtering indirect irradiance for depth {}; batch size:{}",
                        d,
                        batch_size
                    );
                    assert!(batch_size * max_reuse <= neighbor_buf.len() as u32);
                    // for i in (0..pt_count).step_by(batch_size as usize) {
                    //     let end = (i + batch_size).min(pt_count);
                    //     let n = end - i;
                    //     kernel_evaluate_smis.dispatch(
                    //         [n, 1, 1],
                    //         &hash_grid,
                    //         &i,
                    //         &pt_indices,
                    //         &d,
                    //         &neighbor_buf,
                    //         &vertex_buf,
                    //     );
                    // }
                    render_state.counter.copy_from(&[0, 0]);
                    render_state.neighbor_counter.copy_from(&[0]);

                    let progress = util::create_progess_bar(pt_count as usize, "Vertices");

                    for i in (0..pt_count).step_by(batch_size as usize) {
                        let end = (i + batch_size).min(pt_count);
                        let n = end - i;
                        kernel_filter.dispatch(
                            [n, 1, 1],
                            &hash_grid,
                            &i,
                            &pt_indices,
                            &d,
                            &neighbor_buf,
                            &vertex_buf,
                            &record_buf,
                        );
                        progress.inc(n as u64);
                    }
                    progress.finish();
                    if d > 0 {
                        vertex_buf.copy_from(&per_depth_vertex_buffers[d as usize - 1]);
                        kernel_propagate.dispatch(
                            [pt_count, 1, 1],
                            &pt_indices,
                            &d,
                            &vertex_buf,
                            &record_buf,
                        );
                    }
                }
                let total_neighbors = render_state.neighbor_counter.copy_to_vec()[0];
                let avg_neighbors = total_neighbors as f32 / (pt_count) as f32;
                log::info!(
                    "Average neighbors: {}, total neighbors: {}",
                    avg_neighbors,
                    total_neighbors
                );
            }
            vram -= record_buf.len() * std::mem::size_of::<Ray>();
            vram -= hash_grid.mem_usage();
            std::mem::drop(record_buf);
            std::mem::drop(hash_grid);

            log::info!("Splating to film");
            kernel_splat.dispatch([n_pixels * self.config.branch, 1, 1], &vertex_buf);
            kernel_update_film.dispatch(
                [n_pixels * self.config.branch, 1, 1],
                &vertex_buf,
                &(1.0 / self.config.branch as f32),
            );
            kernel_reset_radiance_buf
                .dispatch([n_pixels, 1, 1], &render_state.first_hit_radiance_buf);

            let toc = Instant::now();
            acc_time += toc.duration_since(tic).as_secs_f64();
            update();
        }
        log::info!("Peak VRAM usage: {}MB", peak_vram / 1024 / 1024);
        let ram =
            vertex_buf.len() * std::mem::size_of::<Vertex>() * (self.config.max_depth as usize + 1);
        log::info!("RAM usage: {}MB", ram / 1024 / 1024);
        log::info!("Rendering finished in {:.2}s", acc_time);
    }
}

pub fn render(
    device: Device,
    scene: Arc<Scene>,
    sampler: SamplerConfig,
    color_pipeline: ColorPipeline,
    film: &mut Film,
    config: &Config,
    options: &RenderSession,
) {
    let pt = PathFiltering::new(device.clone(), config.clone());
    pt.render(scene, sampler, color_pipeline, film, options);
}
