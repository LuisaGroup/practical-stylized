use akari_render::rand::Rng;
use luisa::rtx::offset_ray_origin;
use nrc::TinyCudaNRC;
use rand::seq::SliceRandom;
use std::fs::File;
use std::io::BufWriter;
use std::sync::Arc;
use std::time::Instant;
use svm::surface::*;

use super::{Integrator, RenderSession};
use crate::pt::{BsdfDenoisingFeatures, DirectLighting, SurfaceHit, SurfaceHitComps};
use crate::{
    color::*, geometry::*, interaction::SurfaceInteraction, light::LightEvalContext, sampler::*, *,
};

#[derive(Clone)]
pub struct NRCPathTracer {
    pub device: Device,
    config: Config,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(crate = "serde")]
#[serde(default)]
pub struct Config {
    pub max_depth: u32,
    pub rr_depth: u32,

    pub training_iters: u32,
    pub baches_per_iter: u32,
    pub batch_size: u32,
    pub n_hash_grid_features: u32,
    pub n_hash_grid_levels: u32,
    pub n_hidden_layers: u32,
    pub n_hidden_units: u32,
    pub learning_rate: f32,
    pub decay_interval: u32,
    pub decay_rate: f32,
    pub repeats: u32,
    pub final_gather: Option<u32>,
    pub no_first_hit_stylization: bool,
    pub learn_stylized_radiance: bool,
    pub per_level_training: Option<u32>,
    pub scene_bounds: Option<Vec<[f32; 3]>>,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            max_depth: 7,
            rr_depth: 5,
            repeats: 1,
            training_iters: 512,
            baches_per_iter: 16,
            batch_size: 65536,
            n_hidden_layers: 5,
            n_hidden_units: 128,
            learning_rate: 1e-3,
            decay_interval: 50,
            decay_rate: 0.995,
            n_hash_grid_features: 2,
            final_gather: None,
            no_first_hit_stylization: false,
            learn_stylized_radiance: false,
            per_level_training: None,
            n_hash_grid_levels: 18,
            scene_bounds: None,
        }
    }
}
impl NRCPathTracer {
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
pub struct Vertex {
    pub wo: [f32; 3],
    pub hit: SurfaceHit,
    pub throughput: [f32; 3],
    pub nrc_idx: u32,
    pub le: [f32; 3],
    pub w_le: f32,
    pub mis: f32,
    pub direct: [f32; 3],
    pub indirect: [f32; 3],
    pub albedo: [f32; 3],
    pub roughness: f32,
    pub flags: StylizationFlags,
    pub new_flags: StylizationFlags,
}
#[derive(Clone, Copy, Debug, Value)]
#[luisa(crate = "luisa")]
#[repr(C)]
pub struct PathState {
    pub depth: u32,
    pub vertex_idx: u32,
    pub n_final_gathered: u32,
    pub splitting_depth: u32,
    pub px: Uint2,
    pub valid: bool,
}
struct RenderState {
    rng_buf: Buffer<Pcg32>,
    aabb: Buffer<Float3>,
    path_state_buf: Buffer<PathState>,
    vertex_buffer: Buffer<Vertex>,
    counter: Buffer<u32>,
    input_dim: usize,
    output_dim: usize,
    // trainig_px_idx_buf: Buffer<u32>,
    training_inputs: Buffer<f32>,
    training_targets: Buffer<f32>,
    inference_inputs: Buffer<f32>,
    inference_outputs: Buffer<f32>,
}
impl NRCPathTracer {
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
    fn kernel_mk_trace(
        &self,
        state: &RenderState,
        scene: Arc<Scene>,
        film: &Film,
        color_pipeline: ColorPipeline,
        px_idx: Expr<u32>,
        max_trace_depth: Expr<u32>,
        is_inference: bool,
        run_radiance_check: bool,
    ) {
        let tid = dispatch_id().x;

        let px_x = px_idx % scene.camera.resolution().x;
        let px_y = px_idx / scene.camera.resolution().x;
        let vertex_start_idx = tid * (self.config.max_depth + 1);

        let rng = state.rng_buf.read(tid).var();

        let mut sampler = IndependentSampler::from_pcg32(rng);
        let swl = SampledWavelengths::var_zeroed();
        let (ray, _) = scene.camera.generate_ray(
            &scene,
            film.filter(),
            Uint2::expr(px_x, px_y),
            &mut sampler,
            color_pipeline.color_repr,
            **swl,
        );
        let ray = ray.var();
        let ps = PathState::var_zeroed();
        *ps.depth = 0;
        *ps.px = Uint2::expr(px_x, px_y);
        *ps.splitting_depth = u32::MAX;
        let depth = 0u32.var();
        *ps.vertex_idx = vertex_start_idx;
        let prev_p = ray.o.var();
        let prev_ng = ray.d.var();
        let prev_bsdf_pdf = 0.0f32.var();
        let throughput = ColorVar::zero(color_pipeline.color_repr);
        let new_flags = StylizationFlags::var_zeroed();
        loop {
            let si = scene.intersect(**ray);
            if si.valid {
                *ps.valid = true;
                *ps.depth = depth;
                let wo = -ray.d;
                let (le, w_le, le_valid) = self.handle_surface_light(
                    &scene,
                    color_pipeline,
                    swl,
                    si,
                    **depth,
                    **ray,
                    Expr::<Float3>::from(**prev_ng),
                    **prev_bsdf_pdf,
                );

                let vertex_idx = vertex_start_idx + ps.depth;
                let v = Vertex::var_zeroed();
                if le_valid {
                    *v.mis = w_le;
                } else {
                    *v.mis = 1.0;
                }
                *v.hit = SurfaceHit::from_comps_expr(SurfaceHitComps {
                    inst_id: si.inst_id,
                    prim_id: si.prim_id,
                    bary: si.bary,
                });
                *v.flags = new_flags;
                *v.w_le = w_le;
                *v.le = Expr::<[f32; 3]>::from(le.as_rgb());
                *v.wo = Expr::<[f32; 3]>::from(-ray.d);
                *v.throughput = Expr::<[f32; 3]>::from(throughput.load().as_rgb());
                *v.nrc_idx = u32::MAX;
                *new_flags =
                    scene
                        .svm
                        .dispatch_surface(si.surface, color_pipeline, si, **swl, |closure| {
                            closure.update_stylization_flags(**depth, si, wo, **v.flags)
                        });
                *v.new_flags = new_flags;
                if depth >= max_trace_depth {
                    let (albedo, roughness) = scene.svm.dispatch_surface(
                        si.surface,
                        color_pipeline,
                        si,
                        **swl,
                        |closure| {
                            let albedo = closure.albedo(
                                wo,
                                **swl,
                                &self.eval_context(&scene, color_pipeline).1,
                            ) + closure.emission(
                                wo,
                                **swl,
                                &self.eval_context(&scene, color_pipeline).1,
                            );
                            let roughness = closure.avg_roughness(
                                wo,
                                **swl,
                                &self.eval_context(&scene, color_pipeline).1,
                            );
                            (albedo, roughness)
                        },
                    );
                    *v.albedo = Expr::<[f32; 3]>::from(albedo.as_rgb());
                    *v.roughness = roughness;
                    state.vertex_buffer.write(vertex_idx, v);
                    break;
                }

                *depth += 1;
                let u = sampler.next_3d();
                let dl = self.sample_light(
                    scene.clone(),
                    color_pipeline,
                    swl,
                    si,
                    u,
                    **depth,
                    **new_flags,
                );
                let (bs, denoising, direct) = self.sample_surface_and_shade_direct(
                    &scene,
                    color_pipeline,
                    swl,
                    si,
                    wo,
                    dl,
                    sampler.next_3d(),
                );
                *v.albedo = Expr::<[f32; 3]>::from(denoising.albedo.as_rgb());
                *v.roughness = denoising.roughness;
                let occluded = scene.occlude(dl.shadow_ray);
                if !occluded {
                    *v.direct = Expr::<[f32; 3]>::from(direct.as_rgb());
                }
                state.vertex_buffer.write(vertex_idx, v);
                if is_inference {
                    if (denoising.roughness > 0.05) & (ps.depth < max_trace_depth) {
                        *ps.splitting_depth = ps.depth;
                        break;
                    }
                }
                if bs.pdf <= 0.0 {
                    break;
                }
                throughput.store(bs.color / bs.pdf);
                *prev_bsdf_pdf = bs.pdf;
                *prev_ng = si.ng;
                *prev_p = si.p;
                let ro = offset_ray_origin(si.p, face_forward(si.ng, bs.wi));
                *ray = Ray::new_expr(
                    ro,
                    bs.wi,
                    0.0,
                    1e20,
                    Uint2::expr(si.inst_id, si.prim_id),
                    Uint2::expr(u32::MAX, u32::MAX),
                );
            } else {
                break_();
            }
        }
        if run_radiance_check {
            if ps.valid {
                let radiance = ColorVar::zero(color_pipeline.color_repr);
                let throughput = ColorVar::one(color_pipeline.color_repr);
                let i = (**ps.depth).var();
                loop {
                    let v = state.vertex_buffer.read(vertex_start_idx + i);
                    let le = Expr::<Float3>::from(v.le) * v.w_le;
                    let direct = Expr::<Float3>::from(v.direct);
                    let v_throughput = Expr::<Float3>::from(v.throughput);
                    // TODO: handle w_le
                    radiance.store(
                        radiance.load() * throughput.load()
                            + Color::from_flat(
                                color_pipeline.color_repr,
                                (le + direct).extend(0.0),
                            ),
                    );
                    throughput.store(Color::from_flat(
                        color_pipeline.color_repr,
                        v_throughput.extend(0.0),
                    ));
                    if i == 0 {
                        break;
                    }
                    *i -= 1;
                }
                film.add_sample(
                    Uint2::expr(px_x, px_y).cast_f32(),
                    &radiance.load(),
                    **swl,
                    1.0f32.expr(),
                );
            }
        }
        state.path_state_buf.write(tid, ps);
        state.rng_buf.write(tid, rng);
    }
    #[tracked(crate = "luisa")]
    fn kernel_final_gather_trace(
        &self,
        state: &RenderState,
        scene: Arc<Scene>,
        color_pipeline: ColorPipeline,
    ) {
        let tid = dispatch_id().x;
        let ps = state.path_state_buf.read(tid).var();
        let rng = state.rng_buf.read(tid).var();
        let sampler = IndependentSampler::from_pcg32(rng);
        if !ps.valid {
            return;
        }
        if ps.splitting_depth == u32::MAX {
            return;
        }
        let vertex = state
            .vertex_buffer
            .read(ps.vertex_idx + ps.splitting_depth)
            .var();
        let si = scene.surface_interaction(
            **vertex.hit.inst_id,
            **vertex.hit.prim_id,
            **vertex.hit.bary,
        );
        let wo = Expr::<Float3>::from(**vertex.wo);
        let new_flags = scene.svm.dispatch_surface(
            si.surface,
            color_pipeline,
            si,
            SampledWavelengths::expr_zeroed(),
            |closure| {
                closure.update_stylization_flags(**ps.splitting_depth, si, wo, **vertex.flags)
            },
        );
        let dl = self.sample_light(
            scene.clone(),
            color_pipeline,
            SampledWavelengths::var_zeroed(),
            si,
            sampler.next_3d(),
            1 + ps.splitting_depth,
            new_flags,
        );
        let (bs, _, direct) = self.sample_surface_and_shade_direct(
            &scene,
            color_pipeline,
            SampledWavelengths::var_zeroed(),
            si,
            wo,
            dl,
            sampler.next_3d(),
        );
        let occluded = scene.occlude(dl.shadow_ray);
        if !occluded {
            let old_direct = Expr::<Float3>::from(**vertex.direct);
            let direct = old_direct + direct.as_rgb();
            *vertex.direct = Expr::<[f32; 3]>::from(direct);
        }
        *ps.depth = ps.splitting_depth;
        if bs.pdf > 0.0 {
            let ro = offset_ray_origin(si.p, face_forward(si.ng, bs.wi));
            let ray = Ray::new_expr(
                ro,
                bs.wi,
                0.0,
                1e20,
                Uint2::expr(si.inst_id, si.prim_id),
                Uint2::expr(u32::MAX, u32::MAX),
            );
            let new_si = scene.intersect(ray);
            if new_si.valid {
                *ps.depth = 1 + ps.splitting_depth;
                let (_, mis, valid) = self.handle_surface_light(
                    &scene,
                    color_pipeline,
                    SampledWavelengths::var_zeroed(),
                    new_si,
                    **ps.depth,
                    ray,
                    new_si.ng,
                    bs.pdf,
                );
                let mis = if !valid { 1.0f32.expr() } else { mis };
                let (albedo, roughness) = scene.svm.dispatch_surface(
                    new_si.surface,
                    color_pipeline,
                    new_si,
                    SampledWavelengths::expr_zeroed(),
                    |closure| {
                        let wo = -ray.d;
                        let albedo = closure.albedo(
                            wo,
                            SampledWavelengths::expr_zeroed(),
                            &self.eval_context(&scene, color_pipeline).1,
                        ) + closure.emission(
                            wo,
                            SampledWavelengths::expr_zeroed(),
                            &self.eval_context(&scene, color_pipeline).1,
                        );
                        let roughness = closure.avg_roughness(
                            wo,
                            SampledWavelengths::expr_zeroed(),
                            &self.eval_context(&scene, color_pipeline).1,
                        );
                        (albedo, roughness)
                    },
                );
                let new_vertex = Vertex::var_zeroed();
                *new_vertex.wo = Expr::<[f32; 3]>::from(-ray.d);
                *new_vertex.nrc_idx = u32::MAX;
                *new_vertex.hit = SurfaceHit::from_comps_expr(SurfaceHitComps {
                    inst_id: new_si.inst_id,
                    prim_id: new_si.prim_id,
                    bary: new_si.bary,
                });
                *new_vertex.albedo = Expr::<[f32; 3]>::from(albedo.as_rgb());
                *new_vertex.roughness = roughness;
                *new_vertex.mis = mis;
                let throughput = bs.color / bs.pdf;
                *new_vertex.throughput = Expr::<[f32; 3]>::from(throughput.as_rgb());
                *new_vertex.flags = new_flags;
                *new_vertex.new_flags = scene.svm.dispatch_surface(
                    new_si.surface,
                    color_pipeline,
                    new_si,
                    SampledWavelengths::expr_zeroed(),
                    |closure| {
                        closure.update_stylization_flags(
                            1 + ps.splitting_depth,
                            new_si,
                            -ray.d,
                            **new_vertex.flags,
                        )
                    },
                );
                state
                    .vertex_buffer
                    .write(ps.vertex_idx + ps.splitting_depth + 1, new_vertex);
            }
        }
        state
            .vertex_buffer
            .write(ps.vertex_idx + ps.splitting_depth, vertex);
        state.path_state_buf.write(tid, ps);
        state.rng_buf.write(tid, rng);
    }
    #[tracked(crate = "luisa")]
    fn write_nrc_inputs(
        &self,
        state: &RenderState,
        p: Expr<Float3>,
        ns: Expr<Float3>,
        wo: Expr<Float3>,
        albedo: Expr<Float3>,
        roughness: Expr<f32>,
        depth: Expr<u32>,
        flags: Expr<StylizationFlags>,
        input_buf: &Buffer<f32>,
        idx: Expr<u32>,
    ) {
        let aabb_min = state.aabb.read(0);
        let aabb_max = state.aabb.read(1);
        let get_normalized_pos = |p: Expr<Float3>| (p - aabb_min) / (aabb_max - aabb_min);
        let p = get_normalized_pos(p);
        input_buf.write(idx + 0, p.x);
        input_buf.write(idx + 1, p.y);
        input_buf.write(idx + 2, p.z);
        // let (n_theta, n_phi) = xyz_to_spherical_normalized(ns);
        // let (wo_theta, wo_phi) = xyz_to_spherical_normalized(wo);
        // input_buf.write(idx + 3, n_theta);
        // input_buf.write(idx + 4, n_phi);
        // input_buf.write(idx + 5, wo_theta);
        // input_buf.write(idx + 6, wo_phi);
        input_buf.write(idx + 3, ns.x * 0.5 + 0.5);
        input_buf.write(idx + 4, ns.y * 0.5 + 0.5);
        input_buf.write(idx + 5, ns.z * 0.5 + 0.5);
        input_buf.write(idx + 6, wo.x * 0.5 + 0.5);
        input_buf.write(idx + 7, wo.y * 0.5 + 0.5);
        input_buf.write(idx + 8, wo.z * 0.5 + 0.5);
        let identity_idx = 9;
        input_buf.write(idx + identity_idx + 0, roughness); //1.0 - (-roughness).exp());
        input_buf.write(idx + identity_idx + 1, albedo.x);
        input_buf.write(idx + identity_idx + 2, albedo.y);
        input_buf.write(idx + identity_idx + 3, albedo.z);
        let one_hot_idx = identity_idx + 4;
        {
            input_buf.write(
                idx + one_hot_idx + 0,
                if flags.stylization_disabled {
                    1.0f32.expr()
                } else {
                    0.0f32.expr()
                },
            );
            input_buf.write(
                idx + one_hot_idx + 1,
                if flags.stylization_disabled {
                    0.0f32.expr()
                } else {
                    1.0f32.expr()
                },
            );
            let sty_idx = flags.get_style_index();
            for i in 0u32..4u32 {
                input_buf.write(
                    idx + one_hot_idx + 2 + i,
                    if i == sty_idx {
                        1.0f32.expr()
                    } else {
                        0.0f32.expr()
                    },
                );
            }
        }
        for i in 0u32.expr()..=self.config.max_depth.expr() {
            input_buf.write(
                idx + one_hot_idx + 6 + i,
                if i == depth {
                    1.0f32.expr()
                } else {
                    0.0f32.expr()
                },
            );
        }
        assert_eq!(
            self.config.max_depth as usize + 1 + one_hot_idx as usize + 6,
            state.input_dim
        );
    }
    #[tracked(crate = "luisa")]
    fn kernel_training_stage_nrc_inference_inputs(
        &self,
        state: &RenderState,
        scene: Arc<Scene>,
        start_depth: Expr<u32>,
        end_depth: Expr<u32>,
    ) {
        let tid = dispatch_id().x;
        let ps = state.path_state_buf.read(tid).var();
        if !ps.valid {
            return;
        }
        for i in start_depth..=end_depth {
            if i > ps.depth {
                break;
            }
            let v = state.vertex_buffer.read(ps.vertex_idx + i).var();
            let si = scene.surface_interaction(**v.hit.inst_id, **v.hit.prim_id, **v.hit.bary);
            *v.nrc_idx = state.counter.atomic_fetch_add(0, 1);
            let data_idx = v.nrc_idx * state.input_dim as u32;

            self.write_nrc_inputs(
                state,
                Expr::<Float3>::from(si.p),
                Expr::<Float3>::from(si.ns()),
                Expr::<Float3>::from(**v.wo),
                Expr::<Float3>::from(**v.albedo),
                **v.roughness,
                i,
                **v.new_flags,
                &state.inference_inputs,
                data_idx,
            );
            state.vertex_buffer.write(ps.vertex_idx + i, v);
        }
    }
    #[tracked(crate = "luisa")]
    fn kernel_training_stage_nrc_training_data(
        &self,
        state: &RenderState,
        scene: Arc<Scene>,
        color_pipeline: ColorPipeline,
        start_depth: Expr<u32>,
        end_depth: Expr<u32>,
    ) {
        let tid = dispatch_id().x;
        let ps = state.path_state_buf.read(tid).var();
        if !ps.valid {
            return;
        }
        for i in start_depth..=end_depth {
            if i > ps.depth {
                break;
            }
            let v = state.vertex_buffer.read(ps.vertex_idx + i).var();

            let exitant_radiance =
                (Expr::<Float3>::from(**v.le) + Expr::<Float3>::from(**v.direct)).var();
            // if (v.le.read(0)> 1.0) &(i==0) {
            //     device_log!("px: {}, le: {}", ps.px, v.le);
            // }
            // let exitant_radiance = (Expr::<Float3>::from(**v.le)).var();
            // let exitant_radiance = Expr::<Float3>::from(**v.albedo).var();

            if i + 1 <= ps.depth {
                let next_v = state.vertex_buffer.read(ps.vertex_idx + i + 1);
                let nrc_idx = next_v.nrc_idx;
                let next_si = scene.surface_interaction(
                    next_v.hit.inst_id,
                    next_v.hit.prim_id,
                    next_v.hit.bary,
                );
                // *exitant_radiance += Expr::<Float3>::from(next_v.throughput)
                //     * Expr::<Float3>::from(next_v.le)
                //     * next_v.w_le;
                let data_idx = nrc_idx * state.output_dim as u32;
                // device_log!("nrc_idx: {}, data_idx: {}", nrc_idx, data_idx);
                let nrc_output = Float3::var_zeroed();
                *nrc_output.x = state.inference_outputs.read(data_idx + 0);
                *nrc_output.y = state.inference_outputs.read(data_idx + 1);
                *nrc_output.z = state.inference_outputs.read(data_idx + 2);
                // apply stylization
                let stylized_nrc_output = if self.config.learn_stylized_radiance {
                    **nrc_output
                } else {
                    scene
                        .svm
                        .dispatch_surface(
                            next_si.surface,
                            color_pipeline,
                            next_si,
                            SampledWavelengths::expr_zeroed(),
                            |closure| {
                                closure.apply_stylization(
                                    i + 1,
                                    next_si,
                                    Expr::<Float3>::from(next_v.wo),
                                    Color::from_flat(
                                        color_pipeline.color_repr,
                                        nrc_output.extend(0.0),
                                    ),
                                    &self.eval_context(&scene, color_pipeline).1,
                                    next_v.flags,
                                )
                            },
                        )
                        .as_rgb()
                };
                *exitant_radiance +=
                    next_v.mis * stylized_nrc_output * Expr::<Float3>::from(next_v.throughput);
                if self.config.learn_stylized_radiance {
                    let si =
                        scene.surface_interaction(**v.hit.inst_id, **v.hit.prim_id, **v.hit.bary);
                    *exitant_radiance = scene
                        .svm
                        .dispatch_surface(
                            si.surface,
                            color_pipeline,
                            si,
                            SampledWavelengths::expr_zeroed(),
                            |closure| {
                                closure.apply_stylization(
                                    i,
                                    si,
                                    Expr::<Float3>::from(**v.wo),
                                    Color::from_flat(
                                        color_pipeline.color_repr,
                                        exitant_radiance.extend(0.0),
                                    ),
                                    &self.eval_context(&scene, color_pipeline).1,
                                    next_v.flags,
                                )
                            },
                        )
                        .as_rgb()
                }
            }
            let training_idx = state.counter.atomic_fetch_add(0, 1);
            let data_valid = exitant_radiance.is_finite().all();
            if !data_valid {
                continue;
            }
            {
                let si = scene.surface_interaction(**v.hit.inst_id, **v.hit.prim_id, **v.hit.bary);
                let data_idx = training_idx * state.input_dim as u32;

                self.write_nrc_inputs(
                    state,
                    Expr::<Float3>::from(si.p),
                    Expr::<Float3>::from(si.ns()),
                    Expr::<Float3>::from(**v.wo),
                    Expr::<Float3>::from(**v.albedo),
                    **v.roughness,
                    i,
                    **v.new_flags,
                    &state.training_inputs,
                    data_idx,
                );
            }
            {
                let data_idx = training_idx * state.output_dim as u32;
                state
                    .training_targets
                    .write(data_idx + 0, exitant_radiance.x);
                state
                    .training_targets
                    .write(data_idx + 1, exitant_radiance.y);
                state
                    .training_targets
                    .write(data_idx + 2, exitant_radiance.z);
                // state.training_targets.write(data_idx + 0, 0.0);
                // state.training_targets.write(data_idx + 1, 1.0);
                // state.training_targets.write(data_idx + 2, 0.0);
            }
        }
    }
    #[tracked(crate = "luisa")]
    fn kernel_inference_stage_nrc_inference_inputs(&self, state: &RenderState, scene: Arc<Scene>) {
        let tid = dispatch_id().x;
        let ps = state.path_state_buf.read(tid).var();
        if !ps.valid {
            return;
        }
        if (ps.splitting_depth == u32::MAX) & (ps.n_final_gathered > 0) {
            return;
        }
        if ps.depth == ps.splitting_depth {
            // final gather ray missed
            return;
        }
        let v = state.vertex_buffer.read(ps.vertex_idx + ps.depth).var();
        let si = scene.surface_interaction(**v.hit.inst_id, **v.hit.prim_id, **v.hit.bary);
        *v.nrc_idx = state.counter.atomic_fetch_add(0, 1);
        let data_idx = v.nrc_idx * state.input_dim as u32;
        // device_log!("{}", si.p);
        self.write_nrc_inputs(
            state,
            Expr::<Float3>::from(si.p),
            Expr::<Float3>::from(si.ns()),
            Expr::<Float3>::from(**v.wo),
            Expr::<Float3>::from(**v.albedo),
            **v.roughness,
            **ps.depth,
            **v.new_flags,
            &state.inference_inputs,
            data_idx,
        );
        state.vertex_buffer.write(ps.vertex_idx + ps.depth, v);
    }
    #[tracked(crate = "luisa")]
    fn kernel_inference_stage_gathering(
        &self,
        state: &RenderState,
        scene: Arc<Scene>,
        color_pipeline: ColorPipeline,
        film: &Film,
    ) {
        let tid = dispatch_id().x;

        let ps = state.path_state_buf.read(tid).var();
        if !ps.valid {
            return;
        }
        if (ps.splitting_depth == u32::MAX) & (ps.n_final_gathered > 0) {
            return;
        }
        if ps.depth == ps.splitting_depth {
            // final gather ray missed
            *ps.n_final_gathered += 1;
            state.path_state_buf.write(tid, ps);
            return;
        }
        let v = state.vertex_buffer.read(ps.vertex_idx + ps.depth);
        let nrc_idx = v.nrc_idx;
        lc_assert!(nrc_idx.ne(u32::MAX));
        let data_idx = nrc_idx * state.output_dim as u32;
        let nrc_output = Float3::var_zeroed();
        *nrc_output.x = state.inference_outputs.read(data_idx + 0);
        *nrc_output.y = state.inference_outputs.read(data_idx + 1);
        *nrc_output.z = state.inference_outputs.read(data_idx + 2);
        // if px_x == 500 && px_y == 500 {
        // device_log!("{} {}, nrc_output: {}", px_x, px_y, nrc_output);
        // }
        let si = scene.surface_interaction(v.hit.inst_id, v.hit.prim_id, v.hit.bary);
        let exitant_radiance = if self.config.learn_stylized_radiance {
            **nrc_output
        } else {
            scene
                .svm
                .dispatch_surface(
                    si.surface,
                    color_pipeline,
                    si,
                    SampledWavelengths::expr_zeroed(),
                    |closure| {
                        closure.apply_stylization(
                            **ps.depth,
                            si,
                            Expr::<Float3>::from(v.wo),
                            Color::from_flat(color_pipeline.color_repr, nrc_output.extend(0.0)),
                            &self.eval_context(&scene, color_pipeline).1,
                            v.flags,
                        )
                    },
                )
                .remove_nan()
                .as_rgb()
        };
        // let exitant_radiance = Expr::<Float3>::from(v.le);
        if (ps.depth == 0) & (ps.splitting_depth == u32::MAX) {
            // device_log!("px: {}, {}, {}", px_x, px_y, nrc_idx);
            if self.config.no_first_hit_stylization {
                film.add_sample(
                    ps.px.cast_f32(),
                    &Color::from_flat(color_pipeline.color_repr, nrc_output.extend(0.0)),
                    SampledWavelengths::expr_zeroed(),
                    1.0f32.expr(),
                );
            } else {
                film.add_sample(
                    ps.px.cast_f32(),
                    &Color::from_flat(color_pipeline.color_repr, exitant_radiance.extend(0.0)),
                    SampledWavelengths::expr_zeroed(),
                    1.0f32.expr(),
                );
            }
        } else {
            let prev_v = state.vertex_buffer.read(ps.vertex_idx + ps.depth - 1).var();
            let indirect = Expr::<Float3>::from(**prev_v.indirect).var();
            if indirect.is_finite().all() {
                *indirect += exitant_radiance * Expr::<Float3>::from(v.throughput) * v.mis;
                *prev_v.indirect = Expr::<[f32; 3]>::from(**indirect);
                state
                    .vertex_buffer
                    .write(ps.vertex_idx + ps.depth - 1, prev_v);
            }
        }
        *ps.n_final_gathered += 1;
        state.path_state_buf.write(tid, ps);
    }

    #[tracked(crate = "luisa")]
    fn kernel_inference_stage_propagate_and_splat(
        &self,
        state: &RenderState,
        scene: Arc<Scene>,
        color_pipeline: ColorPipeline,
        film: &Film,
    ) {
        let tid = dispatch_id().x;
        let ps = state.path_state_buf.read(tid);
        if !ps.valid {
            return;
        }
        if (ps.depth == 0) & (ps.splitting_depth == u32::MAX) {
            // already handled during gathering
            return;
        }
        let i = 0u32.var();
        if ps.splitting_depth == u32::MAX {
            *i = ps.depth;
        } else {
            *i = ps.splitting_depth;
        }
        let radiance = Float3::var_zeroed();
        let throughput = Float3::var_zeroed();
        let mis = 1.0f32.var();
        loop {
            let vertex = state.vertex_buffer.read(ps.vertex_idx + i);
            let direct = Expr::<Float3>::from(vertex.direct);
            let le = Expr::<Float3>::from(vertex.le);
            if i == ps.splitting_depth {
                let indirect = Expr::<Float3>::from(vertex.indirect);
                *radiance = le
                    + direct / (ps.n_final_gathered.as_f32() + 1.0)
                    + indirect / ps.n_final_gathered.as_f32();
            } else {
                *radiance = le + direct + throughput * radiance * mis;
            }
            let should_stylize = !self.config.learn_stylized_radiance
                & ((i > 0) | !self.config.no_first_hit_stylization);
            if should_stylize {
                let si = scene.surface_interaction(
                    vertex.hit.inst_id,
                    vertex.hit.prim_id,
                    vertex.hit.bary,
                );
                let stylized_radiance = scene.svm.dispatch_surface(
                    si.surface,
                    color_pipeline,
                    si,
                    SampledWavelengths::expr_zeroed(),
                    |closure| {
                        closure.apply_stylization(
                            **i,
                            si,
                            Expr::<Float3>::from(vertex.wo),
                            Color::from_flat(color_pipeline.color_repr, radiance.extend(0.0)),
                            &self.eval_context(&scene, color_pipeline).1,
                            vertex.flags,
                        )
                    },
                );
                *radiance = stylized_radiance.as_rgb();
            }
            *mis = vertex.mis;
            *throughput = Expr::<Float3>::from(vertex.throughput);
            // device_log!("throughput: {}", throughput);
            if i == 0 {
                break;
            }
            *i -= 1;
        }
        let radiance = Color::from_flat(color_pipeline.color_repr, radiance.extend(0.0));
        film.add_sample(
            ps.px.cast_f32(),
            &radiance,
            SampledWavelengths::expr_zeroed(),
            1.0f32.expr(),
        );
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
impl Integrator for NRCPathTracer {
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
        let training_pixels = ((self.config.baches_per_iter * self.config.batch_size)
            / (self.config.max_depth + 1))
            .clamp(1, n_pixels);
        let scene_aabb = if let Some(aabb) = &self.config.scene_bounds {
            (aabb[0].into(), aabb[1].into())
        } else {
            scene.meshes.aabb
        };
        log::info!("Scene bounds: {} {}", scene_aabb.0, scene_aabb.1);
        log::info!(
            "Training pixels: {}, {}% of all pixels",
            training_pixels,
            training_pixels as f32 / n_pixels as f32 * 100.0
        );
        let n_view_dir = 3;
        let n_albedo = 3;
        let n_roughness = 1;
        let n_normal = 3;
        let n_position = 3;
        let n_flags = 6;
        let n_input_dims = n_position
            + n_view_dir
            + n_normal
            + 1
            + self.config.max_depth
            + n_albedo
            + n_roughness
            + n_flags;
        let n_output_dims = 3;
        let network_type = match self.config.n_hidden_units {
            16 | 32 | 64 | 128 => "FullyFusedMLP",
            _ => "CutlassMLP",
        };
        let nrc_config = serde_json::json!({
            "loss":
                {"otype": "RelativeL2"}
            ,
            // "optimizer":{
            //     "otype": "EMA", // Component type.
            //     "decay": 0.9,  // The EMA's decay per step.
            //     "nested": {     // The nested optimizer.
            //         "otype":"Adam",
            //         "learning_rate": 5e-4, // Learning rate.
            //         "beta1": 0.9,          // Beta1 parameter of Adam.
            //         "beta2": 0.999,        // Beta2 parameter of Adam.
            //         "epsilon": 1e-8,
            //         "l2_reg": 0.0,
            //     }
            // // },
            // "optimizer":{
            //     "otype": "ExponentialDecay", // Component type.
            //     "decay_base": 0.92,           // The amount per decay step.
            //     "decay_start": 400,        // The training step at which
            //                                 // to start the decay.
            //     "decay_end": 8000,       // The training step at which
            //                                 // to end the decay.
            //     "decay_interval": 200,     // Training steps inbetween decay.
            //     "nested":{
            //         "otype": "EMA", // Component type.
            //         "decay": 0.9,  // The EMA's decay per step.
            //         "nested": {     // The nested optimizer.
            //             "otype":"Adam",
            //             "learning_rate": 1e-3, // Learning rate.
            //             "beta1": 0.9,          // Beta1 parameter of Adam.
            //             "beta2": 0.99,        // Beta2 parameter of Adam.
            //             "epsilon": 1e-8,
            //             "l2_reg": 0.0,
            //         }
            //     }
            // },
            "optimizer":{
                "otype": "ExponentialDecay", // Component type.
                "decay_base": self.config.decay_rate,           // The amount per decay step.
                "decay_start": 0,        // The training step at which
                                            // to start the decay.
                "decay_end": 1000000,       // The training step at which
                                            // to end the decay.
                "decay_interval": self.config.decay_interval,     // Training steps inbetween decay.

                "nested": {     // The nested optimizer.
                    "otype":"Adam",
                    "learning_rate": self.config.learning_rate, // Learning rate.
                    "beta1": 0.9,          // Beta1 parameter of Adam.
                    "beta2": 0.99,        // Beta2 parameter of Adam.
                    "epsilon": 1e-8,
                    "l2_reg": 0.0,
                }
            },
              // },
            //   "optimizer":{
            //     "otype": "ExponentialDecay", // Component type.
            //     "decay_base": 0.9,           // The amount per decay step.
            //     "decay_start": 0,        // The training step at which
            //                                 // to start the decay.
            //     "decay_end": 500,       // The training step at which
            //                                 // to end the decay.
            //     "decay_interval": 25,     // Training steps inbetween decay.
            //     "nested":{
            //         "otype": "Lookahead", // Component type.
            //         "alpha": 0.5,         // Fraction of lookahead distance to
            //                                 // traverse.
            //         "n_steps": 16,        // Nested optimizer steps for each
            //         "nested": {     // The nested optimizer.
            //             "otype":"Adam",
            //             "learning_rate": self.config.learning_rate, // Learning rate.
            //             "beta1": 0.9,          // Beta1 parameter of Adam.
            //             "beta2": 0.99,        // Beta2 parameter of Adam.
            //         }
            //     },
            // },
            // "optimizer": {
            //     "otype": "Lookahead", // Component type.
            //     "alpha": 0.5,         // Fraction of lookahead distance to
            //                             // traverse.
            //     "n_steps": 16,        // Nested optimizer steps for each
            //     "nested": {     // The nested optimizer.
            //         "otype":"Adam",
            //         "learning_rate": self.config.learning_rate, // Learning rate.
            //         "beta1": 0.9,          // Beta1 parameter of Adam.
            //         "beta2": 0.99,        // Beta2 parameter of Adam.
            //     }
            // },
            // "optimizer":{
            //     "otype": "Lookahead", // Component type.
            //     "alpha": 0.5,         // Fraction of lookahead distance to
            //                           // traverse.
            //     "n_steps": 16,        // Nested optimizer steps for each
            //     "nested": {     // The nested optimizer.
            //         "otype":"Adam",
            //         "learning_rate": 1e-3, // Learning rate.
            //         "beta1": 0.9,          // Beta1 parameter of Adam.
            //         "beta2": 0.99,        // Beta2 parameter of Adam.
            //         "epsilon": 1e-8,
            //         "l2_reg": 0.0,
            //     }
            // },
            // {
            //    "optimizer":{
            //     "otype": "ExponentialDecay", // Component type.
            //     "decay_base": 0.92,           // The amount per decay step.
            //     "decay_start": 400,        // The training step at which
            //                                 // to start the decay.
            //     "decay_end": 8000,       // The training step at which
            //                                 // to end the decay.
            //     "decay_interval": 400,     // Training steps inbetween decay. need to tune this!
            //     "nested": {     // The nested optimizer.
            //         "otype":"Adam",
            //         "learning_rate": 1e-3, // Learning rate.
            //         "beta1": 0.9,          // Beta1 parameter of Adam.
            //         "beta2": 0.99,        // Beta2 parameter of Adam.
            //         "epsilon": 1e-8,
            //         "l2_reg": 0.0,
            //     }
            // },
            //  "optimizer":{
            //     "otype": "ExponentialDecay", // Component type.
            //     "decay_base": 0.95,           // The amount per decay step.
            //     "decay_start": 0,        // The training step at which
            //                                 // to start the decay.
            //     "decay_end": 8000,       // The training step at which
            //                                 // to end the decay.
            //     "decay_interval": 25,     // Training steps inbetween decay.
            //     "nested": {     // The nested optimizer.
            //         "otype":"Adam",
            //         "learning_rate": 3e-4, // Learning rate.
            //         "beta1": 0.9,          // Beta1 parameter of Adam.
            //         "beta2": 0.99,        // Beta2 parameter of Adam.
            //         "epsilon": 1e-8,
            //         "l2_reg": 0.0,
            //     }
            // },
            // "optimizer":{

            //         "otype":"Adam",
            //         "learning_rate": 1e-4, // Learning rate.
            //         "beta1": 0.9,          // Beta1 parameter of Adam.
            //         "beta2": 0.99,        // Beta2 parameter of Adam.
            //         "epsilon": 1e-8,
            //         "l2_reg": 0.0,

            // },
            "ema_decay":0.95,
            "network": {
                "otype": network_type,
                "n_neurons": self.config.n_hidden_units,
                "n_hidden_layers": self.config.n_hidden_layers,
                "activation": "ReLU",
                "output_activation": "None"
            },
            "encoding":{
                "otype":"Composite",
                "nested":[
                    {
                        "n_dims_to_encode": 3,
                        "otype": "Grid",           // Component type.
                        "type": "Hash",            // Type of backing storage of the
                                                   // grids. Can be "Hash", "Tiled"
                                                   // or "Dense".
                        "n_levels": self.config.n_hash_grid_levels,            // Number of levels (resolutions)
                        "n_features_per_level": self.config.n_hash_grid_features,
                                                   // Dimensionality of feature vector
                                                   // stored in each level's entries.
                        "log2_hashmap_size": 19,   // If type is "Hash", is the base-2
                                                   // logarithm of the number of elements
                                                   // in each backing hash table.
                        "base_resolution": 16,     // The resolution of the coarsest le-
                                                   // vel is base_resolution^input_dims.
                        "per_level_scale": 2.0,    // The geometric growth factor, i.e.
                                                   // the factor by which the resolution
                                                   // of each grid is larger (per axis)
                                                   // than that of the preceding level.
                        "interpolation": "Linear"  // How to interpolate nearby grid
                                                   // lookups. Can be "Nearest", "Linear",
                                                   // or "Smoothstep" (for smooth deri-
                                                   // vatives).
                    },
                    {
                        "n_dims_to_encode": 7, // Non-linear appearance dims.
                        "otype": "OneBlob",
                        "n_bins": 4
                    },
                    {
                        // Number of remaining linear dims is automatically derived
                        "otype": "Identity"
                    }
                ]
            }
        });

        let render_state = RenderState {
            rng_buf: init_pcg32_buffer_with_seed(self.device.clone(), n_pixels as usize, 0),
            vertex_buffer: self
                .device
                .create_buffer::<Vertex>(n_pixels as usize * (self.config.max_depth as usize + 1)),
            aabb: self.device.create_buffer_from_slice::<Float3>(&[
                Float3::from(scene_aabb.0),
                Float3::from(scene_aabb.1),
            ]),
            path_state_buf: self.device.create_buffer::<PathState>(n_pixels as usize),
            counter: self.device.create_buffer::<u32>(1),
            input_dim: n_input_dims as usize,
            output_dim: n_output_dims,

            // trainig_px_idx_buf: self.device.create_buffer::<u32>(training_pixels as usize),
            training_inputs: self.device.create_buffer::<f32>(
                training_pixels as usize
                    * n_input_dims as usize
                    * (1 + self.config.max_depth as usize),
            ),
            training_targets: self.device.create_buffer::<f32>(
                training_pixels as usize
                    * n_output_dims as usize
                    * (1 + self.config.max_depth as usize),
            ),
            inference_inputs: self.device.create_buffer::<f32>(
                n_pixels as usize * n_input_dims as usize * (1 + self.config.max_depth as usize),
            ),
            inference_outputs: self.device.create_buffer::<f32>(
                n_pixels as usize * n_output_dims as usize * (1 + self.config.max_depth as usize),
            ),
        };

        let kernel_training_trace =
            self.device
                .create_kernel_async::<fn(u32)>(&track!(|max_trace_depth: Expr<u32>| {
                    let tid = dispatch_id().x;
                    // let px_idx = render_state.trainig_px_idx_buf.read(tid);
                    let rng = render_state.rng_buf.read(tid).var();
                    let px_idx = rng.gen_u32() % n_pixels;
                    render_state.rng_buf.write(tid, rng);
                    // device_log!("px_idx: {}, tid: {}", px_idx, tid);
                    self.kernel_mk_trace(
                        &render_state,
                        scene.clone(),
                        film,
                        color_pipeline,
                        px_idx,
                        max_trace_depth,
                        false,
                        false,
                    );
                }));
        let kernel_inference_trace =
            self.device
                .create_kernel_async::<fn(u32)>(&track!(|max_trace_depth: Expr<u32>| {
                    let tid = dispatch_id().x;
                    self.kernel_mk_trace(
                        &render_state,
                        scene.clone(),
                        film,
                        color_pipeline,
                        tid,
                        max_trace_depth,
                        true,
                        false,
                    );
                }));
        let kernel_final_gather_trace = self.device.create_kernel::<fn()>(&track!(|| {
            self.kernel_final_gather_trace(&render_state, scene.clone(), color_pipeline);
        }));
        let kernel_trace =
            self.device
                .create_kernel_async::<fn(u32)>(&track!(|max_trace_depth: Expr<u32>| {
                    let tid = dispatch_id().x;
                    self.kernel_mk_trace(
                        &render_state,
                        scene.clone(),
                        film,
                        color_pipeline,
                        tid,
                        max_trace_depth,
                        false,
                        true,
                    );
                }));
        let kernel_training_stage_nrc_inference_inputs =
            self.device.create_kernel_async::<fn(u32, u32)>(&track!(
                |start_depth: Expr<u32>, end_depth: Expr<u32>| {
                    self.kernel_training_stage_nrc_inference_inputs(
                        &render_state,
                        scene.clone(),
                        start_depth,
                        end_depth,
                    );
                }
            ));
        let kernel_training_stage_nrc_training_data =
            self.device.create_kernel_async::<fn(u32, u32)>(&track!(
                |start_depth: Expr<u32>, end_depth: Expr<u32>| {
                    self.kernel_training_stage_nrc_training_data(
                        &render_state,
                        scene.clone(),
                        color_pipeline,
                        start_depth,
                        end_depth,
                    );
                }
            ));
        let kernel_inference_stage_nrc_inference_inputs =
            self.device.create_kernel_async::<fn()>(&track!(|| {
                self.kernel_inference_stage_nrc_inference_inputs(&render_state, scene.clone());
            }));
        let kernel_inference_stage_gathering =
            self.device.create_kernel_async::<fn()>(&track!(|| {
                self.kernel_inference_stage_gathering(
                    &render_state,
                    scene.clone(),
                    color_pipeline,
                    film,
                );
            }));
        let kernel_inference_stage_propagate_and_splat =
            self.device.create_kernel_async::<fn()>(&track!(|| {
                self.kernel_inference_stage_propagate_and_splat(
                    &render_state,
                    scene.clone(),
                    color_pipeline,
                    film,
                );
            }));
        let kernel_clear_film = self.device.create_kernel::<fn()>(&track!(|| {
            let tid = dispatch_id().x;
            let buf = film.data();
            buf.write(tid, 0.0);
        }));
        kernel_training_trace.wait_for_compile();
        kernel_inference_trace.wait_for_compile();
        kernel_trace.wait_for_compile();
        kernel_training_stage_nrc_inference_inputs.wait_for_compile();
        kernel_training_stage_nrc_training_data.wait_for_compile();
        kernel_inference_stage_nrc_inference_inputs.wait_for_compile();
        kernel_inference_stage_gathering.wait_for_compile();
        kernel_inference_stage_propagate_and_splat.wait_for_compile();
        let nrc_config = nrc_config.to_string();

        let nrc = TinyCudaNRC::new(
            self.device.name() == "cuda",
            n_input_dims as usize,
            n_output_dims,
            self.config.batch_size as usize,
            &nrc_config,
        );
        log::info!(
            "Running NRC with host on {} with config: {}\nNRC input dims: {}, output dims: {}, {} parameters @ {:.4}MB",
            self.device.name(),
            nrc_config,
            n_input_dims,
            n_output_dims,
            nrc.parameter_count(),
            nrc.parameter_count() as f64 * 2.0 / 1024.0 / 1024.0
        );
        if session.dry_run {
            return;
        }
        let mut acc_time = 0.0;
        let progress =
            util::create_progess_bar(self.config.training_iters as usize, "training iter");
        let mut training_level = if self.config.per_level_training.is_some() {
            self.config.max_depth
        } else {
            0
        };
        let mut stats: RenderStats = Default::default();
        let output_image: Tex2d<Float4> = self.device.create_tex2d(
            PixelStorage::Float4,
            scene.camera.resolution().x,
            scene.camera.resolution().y,
            1,
        );
        let mut last_save_time = Instant::now();
        let mut save_cnt = 0;
        for it in 0..self.config.training_iters {
            if let Some(level_iters) = &self.config.per_level_training {
                if it > 0 && it % level_iters == 0 {
                    if training_level > 0 {
                        training_level -= 1;
                        log::info!("Training level: {}", training_level);
                    }
                }
            }

            // kernel_trace.dispatch([n_pixels, 1, 1], &self.config.max_depth);
            let t0 = Instant::now();
            {
                kernel_training_trace.dispatch([training_pixels, 1, 1], &self.config.max_depth);

                {
                    render_state.counter.copy_from(&[0]);
                    kernel_training_stage_nrc_inference_inputs.dispatch(
                        [training_pixels, 1, 1],
                        &training_level,
                        &self.config.max_depth,
                    );

                    let data_count = render_state.counter.copy_to_vec()[0];

                    nrc.train_inference(
                        data_count as usize,
                        &render_state.inference_inputs,
                        &render_state.inference_outputs,
                    );
                }
                {
                    render_state.counter.copy_from(&[0]);
                    kernel_training_stage_nrc_training_data.dispatch(
                        [training_pixels, 1, 1],
                        &training_level,
                        &self.config.max_depth,
                    );
                }
                {
                    let data_count = render_state.counter.copy_to_vec()[0];
                    nrc.train(
                        data_count as usize,
                        self.config.baches_per_iter as usize,
                        &render_state.training_inputs,
                        &render_state.training_targets,
                    );
                    // log::info!("Loss: {}", loss);
                }
                acc_time += t0.elapsed().as_secs_f64();
                let now = Instant::now();
                let mut stopping = false;
                if nrc.learning_rate() < 1e-5 {
                    log::info!("Learning rate too low, stopping training");
                    stopping = true;
                }
                let maybe_need_save = (now - last_save_time).as_secs_f64() > 1.0;
                if session.display.is_some()
                    || it + 1 == self.config.training_iters
                    || (session.save_intermediate && maybe_need_save)
                    || stopping
                {
                    render_state.counter.copy_from(&[0]);
                    kernel_inference_trace.dispatch([n_pixels, 1, 1], &0);
                    kernel_inference_stage_nrc_inference_inputs.dispatch([n_pixels as u32, 1, 1]);
                    let data_count = render_state.counter.copy_to_vec()[0];
                    nrc.inference(
                        data_count as usize,
                        &render_state.inference_inputs,
                        &render_state.inference_outputs,
                    );
                    kernel_clear_film.dispatch([film.data().len() as u32, 1, 1]);
                    render_state.counter.copy_from(&[0]);
                    kernel_inference_stage_gathering.dispatch([n_pixels as u32, 1, 1]);
                    if session.save_intermediate {
                        let now = Instant::now();
                        if maybe_need_save {
                            last_save_time = now;
                            film.copy_to_rgba_image(&output_image, true);
                            let path = format!("{}-{:03}.exr", session.name, save_cnt);
                            util::write_image(&output_image, &path);
                            stats.intermediate.push(IntermediateStats {
                                time: acc_time,
                                spp: save_cnt,
                                path,
                            });
                            save_cnt += 1;
                        }
                    }
                }
                if let Some(channel) = &session.display {
                    film.copy_to_rgba_image(channel.screen_tex(), false);
                    channel.notify_update();
                }
                if stopping {
                    break;
                }
            }
            progress.inc(1);
        }
        log::info!("Training finished in {:.2}s", acc_time);
        kernel_clear_film.dispatch([film.data().len() as u32, 1, 1]);
        if let Some(fg) = self.config.final_gather {
            for r in 0..self.config.repeats {
                log::info!("Final gathering pass {}/{}", r + 1, self.config.repeats);

                let progess = util::create_progess_bar(fg as usize, "spp");
                let t0 = Instant::now();
                kernel_inference_trace.dispatch([n_pixels, 1, 1], &self.config.max_depth);
                for _ in 0..fg {
                    render_state.counter.copy_from(&[0]);
                    kernel_final_gather_trace.dispatch([n_pixels as u32, 1, 1]);
                    kernel_inference_stage_nrc_inference_inputs.dispatch([n_pixels as u32, 1, 1]);
                    let data_count = render_state.counter.copy_to_vec()[0];
                    nrc.inference(
                        data_count as usize,
                        &render_state.inference_inputs,
                        &render_state.inference_outputs,
                    );
                    kernel_inference_stage_gathering.dispatch([n_pixels as u32, 1, 1]);
                    progess.inc(1);
                }
                kernel_inference_stage_propagate_and_splat.dispatch([n_pixels as u32, 1, 1]);
                acc_time += t0.elapsed().as_secs_f64();
                if let Some(channel) = &session.display {
                    film.copy_to_rgba_image(channel.screen_tex(), false);
                    channel.notify_update();
                }
                progess.finish();
            }
        } else {
            let progress = util::create_progess_bar(self.config.repeats as usize, "repeats");
            for _ in 0..self.config.repeats {
                let t0 = Instant::now();
                render_state.counter.copy_from(&[0]);
                kernel_inference_trace.dispatch([n_pixels, 1, 1], &0);
                kernel_inference_stage_nrc_inference_inputs.dispatch([n_pixels as u32, 1, 1]);
                let data_count = render_state.counter.copy_to_vec()[0];
                nrc.inference(
                    data_count as usize,
                    &render_state.inference_inputs,
                    &render_state.inference_outputs,
                );
                render_state.counter.copy_from(&[0]);
                kernel_inference_stage_gathering.dispatch([n_pixels as u32, 1, 1]);
                acc_time += t0.elapsed().as_secs_f64();
                if let Some(channel) = &session.display {
                    film.copy_to_rgba_image(channel.screen_tex(), false);
                    channel.notify_update();
                }
                progress.inc(1);
            }
            progress.finish();
        }
        if session.save_stats {
            let file = File::create(format!("{}.json", session.name)).unwrap();
            let json = serde_json::to_value(&stats).unwrap();
            let writer = BufWriter::new(file);
            serde_json::to_writer(writer, &json).unwrap();
        }
        std::mem::drop(nrc);
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
    let pt = NRCPathTracer::new(device.clone(), config.clone());
    pt.render(scene, sampler, color_pipeline, film, options);
}
