use std::{f32::consts::FRAC_1_PI, fs::File, io::BufWriter, rc::Rc, sync::Arc, time::Instant};

use akari_render::svm::ShaderRef;
use luisa::rtx::offset_ray_origin;

use super::{Integrator, RenderSession};
use crate::{
    color::*,
    geometry::*,
    interaction::SurfaceInteraction,
    light::LightEvalContext,
    sampler::*,
    svm::surface::{diffuse::DiffuseBsdf, *},
    *,
};

#[derive(Aggregate, Copy, Clone)]
#[luisa(crate = "luisa")]
pub struct DirectLighting {
    pub irradiance: Color,
    pub wi: Expr<Float3>,
    pub pdf: Expr<f32>,
    pub shadow_ray: Expr<Ray>,
    pub valid: Expr<bool>,
}
impl DirectLighting {
    pub fn invalid(color_pipeline: ColorPipeline) -> Self {
        Self {
            irradiance: Color::zero(color_pipeline.color_repr),
            wi: Expr::<Float3>::zeroed(),
            pdf: 0.0f32.expr(),
            shadow_ray: Expr::<Ray>::zeroed(),
            valid: false.expr(),
        }
    }
}

#[derive(Copy, Clone, Value, Debug)]
#[repr(C, align(16))]
#[luisa(crate = "luisa")]
pub struct SurfaceHit {
    pub inst_id: u32,
    pub prim_id: u32,
    pub bary: Float2,
}

#[derive(Aggregate, Copy, Clone)]
#[luisa(crate = "luisa")]
pub struct BsdfDenoisingFeatures {
    pub albedo: Color,
    pub roughness: Expr<f32>,
}

#[derive(Clone)]
pub struct BranchingPathTracer {
    pub device: Device,
    config: Config,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(crate = "serde")]
#[serde(default)]
pub struct DebiasConfig {
    r: f32, // a geometry distribution,
    jackknife: bool,
}
impl Default for DebiasConfig {
    fn default() -> Self {
        Self {
            r: 0.65,
            jackknife: true,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(crate = "serde")]
#[serde(default)]
pub struct Config {
    pub max_depth: u32,
    pub vram: u32, // in MB
    pub use_nee: bool,
    pub rr_depth: u32,
    pub repeats: u32,
    pub inner_samples: Vec<u32>,
    pub debias: Option<DebiasConfig>,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            max_depth: 7,
            rr_depth: 5,
            vram: 8192,
            use_nee: true,
            repeats: 1,
            inner_samples: vec![256, 32, 8, 1],
            debias: None,
        }
    }
}
impl BranchingPathTracer {
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
    pub parent: u32,
    pub mis: f32,
    pub radiance: [f32; 3],
    pub debias_radiance_idx: u32,
    pub indirect_vertex: u32,
    pub slibing: u32,
    pub prev_p: [f32; 3],
    pub prev_ng: [f32; 3],
    pub prev_bsdf_pdf: f32,
    pub flags: StylizationFlags,
    pub n_inner_samples: u32,
    pub actual_branching_depths: u32,
    pub debiasing_samples: u32,
    pub indirect_branch_idx: u32,
}

struct RenderState {
    rng_buf: Buffer<Pcg32>,
    vertex_buffer: Buffer<Vertex>,
    n_inner_samples: Buffer<u32>,
    counter: Buffer<u32>,
    debiasing_direct_radiance: Buffer<Float3>,
    debiasing_indirect_radiance: Buffer<Float3>,
}
impl BranchingPathTracer {
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
    ) -> (BsdfSample, Color) {
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
            (sample, direct)
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
    fn kernel_propagate(
        &self,
        state: &RenderState,
        scene: Arc<Scene>,
        film: &Film,
        color_pipeline: ColorPipeline,
        vertex_id: Expr<u32>,
        path_depth: Expr<u32>,
    ) {
        let v = state.vertex_buffer.read(vertex_id).var();
        let head = v.indirect_vertex.var();
        let n_inner_sample = **v.n_inner_samples;
        let cnt = 0u32.var();
        let indirect_radiance = ColorVar::zero(color_pipeline.color_repr);
        if v.debias_radiance_idx != u32::MAX {
            for i in 0u32.expr()..**v.n_inner_samples {
                state
                    .debiasing_indirect_radiance
                    .write(v.debias_radiance_idx + i, Float3::splat_expr(0.0));
            }
        }
        while head != u32::MAX {
            let indirect_v = state.vertex_buffer.read(head).var();
            let li = Expr::<Float3>::from(**indirect_v.radiance);
            let throughput = Expr::<Float3>::from(**indirect_v.throughput);
            let l = throughput * li * indirect_v.mis;
            indirect_radiance.store(
                indirect_radiance.load()
                    + Color::from_flat(color_pipeline.color_repr, l.extend(0.0)),
            );
            if v.debias_radiance_idx != u32::MAX {
                state
                    .debiasing_indirect_radiance
                    .write(v.debias_radiance_idx + indirect_v.indirect_branch_idx, l);
            }
            *head = indirect_v.slibing;
            *cnt += 1;
        }

        let wo = Expr::<Float3>::from(**v.wo);
        let out_radiance = if self.config.debias.is_some() {
            let debias = self.config.debias.as_ref().unwrap();
            let si = scene.surface_interaction(**v.hit.inst_id, **v.hit.prim_id, **v.hit.bary);
            let swl = Var::<SampledWavelengths>::zeroed();
            let le = Color::from_flat(
                color_pipeline.color_repr,
                Expr::<Float3>::from(**v.radiance).extend(0.0),
            );
            let base_samples = v.n_inner_samples - v.debiasing_samples;
            if debias.jackknife {
                let ctx = self.eval_context(&scene, color_pipeline).1;
                let apply_style = |li| {
                    scene
                        .svm
                        .dispatch_surface(si.surface, color_pipeline, si, **swl, |closure| {
                            closure.apply_stylization(path_depth, si, wo, li, &ctx, **v.flags)
                        })
                };
                if v.debias_radiance_idx != u32::MAX {
                    let (base_estimator, full_estimator) = {
                        let direct_sum = ColorVar::zero(color_pipeline.color_repr);
                        let indirect_sum = ColorVar::zero(color_pipeline.color_repr);
                        for i in 0u32.expr()..base_samples {
                            let direct = Color::from_flat(
                                color_pipeline.color_repr,
                                state
                                    .debiasing_direct_radiance
                                    .read(v.debias_radiance_idx + i)
                                    .extend(0.0),
                            );
                            let indirect = Color::from_flat(
                                color_pipeline.color_repr,
                                state
                                    .debiasing_indirect_radiance
                                    .read(v.debias_radiance_idx + i)
                                    .extend(0.0),
                            );

                            direct_sum.store(direct_sum.load() + direct);
                            indirect_sum.store(indirect_sum.load() + indirect);
                        }

                        let base_estimator = apply_style(
                            (direct_sum.load() + indirect_sum.load()) / base_samples.as_f32() + le,
                        );
                        for i in base_samples..**v.n_inner_samples {
                            let direct = Color::from_flat(
                                color_pipeline.color_repr,
                                state
                                    .debiasing_direct_radiance
                                    .read(v.debias_radiance_idx + i)
                                    .extend(0.0),
                            );
                            let indirect = Color::from_flat(
                                color_pipeline.color_repr,
                                state
                                    .debiasing_indirect_radiance
                                    .read(v.debias_radiance_idx + i)
                                    .extend(0.0),
                            );

                            direct_sum.store(direct_sum.load() + direct);
                            indirect_sum.store(indirect_sum.load() + indirect);
                        }
                        let full_estimator = apply_style(
                            (direct_sum.load() + indirect_sum.load()) / v.n_inner_samples.as_f32()
                                + le,
                        );
                        (base_estimator, full_estimator)
                    };
                    let r = debias.r.expr();
                    let out_radiance = ColorVar::new(base_estimator);
                    let pmf = r * (1.0 - r).powi(v.debiasing_samples.as_i32() - 1);
                    for removal_idx in 0u32.expr()..**v.n_inner_samples {
                        let direct_sum = ColorVar::zero(color_pipeline.color_repr);
                        let indirect_sum = ColorVar::zero(color_pipeline.color_repr);
                        for i in 0u32.expr()..**v.n_inner_samples {
                            if i == removal_idx {
                                continue;
                            }
                            let direct = Color::from_flat(
                                color_pipeline.color_repr,
                                state
                                    .debiasing_direct_radiance
                                    .read(v.debias_radiance_idx + i)
                                    .extend(0.0),
                            );
                            let indirect = Color::from_flat(
                                color_pipeline.color_repr,
                                state
                                    .debiasing_indirect_radiance
                                    .read(v.debias_radiance_idx + i)
                                    .extend(0.0),
                            );
                            direct_sum.store(direct_sum.load() + direct);
                            indirect_sum.store(indirect_sum.load() + indirect);
                        }
                        let estimator = apply_style(
                            (direct_sum.load() + indirect_sum.load())
                                / (**v.n_inner_samples - 1).as_f32()
                                + le,
                        );
                        let delta = (full_estimator - estimator) / v.n_inner_samples.as_f32() / pmf;
                        out_radiance.store(out_radiance.load() + delta);
                    }
                    out_radiance.load()
                } else {
                    apply_style(le)
                }
            } else {
                let direct_sum = ColorVar::zero(color_pipeline.color_repr);
                let indirect_sum = ColorVar::zero(color_pipeline.color_repr);
                let ctx = self.eval_context(&scene, color_pipeline).1;
                let apply_style = |li| {
                    scene
                        .svm
                        .dispatch_surface(si.surface, color_pipeline, si, **swl, |closure| {
                            closure.apply_stylization(path_depth, si, wo, li, &ctx, **v.flags)
                        })
                };
                if v.debias_radiance_idx != u32::MAX {
                    for i in 0u32.expr()..base_samples {
                        let direct = Color::from_flat(
                            color_pipeline.color_repr,
                            state
                                .debiasing_direct_radiance
                                .read(v.debias_radiance_idx + i)
                                .extend(0.0),
                        );
                        let indirect = Color::from_flat(
                            color_pipeline.color_repr,
                            state
                                .debiasing_indirect_radiance
                                .read(v.debias_radiance_idx + i)
                                .extend(0.0),
                        );

                        direct_sum.store(direct_sum.load() + direct);
                        indirect_sum.store(indirect_sum.load() + indirect);
                    }
                    let base_estimator = apply_style(
                        (direct_sum.load() + indirect_sum.load()) / base_samples.as_f32() + le,
                    );
                    let prev_estimator = ColorVar::new(base_estimator);
                    let out_radiance = ColorVar::new(base_estimator);
                    let r = debias.r;
                    let pmf = 1.0f32.var();
                    for i in 0u32.expr()..**v.debiasing_samples {
                        let direct = Color::from_flat(
                            color_pipeline.color_repr,
                            state
                                .debiasing_direct_radiance
                                .read(v.debias_radiance_idx + base_samples + i)
                                .extend(0.0),
                        );
                        let indirect = Color::from_flat(
                            color_pipeline.color_repr,
                            state
                                .debiasing_indirect_radiance
                                .read(v.debias_radiance_idx + base_samples + i)
                                .extend(0.0),
                        );
                        direct_sum.store(direct_sum.load() + direct);
                        indirect_sum.store(indirect_sum.load() + indirect);
                        let new_estimator = apply_style(
                            (direct_sum.load() + indirect_sum.load())
                                / (base_samples + i + 1).as_f32()
                                + le,
                        );
                        let delta = (new_estimator - prev_estimator.load()) / **pmf;
                        out_radiance.store(out_radiance.load() + delta);
                        prev_estimator.store(new_estimator);
                        *pmf *= (1.0 - r);
                    }
                    out_radiance.load()
                } else {
                    apply_style(le)
                }
            }
        } else {
            let direct = Color::from_flat(
                color_pipeline.color_repr,
                Expr::<Float3>::from(**v.radiance).extend(0.0),
            );
            // device_log!("{} {}", path_depth, wo);
            let indirect = indirect_radiance.load() * (1.0 / n_inner_sample.as_f32());
            let li = direct + indirect;
            let ctx = self.eval_context(&scene, color_pipeline).1;
            let si = scene.surface_interaction(**v.hit.inst_id, **v.hit.prim_id, **v.hit.bary);
            let swl = Var::<SampledWavelengths>::zeroed();
            // device_log!(
            //     "{} {} {} {}",
            //     path_depth,
            //     vertex_id,
            //     direct.flatten(),
            //     indirect.flatten()
            // );
            let out_radiance =
                scene
                    .svm
                    .dispatch_surface(si.surface, color_pipeline, si, **swl, |closure| {
                        closure.apply_stylization(path_depth, si, wo, li, &ctx, **v.flags)
                    });
            out_radiance
        };
        *v.radiance = Expr::<[f32; 3]>::from(out_radiance.flatten().xyz());
        // device_log!("{} {} {}", path_depth, vertex_id, v.radiance);
        state.vertex_buffer.write(vertex_id, v);
        if path_depth == 0 {
            let px_idx = **v.parent;
            let px_x = px_idx % scene.camera.resolution().x;
            let px_y = px_idx / scene.camera.resolution().x;
            film.add_sample(
                Uint2::expr(px_x, px_y).cast_f32(),
                &out_radiance,
                SampledWavelengths::expr_zeroed(),
                1.0f32.expr(),
            );
        }
    }
    #[tracked(crate = "luisa")]
    fn kernel_init(
        &self,
        state: &RenderState,
        scene: Arc<Scene>,
        film: &Film,
        color_pipeline: ColorPipeline,
        start_px_idx: Expr<u32>,
    ) {
        let tid = dispatch_id().x;
        let px_idx = start_px_idx + tid;
        let px_x = px_idx % scene.camera.resolution().x;
        let px_y = px_idx / scene.camera.resolution().x;

        let rng = state.rng_buf.read(tid).var();
        let mut sampler = IndependentSampler::from_pcg32(rng);
        let (ray, _) = scene.camera.generate_ray(
            &scene,
            film.filter(),
            Uint2::expr(px_x, px_y),
            &mut sampler,
            color_pipeline.color_repr,
            SampledWavelengths::expr_zeroed(),
        );
        let si = scene.intersect(ray);
        if si.valid {
            let vertex_idx = state.counter.atomic_fetch_add(0, 1);
            let v = Vertex::var_zeroed();
            *v.debias_radiance_idx = u32::MAX;
            *v.indirect_vertex = u32::MAX;
            *v.slibing = u32::MAX;
            *v.parent = px_idx;
            *v.hit = SurfaceHit::from_comps_expr(SurfaceHitComps {
                inst_id: si.inst_id,
                prim_id: si.prim_id,
                bary: si.bary,
            });
            *v.wo = Expr::<[f32; 3]>::from(-ray.d);
            state.vertex_buffer.write(vertex_idx, v);
        }
        state.rng_buf.write(tid, rng);
    }
    #[tracked(crate = "luisa")]
    fn kernel_advance(
        &self,
        state: &RenderState,
        scene: Arc<Scene>,
        color_pipeline: ColorPipeline,
        vertex_id: Expr<u32>,
        sampler: &mut IndependentSampler,
        path_depth: Expr<u32>,
    ) {
        let v = state.vertex_buffer.read(vertex_id).var();

        let indirect_list_head = u32::MAX.var();
        let swl = Var::<SampledWavelengths>::zeroed();
        let si = scene.surface_interaction(**v.hit.inst_id, **v.hit.prim_id, **v.hit.bary);
        // device_log!("v.hit:{}", v.hit);
        let wo = Expr::<Float3>::from(**v.wo);
        // lc_assert!(wo.length().gt(0.0));
        let prev_p = Expr::<Float3>::from(**v.prev_p);
        let (le, w_le, valid) = self.handle_surface_light(
            &scene,
            color_pipeline,
            swl,
            si,
            path_depth,
            Ray::new_expr(
                prev_p,
                -wo,
                0.0,
                1e20,
                Uint2::expr_zeroed(),
                Uint2::expr_zeroed(),
            ),
            Expr::<Float3>::from(**v.prev_ng),
            **v.prev_bsdf_pdf,
        );
        let radiance = ColorVar::zero(color_pipeline.color_repr);
        radiance.store(radiance.load() + le); // TODO: MIS weights should not be applied here
        if !valid {
            *v.mis = 1.0;
        } else {
            *v.mis = w_le;
        }

        // device_log!("le: {} {}", le.flatten(), w_le);
        // device_log!("before {} {}", path_depth, radiance.load().flatten());
        let new_flags =
            scene
                .svm
                .dispatch_surface(si.surface, color_pipeline, si, **swl, |closure| {
                    closure.update_stylization_flags(path_depth, si, wo, **v.flags)
                });
        let roughness =
            scene
                .svm
                .dispatch_surface(si.surface, color_pipeline, si, **swl, |closure| {
                    closure.avg_roughness(wo, **swl, &self.eval_context(&scene, color_pipeline).1)
                });
        let need_branching = roughness > 0.01;
        let actual_branching_depths = v.actual_branching_depths.var();
        let n_inner_sample = if need_branching {
            let d = **actual_branching_depths;

            let n = state.n_inner_samples.read(d);
            *v.debiasing_samples = 0;
            if let Some(debias) = &self.config.debias {
                loop {
                    *v.debiasing_samples += 1;
                    let succ_prob = debias.r;
                    let succ = sampler.next_1d() < succ_prob;
                    if succ {
                        break;
                    }
                }
            }
            *actual_branching_depths += 1;
            // device_log!("{} {}", n, v.debiasing_samples);
            n + v.debiasing_samples
        } else {
            1u32.expr()
        };
        *v.n_inner_samples = n_inner_sample;
        if path_depth + 1 <= self.config.max_depth {
            if self.config.debias.is_some() {
                *v.debias_radiance_idx = state.counter.atomic_fetch_add(1, n_inner_sample);
                let valid = v.debias_radiance_idx + n_inner_sample
                    < state.debiasing_direct_radiance.len_expr_u32();
                if !valid {
                    device_log!(
                        "debiasing buffer overflow {} {} {} {}",
                        v.debias_radiance_idx,
                        n_inner_sample,
                        v.debiasing_samples,
                        state.debiasing_direct_radiance.len_expr_u32()
                    );
                }
            }
            for s_id in 0u32.expr()..n_inner_sample {
                let u = sampler.next_3d();
                let dl = self.sample_light(
                    scene.clone(),
                    color_pipeline,
                    swl,
                    si,
                    u,
                    path_depth + 1,
                    new_flags,
                );
                let (bs, direct) = self.sample_surface_and_shade_direct(
                    &scene,
                    color_pipeline,
                    swl,
                    si,
                    wo,
                    dl,
                    sampler.next_3d(),
                );
                let occluded = scene.occlude(dl.shadow_ray);
                if !occluded {
                    if self.config.debias.is_some() {
                        let idx = s_id + v.debias_radiance_idx;
                        state.debiasing_direct_radiance.write(idx, direct.as_rgb());
                    } else {
                        radiance.store(radiance.load() + direct * (1.0 / n_inner_sample.as_f32()));
                    }
                } else {
                    if self.config.debias.is_some() {
                        let idx = s_id + v.debias_radiance_idx;
                        state
                            .debiasing_direct_radiance
                            .write(idx, Float3::splat_expr(0.0));
                    }
                }
                let ro = offset_ray_origin(si.p, face_forward(si.ng, bs.wi));
                let ray = Ray::new_expr(
                    ro,
                    bs.wi,
                    0.0,
                    1e20,
                    Uint2::expr(si.inst_id, si.prim_id),
                    Uint2::expr(u32::MAX, u32::MAX),
                );
                if bs.pdf <= 0.0 {
                    continue;
                }

                let next_hit = scene.intersect(ray);
                if next_hit.valid {
                    let next_v = Vertex::var_zeroed();
                    *next_v.debias_radiance_idx = u32::MAX;
                    *next_v.slibing = u32::MAX;
                    *next_v.indirect_branch_idx = s_id;
                    let next_v_idx = state.counter.atomic_fetch_add(0, 1);
                    if indirect_list_head == u32::MAX {
                        *indirect_list_head = next_v_idx;
                    } else {
                        *next_v.slibing = indirect_list_head;
                        *indirect_list_head = next_v_idx;
                    }
                    *next_v.wo = Expr::<[f32; 3]>::from(-ray.d);
                    *next_v.hit = SurfaceHit::from_comps_expr(SurfaceHitComps {
                        inst_id: next_hit.inst_id,
                        prim_id: next_hit.prim_id,
                        bary: next_hit.bary,
                    });
                    *next_v.flags = new_flags;
                    *next_v.indirect_vertex = u32::MAX;
                    let throughput = bs.color / bs.pdf;
                    *next_v.throughput = Expr::<[f32; 3]>::from(throughput.flatten().xyz());
                    *next_v.parent = vertex_id;
                    *next_v.prev_bsdf_pdf = bs.pdf;
                    *next_v.prev_ng = Expr::<[f32; 3]>::from(si.ng);
                    *next_v.prev_p = Expr::<[f32; 3]>::from(si.p);
                    *next_v.actual_branching_depths = actual_branching_depths;
                    state.vertex_buffer.write(next_v_idx, next_v);
                }
            }
        }

        *v.indirect_vertex = **indirect_list_head;
        *v.radiance = Expr::<[f32; 3]>::from(radiance.load().flatten().xyz());
        // device_log!("{} {}", path_depth, v.radiance);
        state.vertex_buffer.write(vertex_id, v);
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
impl Integrator for BranchingPathTracer {
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
        let total_vertices_per_pixel: u32 = {
            let mut total = 1;
            let mut last = 1;
            for depth in 0..=self.config.max_depth {
                let n_inner_sample = self
                    .config
                    .inner_samples
                    .get(depth as usize)
                    .copied()
                    .unwrap_or(1);
                if let Some(debias) = &self.config.debias {
                    let succ_prob = debias.r;
                    let variance = 1.0 / succ_prob - 1.0;
                    let sigma = variance.sqrt();
                    let mean = (1.0 / succ_prob * (1.0 + 3.0 * sigma)).ceil() as u32;
                    dbg!(mean);
                    total += last * (n_inner_sample + mean);
                    last *= n_inner_sample + mean;
                } else {
                    total += last * n_inner_sample;
                    last *= n_inner_sample;
                }
            }
            total + 1 + self.config.max_depth
        };
        let pixels_per_pass = self.config.vram as usize * 1024 * 1024
            / (total_vertices_per_pixel as usize
                * (std::mem::size_of::<Vertex>() + std::mem::size_of::<Pcg32>()));
        if pixels_per_pass == 0 {
            log::error!("VRAM is not enough to render a single pixel");
            return;
        }
        log::info!("Pixels per pass: {}", pixels_per_pass);
        let n_inner_samples = (0..=self.config.max_depth)
            .map(|depth| {
                self.config
                    .inner_samples
                    .get(depth as usize)
                    .copied()
                    .unwrap_or(1)
            })
            .collect::<Vec<_>>();
        let render_state = RenderState {
            n_inner_samples: self.device.create_buffer_from_slice(&n_inner_samples),
            rng_buf: init_pcg32_buffer_with_seed(
                self.device.clone(),
                total_vertices_per_pixel as usize * pixels_per_pass as usize,
                0,
            ),
            vertex_buffer: self.device.create_buffer::<Vertex>(
                total_vertices_per_pixel as usize * pixels_per_pass as usize,
            ),
            debiasing_direct_radiance: self.device.create_buffer::<Float3>(
                if self.config.debias.is_some() {
                    total_vertices_per_pixel as usize * pixels_per_pass as usize
                } else {
                    1
                },
            ),
            debiasing_indirect_radiance: self.device.create_buffer::<Float3>(
                if self.config.debias.is_some() {
                    total_vertices_per_pixel as usize * pixels_per_pass as usize
                } else {
                    1
                },
            ),
            counter: self.device.create_buffer::<u32>(2),
        };
        let kernel_init = self
            .device
            .create_kernel::<fn(u32)>(&track!(|px_start_idx: Expr<u32>| {
                self.kernel_init(
                    &render_state,
                    scene.clone(),
                    film,
                    color_pipeline,
                    px_start_idx,
                );
            }));
        let kernel_advance = self.device.create_kernel::<fn(u32, u32)>(&track!(
            |vertex_start_idx: Expr<u32>, path_depth: Expr<u32>| {
                let tid = dispatch_id().x;
                let vertex_idx = vertex_start_idx + tid;
                let rng = render_state.rng_buf.read(vertex_idx).var();
                let mut sampler = IndependentSampler::from_pcg32(rng);
                self.kernel_advance(
                    &render_state,
                    scene.clone(),
                    color_pipeline,
                    vertex_idx,
                    &mut sampler,
                    path_depth,
                );
                render_state.rng_buf.write(vertex_idx, rng);
            }
        ));
        let kernel_propagate = self.device.create_kernel::<fn(u32, u32)>(&track!(
            |vertex_start_idx: Expr<u32>, path_depth: Expr<u32>| {
                let tid = dispatch_id().x;
                self.kernel_propagate(
                    &render_state,
                    scene.clone(),
                    film,
                    color_pipeline,
                    vertex_start_idx + tid,
                    path_depth,
                );
            }
        ));
        if session.dry_run {
            return;
        }
        log::info!("Rendering started");
        let mut acc_time = 0.0;
        for r in 0..self.config.repeats {
            let progress = util::create_progess_bar(
                n_pixels as usize,
                &format!("pixels of {}/{}repeats", r + 1, self.config.repeats),
            );

            let update = || {
                if let Some(channel) = &session.display {
                    film.copy_to_rgba_image(channel.screen_tex(), false);
                    channel.notify_update();
                }
            };
            let mut cur_px_idx = 0u32;
            while cur_px_idx < n_pixels {
                let cur_pass = (n_pixels - cur_px_idx).min(pixels_per_pass as u32);
                let tic = Instant::now();
                render_state.counter.copy_from(&[0u32, 0u32]);
                kernel_init.dispatch([cur_pass, 1, 1], &cur_px_idx);

                let mut vertex_start_indices = vec![0, render_state.counter.copy_to_vec()[0]];
                assert!(vertex_start_indices[1] <= cur_pass);
                for depth in 0..=self.config.max_depth {
                    let vertices = vertex_start_indices[depth as usize + 1]
                        - vertex_start_indices[depth as usize];
                    kernel_advance.dispatch(
                        [vertices, 1, 1],
                        &vertex_start_indices[depth as usize],
                        &depth,
                    );
                    vertex_start_indices.push(render_state.counter.copy_to_vec()[0]);
                }
                for depth in (0..=self.config.max_depth).rev() {
                    let vertex_start_idx = vertex_start_indices[depth as usize];
                    let vertices = vertex_start_indices[depth as usize + 1] - vertex_start_idx;
                    kernel_propagate.dispatch([vertices, 1, 1], &vertex_start_idx, &depth);
                }
                let toc = Instant::now();
                acc_time += toc.duration_since(tic).as_secs_f64();
                update();
                cur_px_idx += cur_pass;
                if session.save_intermediate {
                    todo!()
                }
                progress.inc(cur_pass as u64);
            }
            if session.save_stats {
                todo!()
            }
            progress.finish();
        }
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
    let pt = BranchingPathTracer::new(device.clone(), config.clone());
    pt.render(scene, sampler, color_pipeline, film, options);
}
