pub use akari_render::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{color::ColorPipeline, film::*, gui::DisplayChannel, sampler::SamplerConfig, scene::*};

#[derive(Clone)]
pub struct RenderSession {
    pub save_intermediate: bool, // save intermediate results
    pub name: String,
    pub save_stats: bool, // save stats to {session}.json
    pub display: Option<DisplayChannel>,
    pub dry_run: bool,
}
impl Default for RenderSession {
    fn default() -> Self {
        Self {
            save_intermediate: false,
            name: String::from("default"),
            save_stats: false,
            display: None,
            dry_run: false,
        }
    }
}
#[derive(Default, Clone, Serialize, Deserialize)]
#[serde(crate = "serde")]
#[serde(default)]
pub struct IntermediateStats {
    pub path: String,
    pub time: f64,
    pub spp: u32,
}
#[derive(Default, Clone, Serialize, Deserialize)]
#[serde(crate = "serde")]
#[serde(default)]
pub struct RenderStats {
    pub intermediate: Vec<IntermediateStats>,
}
pub trait Integrator {
    fn render(
        &self,
        scene: Arc<Scene>,
        sampler: SamplerConfig,
        color_pipeline: ColorPipeline,
        film: &mut Film,
        options: &RenderSession,
    );
}

pub mod aov;
pub mod branching_pt;
pub mod nrc;
pub mod nrc_pt;
pub mod pt;
pub mod lm_path_filtering;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
#[serde(tag = "type")]
pub enum Method {
    #[serde(rename = "aov")]
    NormalVis(aov::Config),
    #[serde(rename = "pt")]
    PathTracer(pt::Config),
    #[serde(rename = "branching_pt")]
    BranchingPathTracer(branching_pt::Config),
    #[serde(rename = "nrc_pt")]
    NRCPathTracer(nrc_pt::Config),
    #[serde(rename = "pf")]
    LimitedMemoryPathFiltering(lm_path_filtering::Config),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
#[serde(default)]
pub struct FilmConfig {
    pub out: String,
    pub filter: PixelFilter,
    pub color: FilmColorRepr,
}
impl Default for FilmConfig {
    fn default() -> Self {
        Self {
            out: String::from("out.exr"),
            filter: PixelFilter::default(),
            color: FilmColorRepr::SRgb,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
pub struct RenderConfig {
    pub method: Method,
    #[serde(default)]
    pub color: ColorPipeline,
    #[serde(default = "SamplerConfig::default")]
    pub sampler: SamplerConfig,
    pub film: FilmConfig,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(crate = "serde")]
#[serde(untagged)]
pub enum RenderTask {
    Single(RenderConfig),
    Multi(Vec<RenderConfig>),
}

pub fn render(device: Device, scene: Arc<Scene>, task: &RenderTask, options: RenderSession) {
    fn render_single(
        device: Device,
        scene: &Arc<Scene>,
        config: &RenderConfig,
        options: &RenderSession,
    ) {
        let mut film = Film::new(
            device.clone(),
            scene.camera.resolution(),
            FilmColorRepr::SRgb,
            config.film.filter,
        );
        let output_image: Tex2d<Float4> = device.create_tex2d(
            PixelStorage::Float4,
            scene.camera.resolution().x,
            scene.camera.resolution().y,
            1,
        );
        log::info!("Rendering to {:?}", config.film);
        scene.svm.init_precompute_tables(config.color.color_repr);
        let tic = std::time::Instant::now();
        match &config.method {
            Method::PathTracer(c) => pt::render(
                device.clone(),
                scene.clone(),
                config.sampler,
                config.color,
                &mut film,
                &c,
                &options,
            ),
            Method::NormalVis(c) => aov::render(
                device.clone(),
                scene.clone(),
                config.sampler,
                config.color,
                &mut film,
                &c,
                &options,
            ),
            Method::BranchingPathTracer(c) => branching_pt::render(
                device.clone(),
                scene.clone(),
                config.sampler,
                config.color,
                &mut film,
                &c,
                &options,
            ),
            Method::NRCPathTracer(c) => nrc_pt::render(
                device.clone(),
                scene.clone(),
                config.sampler,
                config.color,
                &mut film,
                &c,
                &options,
            ),
            Method::LimitedMemoryPathFiltering(c) => lm_path_filtering::render(
                device.clone(),
                scene.clone(),
                config.sampler,
                config.color,
                &mut film,
                &c,
                &options,
            ),
        }
        let toc = std::time::Instant::now();
        log::info!("Completed in {:.1}ms", (toc - tic).as_secs_f64() * 1e3);
        if !options.dry_run {
            film.copy_to_rgba_image(&output_image, true);
            util::write_image(&output_image, &config.film.out);
        }
    }
    match task {
        RenderTask::Single(config) => render_single(device, &scene, config, &options),
        RenderTask::Multi(configs) => {
            let tic = std::time::Instant::now();
            log::info!("Rendering with {} methods", configs.len());
            for (i, config) in configs.iter().enumerate() {
                log::info!("Rendering task {}/{}", i + 1, configs.len());
                render_single(device.clone(), &scene, config, &options)
            }
            let toc = std::time::Instant::now();
            log::info!("Completed in {:.3}s", (toc - tic).as_secs_f64());
        }
    }
}
