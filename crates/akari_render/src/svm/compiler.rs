use std::collections::HashMap;

use super::*;
use crate::{
    color::ColorSpaceId,
    load::{sampler_from_rgb_image_tex_node, ImageKey},
    util::ByteVecBuilder,
};
use akari_scenegraph::{shader::ShaderNode, NodeRef, ShaderGraph, ShaderKind};
use geometry::AffineTransform;
use scene_graph::ColorSpace;
#[derive(Clone, Copy)]
pub struct SvmCompileContext<'a> {
    pub images: &'a HashMap<(ImageKey, TextureSampler), usize>,
    pub graph: &'a ShaderGraph,
}
pub struct CompilerDriver {
    shaders: HashMap<ShaderBytecode, u32>,
    shader_data: Vec<u8>,
}

impl CompilerDriver {
    pub fn variant_count(&self) -> u32 {
        self.shaders.len() as u32
    }
    fn push_shader(&mut self, shader: CompiledShader) -> ShaderRef {
        let CompiledShader { bytecode, data } = shader;
        let kind = if self.shaders.contains_key(&bytecode) {
            self.shaders[&bytecode]
        } else {
            let kind = self.shaders.len() as u32;
            self.shaders.insert(bytecode, kind);
            kind
        };
        assert!(self.shader_data.len() % 16 == 0);
        let base_offset = self.shader_data.len();
        self.shader_data.extend_from_slice(&data);
        {
            // align to 16 bytes
            let padding = 16 - (self.shader_data.len() % 16);
            self.shader_data.extend_from_slice(&vec![0; padding]);
        }
        ShaderRef {
            shader_kind: kind,
            data_offset: base_offset.try_into().unwrap(),
        }
    }
    pub fn compile(&mut self, ctx: SvmCompileContext<'_>) -> ShaderRef {
        // dbg!(out);
        let shader = Compiler::compile(ctx);
        self.push_shader(shader)
    }
    pub fn new() -> Self {
        Self {
            shaders: HashMap::new(),
            shader_data: Vec::with_capacity(65536),
        }
    }
    pub fn upload(self, device: &Device) -> ShaderCollection {
        let Self {
            shader_data,
            shaders,
        } = self;

        let data = device.create_byte_buffer(shader_data.len());
        data.copy_from(&shader_data[..]);
        let kind_to_shader = shaders
            .iter()
            .map(|(k, v)| (v.clone(), (*k).clone()))
            .collect::<HashMap<_, _>>();
        ShaderCollection {
            shader_to_kind: shaders,
            kind_to_shader,
            shader_data: data,
        }
    }
}

struct Compiler<'a> {
    ctx: SvmCompileContext<'a>,
    env: HashMap<NodeRef<ShaderNode>, SvmNodeRef>,
    bytecode: Vec<SvmNode>,
    data: ByteVecBuilder,
}

impl<'a> Compiler<'a> {
    fn new(ctx: SvmCompileContext<'a>) -> Self {
        Self {
            ctx,
            env: HashMap::new(),
            bytecode: vec![],
            data: ByteVecBuilder::new(),
        }
    }
    fn push(&mut self, node: SvmNode) -> SvmNodeRef {
        let index = self.bytecode.len();
        self.bytecode.push(node);
        SvmNodeRef {
            index: index as u32,
        }
    }
    fn push_data<T: Value>(&mut self, value: T) -> SvmConst<T> {
        let offset = self.data.push(value);
        SvmConst {
            offset: offset as u32,
            marker: std::marker::PhantomData,
        }
    }
    fn push_data_slice<T: Value>(&mut self, value: &[T]) -> SvmConst<T> {
        let offset = self.data.push_slice(value);
        SvmConst {
            offset: offset as u32,
            marker: std::marker::PhantomData,
        }
    }
    fn compile_node(&mut self, node_id: &NodeRef<ShaderNode>) -> SvmNodeRef {
        let node = self.env.get(node_id);
        if let Some(node) = node {
            return *node;
        }
        self._compile_node(node_id);
        self.env[node_id]
    }
    fn _compile_node(&mut self, node_id: &NodeRef<ShaderNode>) {
        let graph = self.ctx.graph;
        let node = &graph.nodes[node_id];
        let node = match node {
            ShaderNode::Float { value } => {
                let node = self.push_data(*value);
                SvmNode::Float(node)
            }
            ShaderNode::Float3 { value } => {
                let node = self.push_data(Float3::new(value[0], value[1], value[2]));
                SvmNode::Float3(node)
            }
            ShaderNode::Rgb { value, colorspace } => {
                let data = self.push_data(Float3::new(value[0], value[1], value[2]));
                let data = self.push(SvmNode::Float3(data));
                SvmNode::RgbTex(SvmRgbTex {
                    rgb: data,
                    colorspace: ColorSpaceId::from_colorspace((*colorspace).into()),
                })
            }
            ShaderNode::Float4 { .. } => todo!(),
            ShaderNode::TexImage { image, uv } => {
                let colorspace = &image.colorspace;
                let data = &image.data;
                let sampler = sampler_from_rgb_image_tex_node(image);
                let key = ImageKey {
                    buffer: data.clone(),
                    format: image.format,
                    extension: image.extension,
                    interpolation: image.interpolation,
                    width: image.width,
                    height: image.height,
                    channels: image.channels,
                };
                let tex_idx = self.ctx.images[&(key, sampler)] as u32;
                let tex_idx = self.push_data(tex_idx);
                let uv = uv.as_ref().map(|uv| self.compile_node(uv));
                SvmNode::RgbImageTex(SvmRgbImageTex {
                    tex_idx,
                    colorspace: if *colorspace != ColorSpace::None {
                        ColorSpaceId::from_colorspace((*colorspace).into())
                    } else {
                        ColorSpaceId::NONE
                    },
                    uv,
                })
            }
            ShaderNode::PerlinNoise { .. } => {
                todo!()
            }
            ShaderNode::DiffuseBsdf { color } => {
                let color = self.compile_node(&color);
                SvmNode::DiffuseBsdf(SvmDiffuseBsdf { reflectance: color })
            }
            ShaderNode::SpectralUplift { rgb } => {
                let rgb = self.compile_node(&rgb);
                SvmNode::SpectralUplift(SvmSpectralUplift { rgb })
            }
            ShaderNode::PrincipledBsdf { bsdf } => {
                let base_color = self.compile_node(&bsdf.base_color);
                let metallic = self.compile_node(&bsdf.metallic);
                let roughness = self.compile_node(&bsdf.roughness);
                let ior = self.compile_node(&bsdf.ior);
                let alpha = self.compile_node(&bsdf.alpha);
                let normal = self.compile_node(&bsdf.normal);
                let subsurface_weight = self.compile_node(&bsdf.subsurface_weight);
                let subsurface_radius = self.compile_node(&bsdf.subsurface_radius);
                let subsurface_scale = self.compile_node(&bsdf.subsurface_scale);
                // let subsurface_ior = self.compile_node(&bsdf.subsurface_ior);
                let subsurface_anisotropy = self.compile_node(&bsdf.subsurface_anisotropy);
                let specular_ior_level = self.compile_node(&bsdf.specular_ior_level);
                let specular_tint = self.compile_node(&bsdf.specular_tint);
                let anisotropic = self.compile_node(&bsdf.anisotropic);
                let anisotropic_rotation = self.compile_node(&bsdf.anisotropic_rotation);
                let tangent = self.compile_node(&bsdf.tangent);
                let transmission_weight = self.compile_node(&bsdf.transmission_weight);
                let sheen_weight = self.compile_node(&bsdf.sheen_weight);
                let sheen_tint = self.compile_node(&bsdf.sheen_tint);
                let coat_weight = self.compile_node(&bsdf.coat_weight);
                let coat_roughness = self.compile_node(&bsdf.coat_roughness);
                let coat_ior = self.compile_node(&bsdf.coat_ior);
                let coat_tint = self.compile_node(&bsdf.coat_tint);
                let coat_normal = self.compile_node(&bsdf.coat_normal);
                let emission_color = self.compile_node(&bsdf.emission_color);
                let emission_strength = self.compile_node(&bsdf.emission_strength);
                SvmNode::PrincipledBsdf(SvmPrincipledBsdf {
                    base_color,
                    metallic,
                    roughness,
                    ior,
                    alpha,
                    normal,
                    subsurface_weight,
                    subsurface_radius,
                    subsurface_scale,
                    // subsurface_ior,
                    subsurface_anisotropy,
                    specular_ior_level,
                    specular_tint,
                    anisotropic,
                    anisotropic_rotation,
                    tangent,
                    transmission_weight,
                    sheen_weight,
                    sheen_tint,
                    coat_weight,
                    coat_roughness,
                    coat_ior,
                    coat_tint,
                    coat_normal,
                    emission_color,
                    emission_strength,
                })
            }
            ShaderNode::Emission {
                color: emission,
                strength,
            } => {
                let emission = self.compile_node(&emission);
                let strength = self.compile_node(&strength);
                SvmNode::Emission(SvmEmission {
                    color: emission,
                    strength,
                })
            }
            ShaderNode::GlassBsdf {
                color,
                ior,
                roughness,
            } => {
                let color = self.compile_node(color);
                let ior = self.compile_node(ior);
                let roughness = self.compile_node(&roughness);
                SvmNode::GlassBsdf(SvmGlassBsdf {
                    kr: color,
                    kt: color,
                    eta: ior,
                    roughness,
                })
            }
            ShaderNode::MixBsdf {
                first: _,
                second: _,
                factor: _,
            } => todo!(),
            ShaderNode::Extract { node, field } => {
                let node = self.compile_node(&node);
                let field = field.clone();
                SvmNode::ExtractField(SvmExtractField { node, field })
            }
            ShaderNode::Output { node } => match self.ctx.graph.kind {
                ShaderKind::Surface => {
                    let surface = self.compile_node(&node);
                    SvmNode::MaterialOutput(SvmMaterialOutput { surface })
                }
                _ => {
                    todo!()
                }
            },
            ShaderNode::TexCoords {} => SvmNode::TexCoords(SvmTexCoords {}),
            ShaderNode::Mapping {
                vector,
                mapping,
                location,
                rotation,
                scale,
            } => {
                let vector = self.compile_node(&vector);
                let location = self.compile_node(&location);
                let rotation = self.compile_node(&rotation);
                let scale = self.compile_node(&scale);
                SvmNode::Mapping(SvmMapping {
                    vector,
                    ty: *mapping,
                    location,
                    rotation,
                    scale,
                })
            }
            ShaderNode::NormalMap {
                normal,
                strength,
                space,
            } => {
                let normal = self.compile_node(&normal);
                let strength = self.compile_node(&strength);
                SvmNode::NormalMap(SvmNormalMap {
                    normal,
                    space: *space,
                    strength,
                })
            }
            ShaderNode::Checkerboard {
                vector,
                scale,
                color1,
                color2,
            } => {
                let vector = vector.as_ref().map(|v| self.compile_node(v));
                let scale = self.compile_node(&scale);
                let color1 = self.compile_node(&color1);
                let color2 = self.compile_node(&color2);
                SvmNode::CheckerBoard(SvmCheckerboardTex {
                    vector,
                    scale,
                    color1,
                    color2,
                })
            }
            ShaderNode::SeparateColor { mode, color } => {
                let color = self.compile_node(&color);
                SvmNode::SeparateColor(SeparateColor { color, mode: *mode })
            }
            ShaderNode::Toon {
                surface,
                color_band_width,
            } => {
                let surface = self.compile_node(&surface);
                let color_band_width = self.compile_node(&color_band_width);
                SvmNode::Toon(SvmToon {
                    surface,
                    color_band_width,
                })
            }
            ShaderNode::EightBit {
                surface,
                indirect_scale,
            } => {
                let surface = self.compile_node(&surface);
                let indirect_scale = self.compile_node(&indirect_scale);
                SvmNode::EightBit(Svm8Bit {
                    surface,
                    indirect_scale,
                })
            }
            ShaderNode::HalfTone {
                surface,
                trs,
                indirect_scale,
                radius,
                radius_scale,
                max_depth,
                grid_distance,
                dark_tone,
                light_tone,
                blend,
            } => {
                let surface = self.compile_node(&surface);
                let indirect_scale = self.compile_node(&indirect_scale);
                let t = AffineTransform::from_matrix(&trs.to_matrix(false));
                let t = self.push_data(t);
                let t = self.push(SvmNode::AffineTransform(t));
                let radius = self.compile_node(&radius);
                let radius_scale = self.compile_node(&radius_scale);
                let grid_distance = self.compile_node(&grid_distance);
                let max_depth = self.push_data(*max_depth);
                let max_depth = self.push(SvmNode::Uint(max_depth));
                let dark_tone = self.compile_node(&dark_tone);
                let light_tone = self.compile_node(&light_tone);
                let blend = self.compile_node(&blend);
                SvmNode::HalfTone(SvmHalfTone {
                    surface,
                    indirect_scale,
                    affine: t,
                    radius,
                    radius_scale,
                    max_depth,
                    grid_distance,
                    dark_tone,
                    light_tone,
                    blend,
                })
            }
            ShaderNode::SaturationAdjust {
                surface,
                power,
                indirect_scale,
                max_depth,
            } => {
                let surface = self.compile_node(&surface);
                let power = self.compile_node(&power);
                let indirect_scale = self.compile_node(&indirect_scale);
                let max_depth = self.push_data(*max_depth);
                let max_depth = self.push(SvmNode::Uint(max_depth));
                SvmNode::SaturationAdjust(SvmSaturationAdjust {
                    surface,
                    power,
                    indirect_scale,
                    max_depth,
                })
            }
            ShaderNode::SaturationAdjustRex { surface, blend } => {
                let surface = self.compile_node(&surface);
                let blend = self.compile_node(&blend);
                SvmNode::SaturationAdjustRex(SvmSaturationAdjustRex { surface, blend })
            }
            ShaderNode::ColorPalette {
                surface,
                palette,
                indirect_scale,
                gamma,
            } => {
                let surface = self.compile_node(&surface);
                let palette_len = self.push_data(palette.len() as u32);
                let indirect_scale = self.compile_node(&indirect_scale);
                let palette_data = palette
                    .iter()
                    .map(|x| {
                        // x is string
                        // x is rgb in hex format, such as 0xff0000
                        let x = u32::from_str_radix(x.trim_start_matches("0x"), 16).unwrap();
                        // dbg!(x);
                        let r = ((x >> 16) & 0xff) as f32 / 255.0;
                        let g = ((x >> 8) & 0xff) as f32 / 255.0;
                        let b = (x & 0xff) as f32 / 255.0;
                        // dbg!(r, g, b);
                        Float3::new(r, g, b)
                    })
                    .collect::<Vec<Float3>>();
                // dbg!(&palette_data);
                let gamma = self.compile_node(gamma);
                let palette_data = self.push_data_slice(&palette_data);
                SvmNode::ColorPalette(SvmColorPalette {
                    surface,
                    palette: palette_data,
                    length: palette_len,
                    indirect_scale,
                    gamma,
                })
            }
            ShaderNode::SetStylizationIndex { surface, idx } => {
                let surface = self.compile_node(&surface);
                let idx = self.push_data(*idx);
                SvmNode::SetStyleIdx(SvmSetStyleIdx { surface, idx })
            }
            ShaderNode::EnableStylizationIf {
                surface,
                idx,
                else_,
            } => {
                let surface = self.compile_node(&surface);
                let enable_idx = self.push_data(*idx);
                let else_ = if let Some(e) = else_ {
                    self.compile_node(e)
                } else {
                    self.push(SvmNode::Null(SvmNullSurface::new()))
                };
                SvmNode::EnableStylizationIf(SvmEnableIf {
                    surface,
                    idx: enable_idx,
                    else_,
                })
            }
            ShaderNode::Null {} => SvmNode::Null(SvmNullSurface::new()),
            _ => todo!(),
        };

        let node_ref = self.push(node);
        self.env.insert(node_id.clone(), node_ref);
    }

    fn compile(ctx: SvmCompileContext<'a>) -> CompiledShader {
        let mut compiler = Self::new(ctx);
        compiler.compile_node(&ctx.graph.output);
        // dbg!(&compiler.bytecode);
        CompiledShader {
            bytecode: ShaderBytecode::new(compiler.bytecode),
            data: compiler.data.finish(),
        }
    }
}
