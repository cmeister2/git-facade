//! GPU-accelerated SHA1 hashing using wgpu compute shaders.
//!
//! This crate provides a GPU-based SHA1 implementation using WGSL compute shaders
//! dispatched via wgpu. It is designed for brute-force scenarios where millions of
//! SHA1 digests need to be computed in parallel (e.g. vanity hash prefix searching).

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Default number of SHA1 invocations per GPU dispatch.
pub const DEFAULT_BATCH_SIZE: u32 = 1 << 20;

/// Workgroup size matching the shader's `@workgroup_size(64)`.
const WORKGROUP_SIZE: u32 = 64;

/// Conservative per-dimension dispatch limit exposed by WebGPU adapters.
const MAX_DISPATCH_WORKGROUPS_PER_DIMENSION: u32 = 65_535;

/// Errors from GPU operations.
#[derive(Debug)]
pub enum GpuError {
    /// No GPU adapter was found.
    NoAdapter,
    /// Failed to request a device from the adapter.
    RequestDeviceFailed(String),
    /// A generic error.
    Other(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter => write!(f, "no GPU adapter found"),
            GpuError::RequestDeviceFailed(e) => write!(f, "failed to request GPU device: {}", e),
            GpuError::Other(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for GpuError {}

/// Uniform parameters passed to the shader.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Params {
    /// Byte offset where the 16-char hex salt starts in the template.
    salt_offset_bytes: u32,
    /// Byte length of the suffix stored in `template_data`.
    template_len_bytes: u32,
    /// Number of prefix bytes to match.
    prefix_len: u32,
    /// Number of invocations in this dispatch.
    batch_size: u32,
    /// Low 32 bits of the starting salt value.
    salt_base_lo: u32,
    /// High 32 bits of the starting salt value.
    salt_base_hi: u32,
    /// Total byte length of the original message before suffix extraction.
    total_len_bytes: u32,
    /// Number of workgroups dispatched in the x dimension.
    dispatch_groups_x: u32,
}

/// Result struct read back from the GPU.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct FindResultData {
    /// 1 if a match was found, 0 otherwise.
    found: u32,
    /// Low 32 bits of the matching salt.
    salt_lo: u32,
    /// High 32 bits of the matching salt.
    salt_hi: u32,
}

/// A precomputed GPU-ready template.
pub struct GpuTemplate {
    /// Suffix bytes packed as big-endian u32 words.
    words: Vec<u32>,
    /// Byte offset where the salt region starts within the suffix.
    salt_offset_bytes: u32,
    /// Byte count of the suffix stored in `words`.
    suffix_bytes: u32,
    /// Total byte count of the original template.
    total_bytes: u32,
    /// SHA1 state after compressing all full blocks before the suffix.
    prefix_state: [u32; 5],
}

impl GpuTemplate {
    /// Creates a GPU template from raw bytes and the byte offset of the salt.
    ///
    /// The salt is a 16-character hex region starting at `salt_offset`.
    pub fn from_bytes(data: &[u8], salt_offset: usize) -> Self {
        let suffix_start = (salt_offset / 64) * 64;
        let prefix_state = sha1_prefix_state(&data[..suffix_start]);
        let suffix = &data[suffix_start..];
        let words = bytes_to_be_words(suffix);
        Self {
            words,
            salt_offset_bytes: (salt_offset - suffix_start) as u32,
            suffix_bytes: suffix.len() as u32,
            total_bytes: data.len() as u32,
            prefix_state,
        }
    }
}

/// Result of a successful prefix search.
pub struct FindResult {
    /// The salt value that produced a matching digest.
    pub salt: u64,
}

/// GPU-accelerated SHA1 engine.
pub struct GpuSha1 {
    /// The wgpu device.
    device: wgpu::Device,
    /// The wgpu command queue.
    queue: wgpu::Queue,
    /// Compute pipeline for the `find_prefix` entry point.
    find_pipeline: wgpu::ComputePipeline,
    /// Compute pipeline for the `compute_digest` entry point (testing).
    digest_pipeline: wgpu::ComputePipeline,
    /// Bind group layout shared by both pipelines.
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuSha1 {
    /// Initializes the GPU device and compiles the SHA1 compute shader.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::NoAdapter`] if no GPU is available.
    /// Returns [`GpuError::RequestDeviceFailed`] if device creation fails.
    pub fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|_| GpuError::NoAdapter)?;

        let info = adapter.get_info();
        eprintln!(
            "wgpu-sha1: adapter={:?} backend={:?} type={:?}",
            info.name, info.backend, info.device_type
        );

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("wgpu-sha1"),
            ..Default::default()
        }))
        .map_err(|e| GpuError::RequestDeviceFailed(e.to_string()))?;

        let shader_source = include_str!("sha1.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sha1.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sha1_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sha1_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let find_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("find_prefix_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("find_prefix"),
            compilation_options: Default::default(),
            cache: None,
        });

        let digest_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute_digest_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("compute_digest"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            find_pipeline,
            digest_pipeline,
            bind_group_layout,
        })
    }

    /// Dispatches one batch of salt candidates and checks for a prefix match.
    ///
    /// Returns `Some(FindResult)` if a matching salt was found in this batch,
    /// or `None` if no match was found.
    ///
    /// # Errors
    ///
    /// Returns a [`GpuError`] if the GPU dispatch or readback fails.
    pub fn find_prefix(
        &self,
        template: &GpuTemplate,
        prefix: &[u8],
        salt_base: u64,
        batch_size: u32,
    ) -> Result<Option<FindResult>, GpuError> {
        let prefix_words = bytes_to_be_words(prefix);
        let mut prefix_and_state = prefix_words;
        prefix_and_state.extend_from_slice(&template.prefix_state);

        let params = Params {
            salt_offset_bytes: template.salt_offset_bytes,
            template_len_bytes: template.suffix_bytes,
            prefix_len: prefix.len() as u32,
            batch_size,
            salt_base_lo: salt_base as u32,
            salt_base_hi: (salt_base >> 32) as u32,
            total_len_bytes: template.total_bytes,
            dispatch_groups_x: dispatch_groups_x(batch_size),
        };

        let template_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("template"),
                contents: bytemuck::cast_slice(&template.words),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let prefix_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("prefix_and_state"),
                contents: bytemuck::cast_slice(&prefix_and_state),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let result_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("result"),
                contents: bytemuck::bytes_of(&FindResultData {
                    found: 0,
                    salt_lo: 0,
                    salt_hi: 0,
                }),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let debug_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("debug_digests"),
                contents: &[0u8; 4],
                usage: wgpu::BufferUsages::STORAGE,
            });

        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: std::mem::size_of::<FindResultData>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sha1_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: template_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: prefix_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: result_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: debug_buf.as_entire_binding(),
                },
            ],
        });

        let total_workgroups = batch_size.div_ceil(WORKGROUP_SIZE);
        let dispatch_groups_x = total_workgroups.min(MAX_DISPATCH_WORKGROUPS_PER_DIMENSION);
        let dispatch_groups_y = total_workgroups.div_ceil(dispatch_groups_x);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("find_prefix"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("find_prefix"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.find_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_groups_x, dispatch_groups_y, 1);
        }

        encoder.copy_buffer_to_buffer(
            &result_buf,
            0,
            &staging_buf,
            0,
            std::mem::size_of::<FindResultData>() as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        let result_data = read_buffer::<FindResultData>(&self.device, &staging_buf)?;

        if result_data.found != 0 {
            let salt = (result_data.salt_hi as u64) << 32 | result_data.salt_lo as u64;
            Ok(Some(FindResult { salt }))
        } else {
            Ok(None)
        }
    }

    /// Computes full SHA1 digests for specific salt values (for testing).
    ///
    /// Returns one 20-byte digest per salt.
    ///
    /// # Errors
    ///
    /// Returns a [`GpuError`] if the GPU dispatch or readback fails.
    pub fn compute_digests(
        &self,
        template: &GpuTemplate,
        salts: &[u64],
    ) -> Result<Vec<[u8; 20]>, GpuError> {
        let num_salts = salts.len();
        if num_salts == 0 {
            return Ok(Vec::new());
        }

        let salt_pairs: Vec<u32> = salts
            .iter()
            .flat_map(|&s| [s as u32, (s >> 32) as u32])
            .collect();
        let mut salts_and_state = salt_pairs;
        salts_and_state.extend_from_slice(&template.prefix_state);

        let params = Params {
            salt_offset_bytes: template.salt_offset_bytes,
            template_len_bytes: template.suffix_bytes,
            prefix_len: 0,
            batch_size: num_salts as u32,
            salt_base_lo: 0,
            salt_base_hi: 0,
            total_len_bytes: template.total_bytes,
            dispatch_groups_x: num_salts as u32,
        };

        let template_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("template"),
                contents: bytemuck::cast_slice(&template.words),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let salts_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("salts_and_state"),
                contents: bytemuck::cast_slice(&salts_and_state),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let result_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("result_dummy"),
                contents: bytemuck::bytes_of(&FindResultData {
                    found: 0,
                    salt_lo: 0,
                    salt_hi: 0,
                }),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let digest_size = (num_salts * 5 * 4) as u64;
        let debug_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug_digests"),
            size: digest_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: digest_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("digest_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: template_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: salts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: result_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: debug_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compute_digest"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_digest"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.digest_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_salts as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&debug_buf, 0, &staging_buf, 0, digest_size);

        self.queue.submit(Some(encoder.finish()));

        let raw_words = read_buffer_vec::<u32>(&self.device, &staging_buf, num_salts * 5)?;

        let digests = (0..num_salts)
            .map(|i| {
                let mut digest = [0u8; 20];
                for j in 0..5 {
                    let word = raw_words[i * 5 + j];
                    digest[j * 4..j * 4 + 4].copy_from_slice(&word.to_be_bytes());
                }
                digest
            })
            .collect();

        Ok(digests)
    }
}

/// Packs a byte slice into big-endian u32 words (zero-padded to 4-byte boundary).
fn bytes_to_be_words(data: &[u8]) -> Vec<u32> {
    let padded_len = data.len().div_ceil(4) * 4;
    let mut padded = vec![0u8; padded_len];
    padded[..data.len()].copy_from_slice(data);

    padded
        .chunks(4)
        .map(|chunk| u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Returns the x-dimension workgroup count used to flatten shader invocation ids.
fn dispatch_groups_x(batch_size: u32) -> u32 {
    let total_workgroups = batch_size.div_ceil(WORKGROUP_SIZE);
    total_workgroups.min(MAX_DISPATCH_WORKGROUPS_PER_DIMENSION)
}

/// Returns the initial SHA1 state words.
fn sha1_initial_state() -> [u32; 5] {
    [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0]
}

/// Computes the SHA1 state after all complete prefix blocks.
fn sha1_prefix_state(prefix_blocks: &[u8]) -> [u32; 5] {
    debug_assert_eq!(prefix_blocks.len() % 64, 0);
    let mut state = sha1_initial_state();
    for block in prefix_blocks.chunks_exact(64) {
        sha1_compress_cpu(&mut state, block);
    }
    state
}

/// Compresses one 64-byte SHA1 block into `state`.
fn sha1_compress_cpu(state: &mut [u32; 5], block: &[u8]) {
    debug_assert_eq!(block.len(), 64);

    let mut schedule = [0u32; 80];
    for (word, bytes) in schedule.iter_mut().take(16).zip(block.chunks_exact(4)) {
        *word = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    }
    for i in 16..80 {
        schedule[i] = (schedule[i - 3] ^ schedule[i - 8] ^ schedule[i - 14] ^ schedule[i - 16])
            .rotate_left(1);
    }

    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];

    for (i, &word) in schedule.iter().enumerate() {
        let (round_fn, constant) = match i {
            0..=19 => ((b & c) | ((!b) & d), 0x5A827999),
            20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
            40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
            _ => (b ^ c ^ d, 0xCA62C1D6),
        };
        let temp = a
            .rotate_left(5)
            .wrapping_add(round_fn)
            .wrapping_add(e)
            .wrapping_add(constant)
            .wrapping_add(word);
        e = d;
        d = c;
        c = b.rotate_left(30);
        b = a;
        a = temp;
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
}

/// Reads a single `T` from a mappable buffer.
fn read_buffer<T: Pod>(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<T, GpuError> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).ok();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    rx.recv()
        .map_err(|e| GpuError::Other(format!("buffer map recv failed: {}", e)))?
        .map_err(|e| GpuError::Other(format!("buffer map failed: {}", e)))?;

    let data = slice.get_mapped_range();
    let result: T = *bytemuck::from_bytes(&data[..std::mem::size_of::<T>()]);
    drop(data);
    buffer.unmap();
    Ok(result)
}

/// Reads a vec of `T` from a mappable buffer.
fn read_buffer_vec<T: Pod>(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<T>, GpuError> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).ok();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    rx.recv()
        .map_err(|e| GpuError::Other(format!("buffer map recv failed: {}", e)))?
        .map_err(|e| GpuError::Other(format!("buffer map failed: {}", e)))?;

    let data = slice.get_mapped_range();
    let byte_len = count * std::mem::size_of::<T>();
    let result: Vec<T> = bytemuck::cast_slice(&data[..byte_len]).to_vec();
    drop(data);
    buffer.unmap();
    Ok(result)
}
