use crate::cpp_ext::tcnn;
use crate::*;
use std::{
    ffi::CString,
    io::{Read, Write},
};
pub struct NRCProcess {
    n_input_dim: usize,
    n_output_dim: usize,
    python_process: std::process::Child,
}
impl Drop for NRCProcess {
    fn drop(&mut self) {
        self.write_string("exit");
        // self.python_process.wait().unwrap();
    }
}
impl NRCProcess {
    pub fn connect(n_input_dim: usize, n_output_dim: usize) -> Self {
        // let python_output = File::create("nrc_output.txt").unwrap();
        let python_process = std::process::Command::new("python")
            .arg("-u")
            .arg("python/nonlinear_nrc.py")
            .stdout(std::process::Stdio::piped())
            .stdin(std::process::Stdio::piped())
            .spawn()
            .unwrap_or_else(|e| panic!("Failed to start NRC process: {}", e));

        Self {
            n_input_dim,
            n_output_dim,
            python_process,
        }
    }
    pub fn write_string(&mut self, s: &str) {
        self.write_data(s.as_bytes());
    }
    pub fn write_data<T: Copy>(&mut self, data: &[T]) {
        let length = data.len() as u32 * std::mem::size_of::<T>() as u32;
        // let t0 = Instant::now();
        let python_stdin = self.python_process.stdin.as_mut().unwrap();
        python_stdin.write_all(&length.to_le_bytes()).unwrap();
        python_stdin.flush().unwrap();
        python_stdin
            .write_all(unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, length as usize)
            })
            .unwrap();
        python_stdin.flush().unwrap();
        // println!(
        //     "write_data: {} bytes, {} ms",
        //     length,
        //     t0.elapsed().as_millis()
        // );
    }
    pub fn read_data<T: Copy>(&mut self, expected_length: usize) -> Vec<T> {
        let mut actual_length_buf = [0u8; 4];
        let python_stdout = self.python_process.stdout.as_mut().unwrap();
        python_stdout.read_exact(&mut actual_length_buf).unwrap();
        let actual_length = u32::from_le_bytes(actual_length_buf);
        assert_eq!(
            actual_length as usize,
            expected_length * std::mem::size_of::<T>()
        );
        let mut data = Vec::with_capacity(expected_length);
        python_stdout
            .read_exact(unsafe {
                std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, actual_length as usize)
            })
            .unwrap();
        unsafe {
            data.set_len(expected_length);
        }
        data
    }
    pub fn init(&mut self, config: &str) {
        self.write_string("init");
        self.write_string(config);
    }
    pub fn train(&mut self, input: &[f32], target: &[f32]) {
        let count = input.len() / self.n_input_dim;
        assert_eq!(count * self.n_input_dim, input.len());
        assert_eq!(count * self.n_output_dim, target.len());
        self.write_string("train");
        self.write_data::<f32>(input);
        self.write_data::<f32>(target);
    }
    pub fn inference(&mut self, input: &[f32]) -> Vec<f32> {
        let count = input.len() / self.n_input_dim;
        assert_eq!(count * self.n_input_dim, input.len());
        self.write_string("inference");
        self.write_data::<f32>(input);
        // read output
        self.read_data::<f32>(count * self.n_output_dim)
    }
}

pub struct TinyCudaNRC {
    inner: *mut tcnn::NRC,
}
impl TinyCudaNRC {
    pub fn new(
        is_input_cuda_buffer: bool,
        n_input_dims: usize,
        n_output_dims: usize,
        batch_size: usize,
        config: &str,
    ) -> Self {
        assert_eq!(batch_size % 256, 0);
        let config = CString::new(config).unwrap();
        let inner = unsafe {
            tcnn::akr_create_nrc(
                is_input_cuda_buffer,
                n_input_dims as u32,
                n_output_dims as u32,
                batch_size as u32,
                config.as_ptr() as *const i8,
            )
        };
        Self { inner }
    }
    pub fn train(&self, count: usize, n_iters: usize, input: &Buffer<f32>, target: &Buffer<f32>) {
        // dbg!(count);
        // let tic = std::time::Instant::now();
        unsafe {
            tcnn::akr_nrc_train(
                self.inner,
                count as u64,
                n_iters as u64,
                input.native_handle() as *const f32,
                target.native_handle() as *const f32,
            );
        }
        // let elapsed = tic.elapsed();
        // println!("train: {} ms", elapsed.as_millis());
    }
    pub fn train_inference(&self, count: usize, input: &Buffer<f32>, output: &Buffer<f32>) {
        // dbg!(count);
        // let tic = std::time::Instant::now();
        unsafe {
            tcnn::akr_nrc_train_inference(
                self.inner,
                count as u64,
                input.native_handle() as *const f32,
                output.native_handle() as *mut f32,
            )
        }
        // let elapsed = tic.elapsed();
        // println!("train_inference: {} ms", elapsed.as_millis());
    }
    pub fn inference(&self, count: usize, input: &Buffer<f32>, output: &Buffer<f32>) {
        // dbg!(count);
        unsafe {
            tcnn::akr_nrc_inference(
                self.inner,
                count as u64,
                input.native_handle() as *const f32,
                output.native_handle() as *mut f32,
            )
        }
    }
    pub fn set_learning_rate(&self, learning_rate: f32) {
        unsafe {
            tcnn::akr_nrc_set_learning_rate(self.inner, learning_rate);
        }
    }
    pub fn learning_rate(&self) -> f32 {
        unsafe { tcnn::akr_nrc_get_learning_rate(self.inner) }
    }
    pub fn parameter_count(&self) -> usize {
        unsafe { tcnn::akr_nrc_param_count(self.inner) as usize }
    }
    pub fn reset_optimizer(&self) {
        unsafe {
            tcnn::akr_nrc_reset_optimizer(self.inner);
        }
    }
}
impl Drop for TinyCudaNRC {
    fn drop(&mut self) {
        unsafe {
            tcnn::akr_destroy_nrc(self.inner);
        }
    }
}
