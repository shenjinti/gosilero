use active_call::media::vad::VADOption;
use anyhow::Result;
use libc::{c_char, c_float, c_int, size_t};
use std::ffi::CString;
use std::ptr;
pub struct VoiceActivityDetector {
    session: active_call::media::vad::TinySilero,
}

impl VoiceActivityDetector {
    pub fn new(config: VADOption) -> Result<Self> {
        let session = active_call::media::vad::TinySilero::new(config)?;
        Ok(VoiceActivityDetector { session })
    }

    pub fn predict(&mut self, samples: &[i16]) -> Result<f32> {
        let mut input_vec: Vec<f32> = samples.iter().map(|&x| x as f32 / 32768.0).collect();
        if input_vec.len() > 512 {
            input_vec.truncate(512);
        } else if input_vec.len() < 512 {
            input_vec.resize(512, 0.0);
        }
        self.predict_f32(&input_vec)
    }

    pub fn predict_f32(&mut self, samples: &[f32]) -> Result<f32> {
        if samples.len() != 512 {
            return Ok(0.0);
        }
        let probability = self.session.predict(samples);
        Ok(probability)
    }
}

// Create an opaque type for our GosileroVAD
#[repr(C)]
pub struct GosileroVAD {
    _private: [u8; 0], // Opaque type
}

// Internal struct to store the actual VAD data
struct GosileroVADInternal {
    vad: VoiceActivityDetector,
    last_error: Option<String>, // Store the last error locally
}

// Result type for the creation function
#[repr(C)]
pub struct GosileroVADCreateResult {
    vad: *mut GosileroVAD,
    success: bool,
}

/// Create a new GosileroVAD instance
///
/// # Arguments
///
/// * `sample_rate` - The sample rate to use (e.g. 8000, 16000)
/// * `chunk_size` - The chunk size to use (recommended: 256, 512, or 768 for 8000Hz; 512, 768, or 1024 for 16000Hz)
/// * `voice_threshold` - Threshold for speech detection (0.0 to 1.0, default: 0.5)
///
/// # Returns
///
/// A `GosileroVADCreateResult` containing the VAD pointer or null on failure
#[no_mangle]
pub extern "C" fn gosilero_vad_new(
    sample_rate: c_int,
    chunk_size: c_int,
    voice_threshold: c_float,
) -> GosileroVADCreateResult {
    if sample_rate <= 0 || chunk_size <= 0 {
        return GosileroVADCreateResult {
            vad: ptr::null_mut(),
            success: false,
        };
    }

    // Validate other parameters
    if voice_threshold < 0.0 || voice_threshold > 1.0 {
        return GosileroVADCreateResult {
            vad: ptr::null_mut(),
            success: false,
        };
    }

    // Convert from C types to Rust types
    let sample_rate = sample_rate as usize;
    let config = VADOption {
        samplerate: sample_rate as u32,
        voice_threshold: voice_threshold as f32,
        ..Default::default()
    };
    // Create the VAD
    match VoiceActivityDetector::new(config) {
        Ok(vad) => {
            let internal = Box::new(GosileroVADInternal {
                vad,
                last_error: None,
            });
            GosileroVADCreateResult {
                vad: Box::into_raw(internal) as *mut GosileroVAD,
                success: true,
            }
        }
        Err(_) => GosileroVADCreateResult {
            vad: ptr::null_mut(),
            success: false,
        },
    }
}

/// Create a new GosileroVAD instance with default parameters
///
/// # Arguments
///
/// * `sample_rate` - The sample rate to use (e.g. 8000, 16000)
/// * `chunk_size` - The chunk size to use (recommended: 256, 512, or 768 for 8000Hz; 512, 768, or 1024 for 16000Hz)
///
/// # Returns
///
/// A `GosileroVADCreateResult` containing the VAD pointer or null on failure
#[no_mangle]
pub extern "C" fn gosilero_vad_new_with_defaults(
    sample_rate: c_int,
    chunk_size: c_int,
) -> GosileroVADCreateResult {
    // Default values
    let voice_threshold: c_float = 0.6; // 0.6 threshold (increased from 0.5)

    gosilero_vad_new(sample_rate, chunk_size, voice_threshold)
}

/// Free a GosileroVAD instance
///
/// # Arguments
///
/// * `vad` - The GosileroVAD instance to free
#[no_mangle]
pub extern "C" fn gosilero_vad_free(vad: *mut GosileroVAD) {
    if !vad.is_null() {
        unsafe {
            // Convert the raw pointer back to a box and drop it
            let _ = Box::from_raw(vad as *mut GosileroVADInternal);
        }
    }
}

/// Get the last error message from a VAD instance
///
/// # Arguments
///
/// * `vad` - The GosileroVAD instance
///
/// # Returns
///
/// A pointer to a C string containing the error message, or NULL if no error
/// The returned string must be freed with gosilero_vad_free_string
#[no_mangle]
pub extern "C" fn gosilero_vad_get_error(vad: *mut GosileroVAD) -> *mut c_char {
    if vad.is_null() {
        return ptr::null_mut();
    }

    unsafe {
        match get_internal_error(vad) {
            Some(err) => match CString::new(err) {
                Ok(c_str) => c_str.into_raw(),
                Err(_) => ptr::null_mut(),
            },
            None => ptr::null_mut(),
        }
    }
}

/// Free a string returned by gosilero_vad_get_error
///
/// # Arguments
///
/// * `str` - The string to free
#[no_mangle]
pub extern "C" fn gosilero_vad_free_string(str: *mut c_char) {
    if !str.is_null() {
        unsafe {
            let _ = CString::from_raw(str);
        }
    }
}

// Helper function to get the internal VAD
unsafe fn get_internal_vad(vad: *mut GosileroVAD) -> &'static mut GosileroVADInternal {
    &mut *(vad as *mut GosileroVADInternal)
}

// Helper function to set error on internal VAD
unsafe fn set_internal_error(vad: *mut GosileroVAD, err: String) {
    let internal = get_internal_vad(vad);
    internal.last_error = Some(err);
}

// Helper function to get error from internal VAD
unsafe fn get_internal_error(vad: *mut GosileroVAD) -> Option<String> {
    let internal = get_internal_vad(vad);
    internal.last_error.clone()
}

/// Predict if a chunk of 32-bit float audio samples contains speech
///
/// # Arguments
///
/// * `vad` - The GosileroVAD instance
/// * `samples` - Pointer to the audio samples
/// * `num_samples` - Number of samples
///
/// # Returns
///
/// The probability of speech (0.0 to 1.0)
#[no_mangle]
pub extern "C" fn gosilero_vad_predict(
    vad: *mut GosileroVAD,
    samples: *const c_float,
    num_samples: size_t,
) -> c_float {
    if vad.is_null() || samples.is_null() || num_samples == 0 {
        unsafe { set_internal_error(vad, "Invalid arguments".to_string()) };
        return 0.0;
    }

    unsafe {
        // Convert the raw pointer to a slice
        let samples_slice = std::slice::from_raw_parts(samples, num_samples);

        // Predict speech
        match get_internal_vad(vad).vad.predict_f32(samples_slice) {
            Ok(probability) => probability as c_float,
            Err(_) => {
                set_internal_error(vad, "Prediction failed".to_string());
                0.0
            }
        }
    }
}

/// Predict if a chunk of 16-bit integer audio samples contains speech
///
/// # Arguments
///
/// * `vad` - The GosileroVAD instance
/// * `samples` - Pointer to the audio samples
/// * `num_samples` - Number of samples
///
/// # Returns
///
/// The probability of speech (0.0 to 1.0)
#[no_mangle]
pub extern "C" fn gosilero_vad_predict_i16(
    vad: *mut GosileroVAD,
    samples: *const i16,
    num_samples: size_t,
) -> c_float {
    if vad.is_null() || samples.is_null() || num_samples == 0 {
        unsafe { set_internal_error(vad, "Invalid arguments".to_string()) };
        return 0.0;
    }

    unsafe {
        // Convert the raw pointer to a slice
        let samples_slice = std::slice::from_raw_parts(samples, num_samples);

        // Clone the samples to avoid modifying the input data
        let samples_vec = samples_slice.to_vec();

        // Predict speech
        match get_internal_vad(vad).vad.predict(&samples_vec) {
            Ok(probability) => probability as c_float,
            Err(_) => {
                set_internal_error(vad, "Prediction failed".to_string());
                0.0
            }
        }
    }
}
