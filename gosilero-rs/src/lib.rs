use libc::{c_char, c_float, c_int, size_t};
use std::ffi::CString;
use std::ptr;
use voice_activity_detector::VoiceActivityDetector;

// Create an opaque type for our GosileroVAD
#[repr(C)]
pub struct GosileroVAD {
    _private: [u8; 0], // Opaque type
}

// Internal struct to store the actual VAD data
struct GosileroVADInternal {
    vad: VoiceActivityDetector,
    chunk_size: usize,
    voice_threshold: f32,       // 0.0 to 1.0
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
    let chunk_size = chunk_size as usize;

    // Create the VAD
    match VoiceActivityDetector::builder()
        .sample_rate(sample_rate as i64) // Convert to i64 as required by VAD builder
        .chunk_size(chunk_size)
        .build()
    {
        Ok(vad) => {
            let internal = Box::new(GosileroVADInternal {
                vad,
                chunk_size,
                voice_threshold: voice_threshold as f32,
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

        // Clone the samples to avoid modifying the input data
        let samples_vec = samples_slice.to_vec();

        // Predict speech
        get_internal_vad(vad).vad.predict(samples_vec) as c_float
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
        get_internal_vad(vad).vad.predict(samples_vec) as c_float
    }
}

/// Predict if a chunk of 8-bit integer audio samples contains speech
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
pub extern "C" fn gosilero_vad_predict_i8(
    vad: *mut GosileroVAD,
    samples: *const i8,
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
        get_internal_vad(vad).vad.predict(samples_vec) as c_float
    }
}

/// Process a stream of audio samples and return direct results without using callbacks
///
/// # Arguments
///
/// * `vad` - The GosileroVAD instance
/// * `samples` - Pointer to the audio samples
/// * `num_samples` - Number of samples
/// * `threshold_override` - Override the threshold for speech detection (use -1.0 to use the default)
/// * `output_is_speech` - Pointer to an array to store the speech detection results (1 for speech, 0 for non-speech)
/// * `output_positions` - Pointer to an array to store the positions (in samples) of the chunks
/// * `output_size` - The size of the output arrays
/// * `actual_output_size` - Pointer to store the actual number of results written
///
/// # Returns
///
/// 1 on success, 0 on failure
#[no_mangle]
pub extern "C" fn gosilero_vad_process_stream_direct(
    vad: *mut GosileroVAD,
    samples: *const i16,
    num_samples: size_t,
    threshold_override: c_float,
    output_is_speech: *mut u8,
    output_positions: *mut usize,
    output_size: size_t,
    actual_output_size: *mut size_t,
) -> c_int {
    if vad.is_null()
        || samples.is_null()
        || num_samples == 0
        || output_is_speech.is_null()
        || output_positions.is_null()
        || output_size == 0
        || actual_output_size.is_null()
    {
        unsafe { set_internal_error(vad, "Invalid arguments".to_string()) };
        return 0;
    }

    unsafe {
        // Get the internal VAD and its parameters
        let internal_vad = get_internal_vad(vad);
        let vad_ref = &mut internal_vad.vad;
        let chunk_size = internal_vad.chunk_size;

        // Use override threshold if provided, otherwise use the configured threshold
        let threshold = if threshold_override >= 0.0 && threshold_override <= 1.0 {
            threshold_override as f32
        } else {
            internal_vad.voice_threshold
        };

        // Convert the raw pointer to a slice
        let samples_slice = std::slice::from_raw_parts(samples, num_samples);

        // Prepare output arrays
        let output_is_speech_slice = std::slice::from_raw_parts_mut(output_is_speech, output_size);
        let output_positions_slice = std::slice::from_raw_parts_mut(output_positions, output_size);

        // Process chunks of samples
        let mut output_idx = 0;
        for i in (0..samples_slice.len()).step_by(chunk_size) {
            // Check if we've filled the output arrays
            if output_idx >= output_size {
                break;
            }

            let end = (i + chunk_size).min(samples_slice.len());
            let chunk = &samples_slice[i..end];

            // Skip chunks that are too small
            if chunk.len() < 100 {
                continue;
            }

            // Clone the chunk to avoid modifying the input
            let chunk_vec = chunk.to_vec();

            // Predict speech probability
            let probability = vad_ref.predict(chunk_vec);
            let is_speech = probability >= threshold;

            // Store the result
            output_is_speech_slice[output_idx] = if is_speech { 1 } else { 0 };
            output_positions_slice[output_idx] = i;
            output_idx += 1;
        }

        // Store the actual number of results written
        *actual_output_size = output_idx;
    }

    1
}
