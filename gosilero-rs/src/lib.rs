use libc::{c_char, c_float, c_int, c_void, size_t};
use once_cell::sync::Lazy;
use std::ffi::CString;
use std::ptr;
use std::sync::Mutex;
use voice_activity_detector::VoiceActivityDetector;

// Store the last error for error handling
static LAST_ERROR: Lazy<Mutex<Option<String>>> = Lazy::new(|| Mutex::new(None));

// Set the last error
fn set_error(err: String) {
    let mut error = LAST_ERROR.lock().unwrap();
    *error = Some(err);
}

// Create an opaque type for our GosileroVAD
#[repr(C)]
pub struct GosileroVAD {
    _private: [u8; 0], // Opaque type
}

// Internal struct to store the actual VAD data
struct GosileroVADInternal {
    vad: VoiceActivityDetector,
    sample_rate: usize,
    chunk_size: usize,
    silence_duration_threshold: u64, // milliseconds
    pre_speech_padding: u64,         // milliseconds
    post_speech_padding: u64,        // milliseconds
    voice_threshold: f32,            // 0.0 to 1.0
}

// Callback type for stream processing
#[allow(non_camel_case_types)]
pub type gosilero_vad_callback = unsafe extern "C" fn(
    is_speech: bool,
    samples: *mut i16,
    num_samples: size_t,
    user_data: *mut c_void,
);

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
/// * `silence_duration_threshold` - Minimum duration of silence to split speech segments (in ms, default: 500)
/// * `pre_speech_padding` - Padding before speech segment (in ms, default: 150)
/// * `post_speech_padding` - Padding after speech segment (in ms, default: 200)
/// * `voice_threshold` - Threshold for speech detection (0.0 to 1.0, default: 0.5)
///
/// # Returns
///
/// A `GosileroVADCreateResult` containing the VAD pointer or null on failure
#[no_mangle]
pub extern "C" fn gosilero_vad_new(
    sample_rate: c_int,
    chunk_size: c_int,
    silence_duration_threshold: u64,
    pre_speech_padding: u64,
    post_speech_padding: u64,
    voice_threshold: c_float,
) -> GosileroVADCreateResult {
    if sample_rate <= 0 || chunk_size <= 0 {
        set_error("Invalid sample rate or chunk size".to_string());
        return GosileroVADCreateResult {
            vad: ptr::null_mut(),
            success: false,
        };
    }

    // Validate other parameters
    if voice_threshold < 0.0 || voice_threshold > 1.0 {
        set_error("Voice threshold must be between 0.0 and 1.0".to_string());
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
                sample_rate,
                chunk_size,
                silence_duration_threshold,
                pre_speech_padding,
                post_speech_padding,
                voice_threshold: voice_threshold as f32,
            });
            GosileroVADCreateResult {
                vad: Box::into_raw(internal) as *mut GosileroVAD,
                success: true,
            }
        }
        Err(err) => {
            set_error(format!("Failed to create VAD: {}", err));
            GosileroVADCreateResult {
                vad: ptr::null_mut(),
                success: false,
            }
        }
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
    let silence_duration_threshold: u64 = 200; // 200ms (reduced from 500ms)
    let pre_speech_padding: u64 = 50; // 50ms (reduced from 150ms)
    let post_speech_padding: u64 = 50; // 50ms (reduced from 200ms)
    let voice_threshold: c_float = 0.6; // 0.6 threshold (increased from 0.5)

    gosilero_vad_new(
        sample_rate,
        chunk_size,
        silence_duration_threshold,
        pre_speech_padding,
        post_speech_padding,
        voice_threshold,
    )
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

/// Get the last error message
///
/// # Returns
///
/// A pointer to a C string containing the error message, or NULL if no error
/// The returned string must be freed with gosilero_vad_free_string
#[no_mangle]
pub extern "C" fn gosilero_vad_last_error() -> *mut c_char {
    let error = LAST_ERROR.lock().unwrap();
    match &*error {
        Some(err) => match CString::new(err.clone()) {
            Ok(c_str) => c_str.into_raw(),
            Err(_) => ptr::null_mut(),
        },
        None => ptr::null_mut(),
    }
}

/// Free a string returned by gosilero_vad_last_error
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
        set_error("Invalid arguments".to_string());
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
        set_error("Invalid arguments".to_string());
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
        set_error("Invalid arguments".to_string());
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

/// Process a stream of audio samples and return speech detection results with labels
///
/// # Arguments
///
/// * `vad` - The GosileroVAD instance
/// * `samples` - Pointer to the audio samples
/// * `num_samples` - Number of samples
/// * `threshold_override` - Override the threshold for speech detection (use -1.0 to use the default)
/// * `padding_chunks` - Number of padding chunks to add
/// * `callback` - Callback function to call for each chunk of audio
/// * `user_data` - User data to pass to the callback
///
/// # Returns
///
/// 1 on success, 0 on failure
#[no_mangle]
#[allow(unused)]
pub extern "C" fn gosilero_vad_process_stream(
    vad: *mut GosileroVAD,
    samples: *const i16,
    num_samples: size_t,
    threshold_override: c_float,
    padding_chunks: c_int,
    callback: gosilero_vad_callback,
    user_data: *mut c_void,
) -> c_int {
    if vad.is_null() || samples.is_null() || num_samples == 0 {
        set_error("Invalid arguments".to_string());
        return 0;
    }

    unsafe {
        // Get the internal VAD and its parameters
        let internal_vad = get_internal_vad(vad);
        let vad_ref = &mut internal_vad.vad;
        let chunk_size = internal_vad.chunk_size;
        let sample_rate = internal_vad.sample_rate;

        // Use override threshold if provided, otherwise use the configured threshold
        let threshold = if threshold_override >= 0.0 && threshold_override <= 1.0 {
            threshold_override as f32
        } else {
            internal_vad.voice_threshold
        };

        // Calculate various parameters in samples
        let samples_per_ms = sample_rate as f64 / 1000.0;
        let silence_duration_samples =
            (internal_vad.silence_duration_threshold as f64 * samples_per_ms) as usize;
        let pre_speech_padding_samples =
            (internal_vad.pre_speech_padding as f64 * samples_per_ms) as usize;
        let post_speech_padding_samples =
            (internal_vad.post_speech_padding as f64 * samples_per_ms) as usize;

        // Convert the raw pointer to a slice
        let samples_slice = std::slice::from_raw_parts(samples, num_samples);

        // State tracking
        let mut is_in_speech = false;
        let mut silence_samples_count = 0;
        let mut speech_buffer = Vec::new();
        let mut silence_buffer = Vec::new();

        // Process chunks of samples
        for i in (0..samples_slice.len()).step_by(chunk_size) {
            let end = (i + chunk_size).min(samples_slice.len());
            let chunk = &samples_slice[i..end];

            // Clone the chunk to avoid modifying the input
            let chunk_vec = chunk.to_vec();

            // Predict speech probability
            let probability = vad_ref.predict(chunk_vec);
            let chunk_is_speech = probability >= threshold;

            if chunk_is_speech {
                // Reset silence counter
                silence_samples_count = 0;

                // If not already in speech, handle the transition
                if !is_in_speech {
                    is_in_speech = true;

                    // Apply pre-speech padding if we have buffered silence
                    let pre_padding_size = pre_speech_padding_samples.min(silence_buffer.len());
                    if pre_padding_size > 0 {
                        let start_idx = silence_buffer.len() - pre_padding_size;
                        // Call callback with each pre-padding chunk
                        for j in (start_idx..silence_buffer.len()).step_by(chunk_size) {
                            let end_j = (j + chunk_size).min(silence_buffer.len());
                            if end_j - j > 0 {
                                let pre_chunk = &silence_buffer[j..end_j];
                                callback(
                                    true,
                                    pre_chunk.as_ptr() as *mut i16,
                                    pre_chunk.len(),
                                    user_data,
                                );
                            }
                        }
                    }

                    // Clear silence buffer
                    silence_buffer.clear();
                }

                // Add to speech buffer
                speech_buffer.extend_from_slice(chunk);

                // Call the callback with the speech chunk
                callback(true, chunk.as_ptr() as *mut i16, chunk.len(), user_data);
            } else {
                // Add to silence count
                silence_samples_count += chunk.len();

                // If we were in speech, check if silence is long enough to end the segment
                if is_in_speech {
                    // Add to silence buffer
                    silence_buffer.extend_from_slice(chunk);

                    if silence_samples_count >= silence_duration_samples {
                        // End of speech segment
                        is_in_speech = false;

                        // Apply post-speech padding
                        let post_padding_size =
                            post_speech_padding_samples.min(silence_buffer.len());
                        if post_padding_size > 0 {
                            // Call callback with each post-padding chunk
                            for j in 0..post_padding_size.min(silence_buffer.len()) {
                                let end_j = (j + chunk_size).min(post_padding_size);
                                if end_j - j > 0 {
                                    let post_chunk = &silence_buffer[j..end_j];
                                    callback(
                                        true,
                                        post_chunk.as_ptr() as *mut i16,
                                        post_chunk.len(),
                                        user_data,
                                    );
                                }
                            }
                        }

                        // Clear buffers
                        speech_buffer.clear();
                        silence_buffer.clear();
                    } else {
                        // Still in speech, call callback with silence chunk (treated as speech for continuity)
                        callback(true, chunk.as_ptr() as *mut i16, chunk.len(), user_data);
                    }
                } else {
                    // Update silence buffer (keep only enough for pre-speech padding)
                    silence_buffer.extend_from_slice(chunk);
                    if silence_buffer.len() > pre_speech_padding_samples * 2 {
                        silence_buffer = silence_buffer
                            [(silence_buffer.len() - pre_speech_padding_samples * 2)..]
                            .to_vec();
                    }

                    // Call the callback with the non-speech chunk
                    callback(false, chunk.as_ptr() as *mut i16, chunk.len(), user_data);
                }
            }
        }
    }

    1
}
