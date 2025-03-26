/* Auto-generated with cbindgen */

#ifndef GOSILERO_H
#define GOSILERO_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include "stdlib.h"
#include "stdint.h"
#include "stdbool.h"

typedef struct GosileroVAD {
  uint8_t _private[0];
} GosileroVAD;

typedef struct GosileroVADCreateResult {
  struct GosileroVAD *vad;
  bool success;
} GosileroVADCreateResult;

typedef void (*gosilero_vad_callback)(bool is_speech,
                                      int16_t *samples,
                                      size_t num_samples,
                                      void *user_data);

/**
 * Create a new GosileroVAD instance
 *
 * # Arguments
 *
 * * `sample_rate` - The sample rate to use (e.g. 8000, 16000)
 * * `chunk_size` - The chunk size to use (recommended: 256, 512, or 768 for 8000Hz; 512, 768, or 1024 for 16000Hz)
 * * `silence_duration_threshold` - Minimum duration of silence to split speech segments (in ms, default: 500)
 * * `pre_speech_padding` - Padding before speech segment (in ms, default: 150)
 * * `post_speech_padding` - Padding after speech segment (in ms, default: 200)
 * * `voice_threshold` - Threshold for speech detection (0.0 to 1.0, default: 0.5)
 *
 * # Returns
 *
 * A `GosileroVADCreateResult` containing the VAD pointer or null on failure
 */
struct GosileroVADCreateResult gosilero_vad_new(int sample_rate,
                                                int chunk_size,
                                                uint64_t silence_duration_threshold,
                                                uint64_t pre_speech_padding,
                                                uint64_t post_speech_padding,
                                                float voice_threshold);

/**
 * Create a new GosileroVAD instance with default parameters
 *
 * # Arguments
 *
 * * `sample_rate` - The sample rate to use (e.g. 8000, 16000)
 * * `chunk_size` - The chunk size to use (recommended: 256, 512, or 768 for 8000Hz; 512, 768, or 1024 for 16000Hz)
 *
 * # Returns
 *
 * A `GosileroVADCreateResult` containing the VAD pointer or null on failure
 */
struct GosileroVADCreateResult gosilero_vad_new_with_defaults(int sample_rate,
                                                              int chunk_size);

/**
 * Free a GosileroVAD instance
 *
 * # Arguments
 *
 * * `vad` - The GosileroVAD instance to free
 */
void gosilero_vad_free(struct GosileroVAD *vad);

/**
 * Get the last error message
 *
 * # Returns
 *
 * A pointer to a C string containing the error message, or NULL if no error
 * The returned string must be freed with gosilero_vad_free_string
 */
char *gosilero_vad_last_error(void);

/**
 * Free a string returned by gosilero_vad_last_error
 *
 * # Arguments
 *
 * * `str` - The string to free
 */
void gosilero_vad_free_string(char *str);

/**
 * Predict if a chunk of 32-bit float audio samples contains speech
 *
 * # Arguments
 *
 * * `vad` - The GosileroVAD instance
 * * `samples` - Pointer to the audio samples
 * * `num_samples` - Number of samples
 *
 * # Returns
 *
 * The probability of speech (0.0 to 1.0)
 */
float gosilero_vad_predict(struct GosileroVAD *vad, const float *samples, size_t num_samples);

/**
 * Predict if a chunk of 16-bit integer audio samples contains speech
 *
 * # Arguments
 *
 * * `vad` - The GosileroVAD instance
 * * `samples` - Pointer to the audio samples
 * * `num_samples` - Number of samples
 *
 * # Returns
 *
 * The probability of speech (0.0 to 1.0)
 */
float gosilero_vad_predict_i16(struct GosileroVAD *vad, const int16_t *samples, size_t num_samples);

/**
 * Predict if a chunk of 8-bit integer audio samples contains speech
 *
 * # Arguments
 *
 * * `vad` - The GosileroVAD instance
 * * `samples` - Pointer to the audio samples
 * * `num_samples` - Number of samples
 *
 * # Returns
 *
 * The probability of speech (0.0 to 1.0)
 */
float gosilero_vad_predict_i8(struct GosileroVAD *vad, const int8_t *samples, size_t num_samples);

/**
 * Process a stream of audio samples and return speech detection results with labels
 *
 * # Arguments
 *
 * * `vad` - The GosileroVAD instance
 * * `samples` - Pointer to the audio samples
 * * `num_samples` - Number of samples
 * * `threshold_override` - Override the threshold for speech detection (use -1.0 to use the default)
 * * `padding_chunks` - Number of padding chunks to add
 * * `callback` - Callback function to call for each chunk of audio
 * * `user_data` - User data to pass to the callback
 *
 * # Returns
 *
 * 1 on success, 0 on failure
 */
int gosilero_vad_process_stream(struct GosileroVAD *vad,
                                const int16_t *samples,
                                size_t num_samples,
                                float threshold_override,
                                int padding_chunks,
                                gosilero_vad_callback callback,
                                void *user_data);

#endif  /* GOSILERO_H */
