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

/**
 * Create a new GosileroVAD instance
 *
 * # Arguments
 *
 * * `sample_rate` - The sample rate to use (e.g. 8000, 16000)
 * * `chunk_size` - The chunk size to use (recommended: 256, 512, or 768 for 8000Hz; 512, 768, or 1024 for 16000Hz)
 * * `voice_threshold` - Threshold for speech detection (0.0 to 1.0, default: 0.5)
 *
 * # Returns
 *
 * A `GosileroVADCreateResult` containing the VAD pointer or null on failure
 */
struct GosileroVADCreateResult gosilero_vad_new(int sample_rate,
                                                int chunk_size,
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
 * Get the last error message from a VAD instance
 *
 * # Arguments
 *
 * * `vad` - The GosileroVAD instance
 *
 * # Returns
 *
 * A pointer to a C string containing the error message, or NULL if no error
 * The returned string must be freed with gosilero_vad_free_string
 */
char *gosilero_vad_get_error(struct GosileroVAD *vad);

/**
 * Free a string returned by gosilero_vad_get_error
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

#endif  /* GOSILERO_H */
