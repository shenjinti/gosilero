package gosilero

/*
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/dist -lgosilero_rs -Wl,-rpath,${SRCDIR}/dist
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/dist -lgosilero_rs
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "gosilero-rs/gosilero.h"

// Forward declarations for functions that might not be in the header yet
extern char* gosilero_vad_get_error(struct GosileroVAD* vad);
extern int gosilero_vad_process_stream_direct(
    struct GosileroVAD* vad,
    const int16_t* samples,
    size_t num_samples,
    float threshold_override,
    uint8_t* output_is_speech,
    size_t* output_positions,
    size_t output_size,
    size_t* actual_output_size
);
*/
import "C"
import (
	"errors"
	"runtime"
	"time"
	"unsafe"
)

// VAD represents a Voice Activity Detector instance
type VAD struct {
	handle  *C.struct_GosileroVAD
	options VADOptions // Store options for later use
}

// VADOptions contains all configuration options for the VAD
type VADOptions struct {
	// SampleRate is the audio sample rate in Hz (e.g. 8000, 16000)
	SampleRate int
	// ChunkSize is the chunk size to use (recommended: 256, 512, or 768 for 8000Hz; 512, 768, or 1024 for 16000Hz)
	ChunkSize int
	// SilenceDurationThreshold is the minimum duration of silence to split speech segments
	SilenceDurationThreshold time.Duration
	// PreSpeechPadding is the padding before speech segment
	PreSpeechPadding time.Duration
	// PostSpeechPadding is the padding after speech segment
	PostSpeechPadding time.Duration
	// VoiceThreshold is the threshold for speech detection (0.0 to 1.0)
	VoiceThreshold float32
}

// DefaultVADOptions returns the default VAD options
func DefaultVADOptions(sampleRate, chunkSize int) VADOptions {
	return VADOptions{
		SampleRate:               sampleRate,
		ChunkSize:                chunkSize,
		SilenceDurationThreshold: 200 * time.Millisecond,
		PreSpeechPadding:         50 * time.Millisecond,
		PostSpeechPadding:        50 * time.Millisecond,
		VoiceThreshold:           0.6,
	}
}

// NewVAD creates a new Voice Activity Detector with the given parameters
func NewVAD(sampleRate, chunkSize int) (*VAD, error) {
	options := DefaultVADOptions(sampleRate, chunkSize)
	return NewVADWithOptions(options)
}

// NewVADWithOptions creates a new Voice Activity Detector with the given options
func NewVADWithOptions(options VADOptions) (*VAD, error) {
	result := C.gosilero_vad_new(
		C.int(options.SampleRate),
		C.int(options.ChunkSize),
		C.float(options.VoiceThreshold),
	)
	if !bool(result.success) {
		return nil, getLastError()
	}

	vad := &VAD{
		handle:  result.vad,
		options: options, // Store the options
	}
	runtime.SetFinalizer(vad, (*VAD).Free)
	return vad, nil
}

// Free releases the resources associated with this VAD
func (v *VAD) Free() {
	if v.handle != nil {
		C.gosilero_vad_free(v.handle)
		v.handle = nil
	}
}

// Predict determines if the given audio samples contain speech
// Returns a probability between 0.0 and 1.0
func (v *VAD) Predict(samples []float32) (float32, error) {
	if v.handle == nil {
		return 0, errors.New("VAD has been freed")
	}

	if len(samples) == 0 {
		return 0, errors.New("no samples provided")
	}

	result := C.gosilero_vad_predict(
		v.handle,
		(*C.float)(unsafe.Pointer(&samples[0])),
		C.size_t(len(samples)),
	)

	return float32(result), nil
}

// PredictInt16 determines if the given 16-bit audio samples contain speech
// Returns a probability between 0.0 and 1.0
func (v *VAD) PredictInt16(samples []int16) (float32, error) {
	if v.handle == nil {
		return 0, errors.New("VAD has been freed")
	}

	if len(samples) == 0 {
		return 0, errors.New("no samples provided")
	}

	result := C.gosilero_vad_predict_i16(
		v.handle,
		(*C.int16_t)(unsafe.Pointer(&samples[0])),
		C.size_t(len(samples)),
	)

	return float32(result), nil
}

// SpeechSegment represents a detected speech segment
type SpeechSegment struct {
	// Start position in samples
	StartPos int
	// IsSpeech indicates if this segment contains speech
	IsSpeech bool
}

// ProcessStreamDirect processes a stream of audio samples and returns the detected speech segments directly
// This avoids the need for callbacks and provides a simpler API
func (v *VAD) ProcessStreamDirect(samples []int16, threshold float32) ([]SpeechSegment, error) {
	if v.handle == nil {
		return nil, errors.New("VAD has been freed")
	}

	if len(samples) == 0 {
		return nil, errors.New("no samples provided")
	}

	// Use the passed threshold if valid, otherwise use the default
	useThreshold := threshold
	if threshold < 0 || threshold > 1 {
		useThreshold = v.options.VoiceThreshold
	}

	// Process each chunk of audio and build segments directly
	chunkSize := v.options.ChunkSize
	segments := make([]SpeechSegment, 0, len(samples)/chunkSize+1)

	for i := 0; i < len(samples); i += chunkSize {
		end := i + chunkSize
		if end > len(samples) {
			end = len(samples)
		}

		// Skip chunks that are too small
		if end-i < 100 {
			continue
		}

		chunk := samples[i:end]
		// Call C function directly to avoid lock reentry
		prob := float32(C.gosilero_vad_predict_i16(
			v.handle,
			(*C.int16_t)(unsafe.Pointer(&chunk[0])),
			C.size_t(len(chunk)),
		))

		segments = append(segments, SpeechSegment{
			StartPos: i,
			IsSpeech: prob >= useThreshold,
		})
	}

	return segments, nil
}

// getLastError retrieves the last error from the Rust library
func getLastError() error {
	// Return a generic error since we can't access the global error
	return errors.New("an error occurred while operating on the VAD")
}
