package gosilero

/*
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/dist -lgosilero_rs -Wl,-rpath,${SRCDIR}/dist
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/gosilero-rs/target/x86_64-unknown-linux-gnu/release -lgosilero_rs
#include <stdlib.h>
#include <stdint.h>
#include "gosilero-rs/gosilero.h"

// Forward declaration of the Go callback bridge function
void goCallbackBridge(bool is_speech, int16_t* samples, size_t num_samples, void* user_data);
*/
import "C"
import (
	"errors"
	"runtime"
	"sync"
	"time"
	"unsafe"
)

// VAD represents a Voice Activity Detector instance
type VAD struct {
	handle *C.struct_GosileroVAD
	mu     sync.Mutex
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

// SpeechCallback is a callback function for processing speech segments
type SpeechCallback func(isSpeech bool, samples []int16)

// CallbackContext holds the callback and any associated data for stream processing
type CallbackContext struct {
	callback SpeechCallback
}

// Global map to store callback contexts
var (
	callbackContexts      = make(map[uintptr]*CallbackContext)
	callbackContextsMu    sync.Mutex
	nextCallbackContextID uintptr
)

//export goCallbackBridge
func goCallbackBridge(isSpeech C.bool, samples *C.int16_t, numSamples C.size_t, userData unsafe.Pointer) {
	callbackContextsMu.Lock()
	contextID := uintptr(userData)
	context, exists := callbackContexts[contextID]
	callbackContextsMu.Unlock()

	if !exists || context == nil {
		return
	}

	// Convert C array to Go slice without copying
	samplesSlice := unsafe.Slice((*int16)(unsafe.Pointer(samples)), int(numSamples))

	// Create a copy of the samples to avoid data races
	samplesCopy := make([]int16, len(samplesSlice))
	copy(samplesCopy, samplesSlice)

	context.callback(bool(isSpeech), samplesCopy)
}

// NewVAD creates a new Voice Activity Detector with the given parameters
func NewVAD(sampleRate, chunkSize int) (*VAD, error) {
	options := DefaultVADOptions(sampleRate, chunkSize)
	return NewVADWithOptions(options)
}

// NewVADWithOptions creates a new Voice Activity Detector with the given options
func NewVADWithOptions(options VADOptions) (*VAD, error) {
	// Convert time.Duration to milliseconds
	silenceDurationMs := uint64(options.SilenceDurationThreshold.Milliseconds())
	preSpeechPaddingMs := uint64(options.PreSpeechPadding.Milliseconds())
	postSpeechPaddingMs := uint64(options.PostSpeechPadding.Milliseconds())

	result := C.gosilero_vad_new(
		C.int(options.SampleRate),
		C.int(options.ChunkSize),
		C.uint64_t(silenceDurationMs),
		C.uint64_t(preSpeechPaddingMs),
		C.uint64_t(postSpeechPaddingMs),
		C.float(options.VoiceThreshold),
	)
	if !bool(result.success) {
		return nil, getLastError()
	}

	vad := &VAD{handle: result.vad}
	runtime.SetFinalizer(vad, (*VAD).Free)
	return vad, nil
}

// Free releases the resources associated with this VAD
func (v *VAD) Free() {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.handle != nil {
		C.gosilero_vad_free(v.handle)
		v.handle = nil
	}
}

// Predict determines if the given audio samples contain speech
// Returns a probability between 0.0 and 1.0
func (v *VAD) Predict(samples []float32) (float32, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

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
	v.mu.Lock()
	defer v.mu.Unlock()

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

// PredictInt8 determines if the given 8-bit audio samples contain speech
// Returns a probability between 0.0 and 1.0
func (v *VAD) PredictInt8(samples []int8) (float32, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.handle == nil {
		return 0, errors.New("VAD has been freed")
	}

	if len(samples) == 0 {
		return 0, errors.New("no samples provided")
	}

	result := C.gosilero_vad_predict_i8(
		v.handle,
		(*C.int8_t)(unsafe.Pointer(&samples[0])),
		C.size_t(len(samples)),
	)

	return float32(result), nil
}

// ProcessStream processes a stream of audio samples and calls the callback with segments of speech/non-speech
func (v *VAD) ProcessStream(samples []int16, threshold float32, paddingChunks int, callback SpeechCallback) error {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.handle == nil {
		return errors.New("VAD has been freed")
	}

	if len(samples) == 0 {
		return errors.New("no samples provided")
	}

	// Store the callback context
	callbackContextsMu.Lock()
	contextID := nextCallbackContextID
	nextCallbackContextID++
	callbackContexts[contextID] = &CallbackContext{callback: callback}
	callbackContextsMu.Unlock()

	// Clean up the callback context when done
	defer func() {
		callbackContextsMu.Lock()
		delete(callbackContexts, contextID)
		callbackContextsMu.Unlock()
	}()

	result := C.gosilero_vad_process_stream(
		v.handle,
		(*C.int16_t)(unsafe.Pointer(&samples[0])),
		C.size_t(len(samples)),
		C.float(threshold),
		C.int(paddingChunks),
		C.gosilero_vad_callback(C.goCallbackBridge),
		unsafe.Pointer(uintptr(contextID)),
	)

	if result == 0 {
		return getLastError()
	}

	return nil
}

// getLastError retrieves the last error from the Rust library
func getLastError() error {
	errPtr := C.gosilero_vad_last_error()
	if errPtr == nil {
		return errors.New("unknown error")
	}
	defer C.gosilero_vad_free_string(errPtr)
	return errors.New(C.GoString(errPtr))
}
