package gosilero

import (
	"errors"
	"fmt"
	"time"
)

// VAD provides a voice activity detector powered by the Tiny Silero model.
type VAD struct {
	engine   *tinySileroEngine
	options  VADOptions
	chunkBuf []float32
}

// VADOptions configures the behavior of predicting speech activity.
type VADOptions struct {
	SampleRate               int
	ChunkSize                int
	MinSpeechDuration        time.Duration
	SilenceDurationThreshold time.Duration
	PreSpeechPadding         time.Duration
	PostSpeechPadding        time.Duration
	VoiceThreshold           float32
	NegativeThreshold        float32
}

// DefaultVADOptions returns sane defaults for a given sample rate and chunk size.
func DefaultVADOptions(sampleRate, chunkSize int) VADOptions {
	return VADOptions{
		SampleRate:               sampleRate,
		ChunkSize:                chunkSize,
		MinSpeechDuration:        250 * time.Millisecond,
		SilenceDurationThreshold: 100 * time.Millisecond,
		PreSpeechPadding:         30 * time.Millisecond,
		PostSpeechPadding:        30 * time.Millisecond,
		VoiceThreshold:           0.5,
		NegativeThreshold:        0.35,
	}
}

// NewVAD creates a new detector with the default options for the given sample rate and chunk size.
func NewVAD(sampleRate, chunkSize int) (*VAD, error) {
	options := DefaultVADOptions(sampleRate, chunkSize)
	return NewVADWithOptions(options)
}

// NewVADWithOptions creates a detector configured with specific options.
func NewVADWithOptions(options VADOptions) (*VAD, error) {
	if options.SampleRate <= 0 || options.ChunkSize <= 0 {
		return nil, errors.New("sample rate and chunk size must be positive")
	}
	if options.ChunkSize != tinyChunkSize {
		return nil, fmt.Errorf("chunk size must be %d", tinyChunkSize)
	}
	if options.VoiceThreshold < 0 || options.VoiceThreshold > 1 {
		return nil, errors.New("threshold must be between 0 and 1")
	}

	engine, err := newTinySileroEngine()
	if err != nil {
		return nil, err
	}

	return &VAD{
		engine:   engine,
		options:  options,
		chunkBuf: make([]float32, tinyChunkSize),
	}, nil
}

// Free releases any engine resources. It is safe to call multiple times.
func (v *VAD) Free() {
	if v == nil {
		return
	}
	v.engine = nil
	v.chunkBuf = nil
}

// Predict returns how likely the provided samples contain speech.
func (v *VAD) Predict(samples []float32) (float32, error) {
	if v.engine == nil {
		return 0, errors.New("VAD has been freed")
	}
	if len(samples) == 0 {
		return 0, errors.New("no samples provided")
	}

	chunk := v.chunkBuf
	limit := len(chunk)
	if len(samples) < limit {
		limit = len(samples)
	}

	copy(chunk, samples[:limit])
	for i := limit; i < len(chunk); i++ {
		chunk[i] = 0
	}

	return v.engine.Predict(chunk), nil
}

// PredictInt16 is a convenience wrapper converting 16-bit PCM samples to float32.
func (v *VAD) PredictInt16(samples []int16) (float32, error) {
	if v.engine == nil {
		return 0, errors.New("VAD has been freed")
	}
	if len(samples) == 0 {
		return 0, errors.New("no samples provided")
	}

	chunk := v.chunkBuf
	limit := len(chunk)
	if len(samples) < limit {
		limit = len(samples)
	}

	for i := 0; i < limit; i++ {
		chunk[i] = float32(samples[i]) / 32768.0
	}
	for i := limit; i < len(chunk); i++ {
		chunk[i] = 0
	}

	return v.engine.Predict(chunk), nil
}

// SpeechSegment describes a speech flag for a chunk starting at StartPos.
type SpeechSegment struct {
	StartPos int
	IsSpeech bool
}

// ProcessStreamDirect chunks audio according to ChunkSize and returns a boolean vector of speech.
func (v *VAD) ProcessStreamDirect(samples []int16, threshold float32) ([]SpeechSegment, error) {
	if v.engine == nil {
		return nil, errors.New("VAD has been freed")
	}
	if len(samples) == 0 {
		return nil, errors.New("no samples provided")
	}

	useThreshold := threshold
	if useThreshold <= 0 || useThreshold > 1 {
		useThreshold = v.options.VoiceThreshold
	}

	chunkSize := v.options.ChunkSize
	segments := make([]SpeechSegment, 0, len(samples)/chunkSize+1)

	for i := 0; i < len(samples); i += chunkSize {
		end := i + chunkSize
		if end > len(samples) {
			end = len(samples)
		}
		if end-i < 100 {
			continue
		}

		prob, err := v.PredictInt16(samples[i:end])
		if err != nil {
			return nil, err
		}

		segments = append(segments, SpeechSegment{
			StartPos: i,
			IsSpeech: prob >= useThreshold,
		})
	}

	return segments, nil
}

// Segment represents a detected speech segment with timing information.
type Segment struct {
	Start       float64 `json:"start"`
	End         float64 `json:"end"`
	Duration    float64 `json:"duration"`
	StartSample int     `json:"start_sample"`
	EndSample   int     `json:"end_sample"`
	PeakProb    float32 `json:"peak_probability"`
	PeakProbPos float64 `json:"peak_probability_position"`
}

// Reset clears the internal state of the VAD (LSTM hidden states).
// Call this between processing different audio streams.
func (v *VAD) Reset() {
	if v.engine != nil {
		v.engine.reset()
	}
}

// GetSpeechSegments process the audio and returns continuous speech segments.
func (v *VAD) GetSpeechSegments(samples []int16) ([]Segment, error) {
	if v.engine == nil {
		return nil, errors.New("VAD has been freed")
	}
	if len(samples) == 0 {
		return nil, nil
	}

	v.Reset() // Always start from a clean state for the whole buffer

	sampleRate := float64(v.options.SampleRate)
	minSpeechSamples := int(v.options.MinSpeechDuration.Seconds() * sampleRate)
	minSilenceSamples := int(v.options.SilenceDurationThreshold.Seconds() * sampleRate)
	prePaddingSamples := int(v.options.PreSpeechPadding.Seconds() * sampleRate)
	postPaddingSamples := int(v.options.PostSpeechPadding.Seconds() * sampleRate)

	var segments []Segment
	triggered := false
	tempEnd := 0
	currentStart := 0
	var currentPeakProb float32
	var currentPeakPos int

	chunkSize := v.options.ChunkSize

	for i := 0; i < len(samples); i += chunkSize {
		end := i + chunkSize
		if end > len(samples) {
			end = len(samples)
		}
		if end-i < 100 {
			continue
		}

		prob, err := v.PredictInt16(samples[i:end])
		if err != nil {
			return nil, err
		}

		if prob >= v.options.VoiceThreshold && tempEnd != 0 {
			tempEnd = 0
		}

		if prob >= v.options.VoiceThreshold {
			if !triggered {
				triggered = true
				currentStart = i
				currentPeakProb = prob
				currentPeakPos = i
			}
			if prob > currentPeakProb {
				currentPeakProb = prob
				currentPeakPos = i
			}
		} else if prob < v.options.NegativeThreshold && triggered {
			if tempEnd == 0 {
				tempEnd = i
			}

			if i-tempEnd >= minSilenceSamples {
				// The actual speech ended at tempEnd
				if tempEnd-currentStart >= minSpeechSamples {
					speechEnd := tempEnd + postPaddingSamples
					if speechEnd > len(samples) {
						speechEnd = len(samples)
					}

					actualStart := currentStart - prePaddingSamples
					if actualStart < 0 {
						actualStart = 0
					}

					segment := Segment{
						Start:       float64(actualStart) / sampleRate,
						End:         float64(speechEnd) / sampleRate,
						Duration:    float64(speechEnd-actualStart) / sampleRate,
						StartSample: actualStart,
						EndSample:   speechEnd,
						PeakProb:    currentPeakProb,
						PeakProbPos: float64(currentPeakPos) / sampleRate,
					}
					// Ensure no overlap with previous segment
					if len(segments) > 0 {
						prevEnd := segments[len(segments)-1].EndSample
						if segment.StartSample < prevEnd {
							segment.StartSample = prevEnd
							segment.Start = float64(segment.StartSample) / sampleRate
							segment.Duration = segment.End - segment.Start
						}
					}
					if segment.Duration > 0 {
						segments = append(segments, segment)
					}
				}
				triggered = false
				tempEnd = 0
				currentPeakProb = 0
			}
		}
	}

	if triggered {
		speechEnd := len(samples)
		if speechEnd-currentStart >= minSpeechSamples {
			actualStart := currentStart - prePaddingSamples
			if actualStart < 0 {
				actualStart = 0
			}
			segment := Segment{
				Start:       float64(actualStart) / sampleRate,
				End:         float64(speechEnd) / sampleRate,
				Duration:    float64(speechEnd-actualStart) / sampleRate,
				StartSample: actualStart,
				EndSample:   speechEnd,
				PeakProb:    currentPeakProb,
				PeakProbPos: float64(currentPeakPos) / sampleRate,
			}
			if len(segments) > 0 {
				prevEnd := segments[len(segments)-1].EndSample
				if segment.StartSample < prevEnd {
					segment.StartSample = prevEnd
					segment.Start = float64(segment.StartSample) / sampleRate
					segment.Duration = segment.End - segment.Start
				}
			}
			if segment.Duration > 0 {
				segments = append(segments, segment)
			}
		}
	}

	return segments, nil
}
