package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/shenjinti/gosilero"
	"github.com/youpy/go-wav"
)

// SpeechSegment represents a detected speech segment
type SpeechSegment struct {
	Start       float64 `json:"start"`
	End         float64 `json:"end"`
	Duration    float64 `json:"duration"`
	StartFrame  int     `json:"start_frame"`
	EndFrame    int     `json:"end_frame"`
	PeakProb    float32 `json:"peak_probability"`
	PeakProbPos float64 `json:"peak_probability_position"`
}

// AnalysisResult contains the results of the voice analysis
type AnalysisResult struct {
	File             string          `json:"file"`
	SampleRate       int             `json:"sample_rate"`
	Duration         float64         `json:"duration"`
	SpeechDuration   float64         `json:"speech_duration"`
	SpeechPercentage float64         `json:"speech_percentage"`
	Segments         []SpeechSegment `json:"segments"`
}

func main() {
	// Command line flags
	wavFile := flag.String("file", "", "Path to WAV file to analyze")
	sampleRate := flag.Int("sample-rate", 0, "Sample rate override (0 to use file's sample rate)")
	chunkSize := flag.Int("chunk-size", 512, "Chunk size (512 recommended for 16kHz)")
	silenceThreshold := flag.Int("silence-ms", 200, "Silence duration threshold (ms)")
	prePadding := flag.Int("pre-padding-ms", 50, "Pre-speech padding (ms)")
	postPadding := flag.Int("post-padding-ms", 50, "Post-speech padding (ms)")
	voiceThreshold := flag.Float64("threshold", 0.6, "Voice activity threshold (0.0-1.0)")
	useJson := flag.Bool("json", false, "Output results in JSON format")
	verboseMode := flag.Bool("verbose", false, "Show detailed probability information for each chunk")

	flag.Parse()

	if *wavFile == "" {
		fmt.Println("Error: WAV file path is required")
		flag.Usage()
		os.Exit(1)
	}

	// Read and process the WAV file
	file, err := os.Open(*wavFile)
	if err != nil {
		log.Fatalf("Error opening WAV file: %v", err)
	}
	defer file.Close()

	// Parse WAV format
	reader := wav.NewReader(file)
	format, err := reader.Format()
	if err != nil {
		log.Fatalf("Error reading WAV format: %v", err)
	}

	// Use file's sample rate if not specified
	actualSampleRate := *sampleRate
	if actualSampleRate == 0 {
		actualSampleRate = int(format.SampleRate)
	}

	// Read all audio samples
	var samples []int16
	for {
		data, err := reader.ReadSamples()
		if err != nil || len(data) == 0 {
			break
		}

		// Convert to int16 samples (first channel only if stereo)
		for _, d := range data {
			sample := int16(reader.IntValue(d, 0))
			samples = append(samples, sample)
		}
	}

	// Check if we have samples
	if len(samples) == 0 {
		log.Fatalf("No audio samples found in the file")
	}

	// Create VAD with options
	options := gosilero.DefaultVADOptions(actualSampleRate, *chunkSize)
	options.SilenceDurationThreshold = time.Duration(*silenceThreshold) * time.Millisecond
	options.PreSpeechPadding = time.Duration(*prePadding) * time.Millisecond
	options.PostSpeechPadding = time.Duration(*postPadding) * time.Millisecond
	options.VoiceThreshold = float32(*voiceThreshold)

	vad, err := gosilero.NewVADWithOptions(options)
	if err != nil {
		log.Fatalf("Error creating VAD: %v", err)
	}
	defer vad.Free()

	// Print info about the file
	if !*useJson {
		fmt.Printf("File: %s\n", *wavFile)
		fmt.Printf("Sample rate: %d Hz, Channels: %d, Bits per sample: %d\n",
			format.SampleRate, format.NumChannels, format.BitsPerSample)
		fmt.Printf("Duration: %.3f seconds, Samples: %d\n",
			float64(len(samples))/float64(actualSampleRate), len(samples))
		fmt.Printf("VAD parameters: threshold=%.2f, silence=%dms, pre-padding=%dms, post-padding=%dms\n",
			*voiceThreshold, *silenceThreshold, *prePadding, *postPadding)
		fmt.Println("\nDetected speech segments:")
		fmt.Println("-------------------------")
	}

	// Calculate various parameters
	silenceDurationSamples := int(float64(*silenceThreshold) * float64(actualSampleRate) / 1000.0)
	prePaddingSamples := int(float64(*prePadding) * float64(actualSampleRate) / 1000.0)
	postPaddingSamples := int(float64(*postPadding) * float64(actualSampleRate) / 1000.0)

	// Process the audio in chunks
	var speechSegments []SpeechSegment
	var inSpeech bool = false
	var silenceCount int = 0
	var currentSegmentStart int = -1
	var currentSegmentPeakProb float32 = 0.0
	var currentSegmentPeakPos int = 0

	// Process each chunk
	for i := 0; i < len(samples); i += *chunkSize {
		end := i + *chunkSize
		if end > len(samples) {
			end = len(samples)
		}

		// Skip chunks that are too small
		if end-i < 100 {
			continue
		}

		// Get the current chunk
		chunk := samples[i:end]

		// Predict speech probability
		prob, err := vad.PredictInt16(chunk)
		if err != nil {
			log.Printf("Warning: Failed to predict chunk at %d-%d: %v", i, end, err)
			continue
		}

		// Current position in seconds
		posSeconds := float64(i) / float64(actualSampleRate)

		// Print detailed info if verbose mode is enabled
		if *verboseMode && !*useJson {
			fmt.Printf("Chunk %.3fs-%.3fs: probability=%.4f\n",
				posSeconds,
				float64(end)/float64(actualSampleRate),
				prob)
		}

		// Check if this is speech
		isSpeech := prob >= float32(*voiceThreshold)

		if isSpeech {
			// Reset silence counter
			silenceCount = 0

			// If not already in speech, start a new segment
			if !inSpeech {
				inSpeech = true

				// Calculate start with pre-padding
				start := i - prePaddingSamples
				if start < 0 {
					start = 0
				}

				currentSegmentStart = start
				currentSegmentPeakProb = prob
				currentSegmentPeakPos = i
			}

			// Update peak probability if needed
			if prob > currentSegmentPeakProb {
				currentSegmentPeakProb = prob
				currentSegmentPeakPos = i
			}
		} else {
			// Increment silence counter
			silenceCount += end - i

			// If we've been in speech and silence is long enough, end the segment
			if inSpeech && silenceCount >= silenceDurationSamples {
				inSpeech = false

				// Calculate end with post-padding
				segmentEnd := i + postPaddingSamples
				if segmentEnd > len(samples) {
					segmentEnd = len(samples)
				}

				// Create segment
				segmentDuration := float64(segmentEnd-currentSegmentStart) / float64(actualSampleRate)
				startTime := float64(currentSegmentStart) / float64(actualSampleRate)
				endTime := float64(segmentEnd) / float64(actualSampleRate)
				peakProbTime := float64(currentSegmentPeakPos) / float64(actualSampleRate)

				segment := SpeechSegment{
					Start:       startTime,
					End:         endTime,
					Duration:    segmentDuration,
					StartFrame:  currentSegmentStart,
					EndFrame:    segmentEnd,
					PeakProb:    currentSegmentPeakProb,
					PeakProbPos: peakProbTime,
				}

				speechSegments = append(speechSegments, segment)

				// Reset peak tracking
				currentSegmentPeakProb = 0.0
			}
		}
	}

	// Handle speech at the end of the file
	if inSpeech {
		segmentEnd := len(samples)

		// Create segment
		segmentDuration := float64(segmentEnd-currentSegmentStart) / float64(actualSampleRate)
		startTime := float64(currentSegmentStart) / float64(actualSampleRate)
		endTime := float64(segmentEnd) / float64(actualSampleRate)
		peakProbTime := float64(currentSegmentPeakPos) / float64(actualSampleRate)

		segment := SpeechSegment{
			Start:       startTime,
			End:         endTime,
			Duration:    segmentDuration,
			StartFrame:  currentSegmentStart,
			EndFrame:    segmentEnd,
			PeakProb:    currentSegmentPeakProb,
			PeakProbPos: peakProbTime,
		}

		speechSegments = append(speechSegments, segment)
	}

	// Calculate total speech duration
	totalDuration := float64(len(samples)) / float64(actualSampleRate)
	totalSpeechDuration := 0.0
	for _, segment := range speechSegments {
		totalSpeechDuration += segment.Duration
	}

	// Create result
	result := AnalysisResult{
		File:             *wavFile,
		SampleRate:       actualSampleRate,
		Duration:         totalDuration,
		SpeechDuration:   totalSpeechDuration,
		SpeechPercentage: (totalSpeechDuration * 100) / totalDuration,
		Segments:         speechSegments,
	}

	// Output results
	if *useJson {
		// JSON output
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(result); err != nil {
			log.Fatalf("Error encoding JSON: %v", err)
		}
	} else {
		// Human-readable output
		for i, segment := range speechSegments {
			fmt.Printf("Segment %d: %.3fs - %.3fs (duration: %.3fs), peak: %.4f at %.3fs\n",
				i+1,
				segment.Start,
				segment.End,
				segment.Duration,
				segment.PeakProb,
				segment.PeakProbPos)
		}

		fmt.Println("-------------------------")
		fmt.Printf("Total speech: %.3f seconds (%.1f%% of file)\n",
			totalSpeechDuration, result.SpeechPercentage)
		fmt.Printf("Segments detected: %d\n", len(speechSegments))
	}
}
