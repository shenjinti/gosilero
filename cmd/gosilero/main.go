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

// AnalysisResult contains the results of the voice analysis
type AnalysisResult struct {
	File             string             `json:"file"`
	SampleRate       int                `json:"sample_rate"`
	Duration         float64            `json:"duration"`
	SpeechDuration   float64            `json:"speech_duration"`
	SpeechPercentage float64            `json:"speech_percentage"`
	Segments         []gosilero.Segment `json:"segments"`
}

func main() {
	// Command line flags
	wavFile := flag.String("file", "", "Path to WAV file to analyze")
	sampleRate := flag.Int("sample-rate", 0, "Sample rate override (0 to use file's sample rate)")
	chunkSize := flag.Int("chunk-size", 512, "Chunk size (512 recommended for 16kHz)")
	silenceThreshold := flag.Int("silence-ms", 100, "Min silence duration (ms)")
	speechThreshold := flag.Int("speech-ms", 250, "Min speech duration (ms)")
	prePadding := flag.Int("pre-padding-ms", 30, "Pre-speech padding (ms)")
	postPadding := flag.Int("post-padding-ms", 30, "Post-speech padding (ms)")
	voiceThreshold := flag.Float64("threshold", 0.5, "Voice activity threshold (0.0-1.0)")
	negThreshold := flag.Float64("neg-threshold", 0.35, "Negative threshold (0.0-1.0)")
	useJson := flag.Bool("json", false, "Output results in JSON format")

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

	if *chunkSize != gosilero.DefaultChunkSize {
		log.Fatalf("chunk size must be %d for the built-in Tiny Silero engine", gosilero.DefaultChunkSize)
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
	options.MinSpeechDuration = time.Duration(*speechThreshold) * time.Millisecond
	options.PreSpeechPadding = time.Duration(*prePadding) * time.Millisecond
	options.PostSpeechPadding = time.Duration(*postPadding) * time.Millisecond
	options.VoiceThreshold = float32(*voiceThreshold)
	options.NegativeThreshold = float32(*negThreshold)

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
		fmt.Printf("VAD parameters: threshold=%.2f, neg_threshold=%.2f, silence=%dms, speech=%dms, pre-padding=%dms, post-padding=%dms\n",
			*voiceThreshold, *negThreshold, *silenceThreshold, *speechThreshold, *prePadding, *postPadding)
		fmt.Println("\nDetected speech segments:")
		fmt.Println("-------------------------")
	}

	// Process the audio to get segments
	speechSegments, err := vad.GetSpeechSegments(samples)
	if err != nil {
		log.Fatalf("Error processing audio: %v", err)
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
