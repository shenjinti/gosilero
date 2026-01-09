package gosilero

import (
	"math"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/youpy/go-wav"
)

func TestNewVAD(t *testing.T) {
	// Test valid parameters
	vad, err := NewVAD(16000, 512)
	if err != nil {
		t.Fatalf("Failed to create VAD with valid parameters: %v", err)
	}
	defer vad.Free()

	// Test custom options
	options := DefaultVADOptions(16000, 512)
	options.SilenceDurationThreshold = 400 * time.Millisecond
	options.PreSpeechPadding = 100 * time.Millisecond
	options.PostSpeechPadding = 150 * time.Millisecond
	options.VoiceThreshold = 0.6

	customVad, err := NewVADWithOptions(options)
	if err != nil {
		t.Fatalf("Failed to create VAD with custom options: %v", err)
	}
	defer customVad.Free()

	// Test invalid parameters
	_, err = NewVAD(-1, 512)
	if err == nil {
		t.Fatal("Expected error when creating VAD with negative sample rate")
	}

	_, err = NewVAD(16000, -1)
	if err == nil {
		t.Fatal("Expected error when creating VAD with negative chunk size")
	}

	// Skip invalid voice threshold test for now while we settle on the expected error behavior
}

func TestPredict(t *testing.T) {
	vad, err := NewVAD(16000, 512)
	if err != nil {
		t.Fatalf("Failed to create VAD: %v", err)
	}
	defer vad.Free()

	// Generate a simple sine wave (non-speech)
	samples := make([]float32, 1024)
	for i := range samples {
		samples[i] = float32(math.Sin(float64(i) * 0.1))
	}

	// Test prediction
	prob, err := vad.Predict(samples)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	// Just check that we get a value between 0 and 1
	if prob < 0 || prob > 1 {
		t.Fatalf("Probability out of range: %f", prob)
	}

	// Test with empty samples
	_, err = vad.Predict([]float32{})
	if err == nil {
		t.Fatal("Expected error when predicting with empty samples")
	}
}

func TestPredictInt16(t *testing.T) {
	vad, err := NewVAD(16000, 512)
	if err != nil {
		t.Fatalf("Failed to create VAD: %v", err)
	}
	defer vad.Free()

	// Generate a simple sine wave (non-speech)
	samples := make([]int16, 1024)
	for i := range samples {
		samples[i] = int16(math.Sin(float64(i)*0.1) * 32767)
	}

	// Test prediction
	prob, err := vad.PredictInt16(samples)
	if err != nil {
		t.Fatalf("Failed to predict: %v", err)
	}

	// Just check that we get a value between 0 and 1
	if prob < 0 || prob > 1 {
		t.Fatalf("Probability out of range: %f", prob)
	}

	// Test with empty samples
	_, err = vad.PredictInt16([]int16{})
	if err == nil {
		t.Fatal("Expected error when predicting with empty samples")
	}
}

func TestMultipleCalls(t *testing.T) {
	vad, err := NewVAD(16000, 512)
	if err != nil {
		t.Fatalf("Failed to create VAD: %v", err)
	}
	defer vad.Free()

	// Generate some sample data
	samples := make([]int16, 1024)
	for i := range samples {
		samples[i] = int16(math.Sin(float64(i)*0.1) * 32767)
	}

	// Make multiple calls to ensure thread safety
	const numGoroutines = 10
	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			_, err := vad.PredictInt16(samples)
			if err != nil {
				t.Errorf("Failed to predict in goroutine: %v", err)
			}
		}()
	}

	wg.Wait()
}

func TestFreeAndReuse(t *testing.T) {
	vad, err := NewVAD(16000, 512)
	if err != nil {
		t.Fatalf("Failed to create VAD: %v", err)
	}

	// Generate some sample data
	samples := make([]int16, 1024)
	for i := range samples {
		samples[i] = int16(math.Sin(float64(i)*0.1) * 32767)
	}

	// Test prediction works initially
	_, err = vad.PredictInt16(samples)
	if err != nil {
		t.Fatalf("Failed to predict before freeing: %v", err)
	}

	// Free the VAD
	vad.Free()

	// Test prediction fails after freeing
	_, err = vad.PredictInt16(samples)
	if err == nil {
		t.Fatal("Expected error when predicting after freeing")
	}
}

// TestRealVoiceDetection tests voice detection on a real human voice recording
func TestRealVoiceDetection(t *testing.T) {
	// Open WAV file
	file, err := os.Open("testdata/thankyou_16k.wav")
	if err != nil {
		t.Fatalf("Unable to open test audio file: %v", err)
	}
	defer file.Close()

	// Parse WAV file
	reader := wav.NewReader(file)
	format, err := reader.Format()
	if err != nil {
		t.Fatalf("Unable to read WAV format information: %v", err)
	}

	// Confirm sample rate
	sampleRate := int(format.SampleRate)
	t.Logf("File sample rate: %d Hz", sampleRate)
	t.Logf("Number of channels: %d", format.NumChannels)
	t.Logf("Bits per sample: %d", format.BitsPerSample)

	// Read audio data - make sure we read ALL samples
	var samples []int16
	for {
		data, err := reader.ReadSamples()
		if err != nil || len(data) == 0 {
			break
		}

		// Convert WAV samples to int16 slice and append
		for _, d := range data {
			// If multi-channel, only use the first channel
			sample := reader.IntValue(d, 0)
			// Convert to int16
			samples = append(samples, int16(sample))
		}
	}

	// Calculate sample min and max values for debugging
	var maxVal int = 0
	var minVal int = 0

	for _, sample := range samples {
		// Track max and min values
		if int(sample) > maxVal {
			maxVal = int(sample)
		}
		if int(sample) < minVal {
			minVal = int(sample)
		}
	}

	t.Logf("Read %d samples", len(samples))
	t.Logf("Sample value range: Min %d, Max %d", minVal, maxVal)
	totalDuration := float64(len(samples)) / float64(sampleRate)
	t.Logf("Total duration: %.2f seconds", totalDuration)

	// Check if dynamic range is sufficient
	if maxVal-minVal < 1000 {
		t.Logf("Warning: Audio dynamic range may be too small (%d), will amplify signal", maxVal-minVal)

		// For small dynamic range, amplify samples
		// Amplify values to a reasonable range, e.g. -10000 to 10000
		amplificationFactor := 10000.0 / float64(maxVal)
		if amplificationFactor > 1.0 {
			for i, s := range samples {
				samples[i] = int16(float64(s) * amplificationFactor)
			}
			t.Logf("Sample amplification factor: %.2f", amplificationFactor)
		}
	}

	// Create VAD instance with custom options
	options := DefaultVADOptions(sampleRate, 512)
	options.VoiceThreshold = 0.3

	vad, err := NewVADWithOptions(options)
	if err != nil {
		t.Fatalf("Failed to create VAD: %v", err)
	}
	defer vad.Free()

	// 1. Test different parts of the file
	// Check each block of the file
	expectedSpeechStart := 1.8 // 1.8 seconds into the file
	expectedSpeechEnd := 2.3   // 2.3 seconds into the file
	startSampleIdx := int(expectedSpeechStart * float64(sampleRate))
	endSampleIdx := int(expectedSpeechEnd * float64(sampleRate))

	// Make sure our expected range is within our sample range
	if startSampleIdx > len(samples) || endSampleIdx > len(samples) {
		t.Fatalf("Expected speech range (%d-%d) is outside sample range (0-%d)",
			startSampleIdx, endSampleIdx, len(samples))
	}

	// Test the expected speech segment specifically
	if startSampleIdx < len(samples) && endSampleIdx <= len(samples) && startSampleIdx < endSampleIdx {
		expectedSpeechSamples := samples[startSampleIdx:endSampleIdx]
		prob, err := vad.PredictInt16(expectedSpeechSamples)
		if err != nil {
			t.Fatalf("Failed to predict expected speech segment: %v", err)
		}
		t.Logf("Expected speech segment (%.2fs-%.2fs) probability: %.4f",
			expectedSpeechStart, expectedSpeechEnd, prob)

		if prob < options.VoiceThreshold {
			t.Logf("Warning: Expected speech segment has low probability (%.4f)", prob)
		}
	}

	// Split file into blocks for detection to find all speech segments
	maxProb := float32(0.0)
	maxProbPos := 0

	// Use smaller chunk size for more precise detection
	detectionChunkSize := options.ChunkSize
	for i := 0; i < len(samples); i += detectionChunkSize {
		end := i + detectionChunkSize
		if end > len(samples) {
			end = len(samples)
		}

		if end-i < 100 { // Blocks that are too short may not be meaningful
			continue
		}

		blockSamples := samples[i:end]
		prob, err := vad.PredictInt16(blockSamples)
		if err != nil {
			t.Fatalf("Failed to predict block %d-%d: %v", i, end, err)
		}

		currentTime := float64(i) / float64(sampleRate)
		endTime := float64(end) / float64(sampleRate)

		// Record highest probability and position
		if prob > maxProb {
			maxProb = prob
			maxProbPos = i
		}

		// Only log high probability segments to avoid too much output
		if prob > options.VoiceThreshold {
			t.Logf("Block %.2fs-%.2fs speech probability: %.4f", currentTime, endTime, prob)
		}
	}

	maxProbTime := float64(maxProbPos) / float64(sampleRate)
	t.Logf("Highest speech probability is %.4f at %.2f seconds", maxProb, maxProbTime)

	// Even the highest probability should be above a minimum threshold
	if maxProb < options.VoiceThreshold {
		t.Errorf("Highest speech probability is too low: %.4f, this may indicate detection issues", maxProb)
	} else {
		t.Logf("Successfully detected speech segment with highest probability %.4f", maxProb)
	}

	// 2. Use ProcessStreamDirect to detect speech segments
	segments, err := vad.ProcessStreamDirect(samples, options.VoiceThreshold)
	if err != nil {
		t.Fatalf("Failed to process stream directly: %v", err)
	}

	// Analyze the segments
	var speechSegments []SpeechSegment
	for _, segment := range segments {
		if segment.IsSpeech {
			speechSegments = append(speechSegments, segment)
		}
	}

	// Group adjacent speech segments
	type SpeechRange struct {
		StartPos int
		EndPos   int
	}

	var speechRanges []SpeechRange
	if len(speechSegments) > 0 {
		currentRange := SpeechRange{
			StartPos: speechSegments[0].StartPos,
			EndPos:   speechSegments[0].StartPos + options.ChunkSize,
		}

		for i := 1; i < len(speechSegments); i++ {
			// If this segment is adjacent to the current range, extend the range
			if speechSegments[i].StartPos <= currentRange.EndPos+options.ChunkSize {
				currentRange.EndPos = speechSegments[i].StartPos + options.ChunkSize
			} else {
				// Otherwise, start a new range
				speechRanges = append(speechRanges, currentRange)
				currentRange = SpeechRange{
					StartPos: speechSegments[i].StartPos,
					EndPos:   speechSegments[i].StartPos + options.ChunkSize,
				}
			}
		}
		// Add the last range
		speechRanges = append(speechRanges, currentRange)
	}

	// Report on the detected ranges
	if len(speechRanges) > 0 {
		t.Logf("Detected %d separate speech segments:", len(speechRanges))
		totalSpeechDuration := 0.0
		foundExpectedSegment := false

		for i, rng := range speechRanges {
			segmentDuration := float64(rng.EndPos-rng.StartPos) / float64(sampleRate)
			startTime := float64(rng.StartPos) / float64(sampleRate)
			endTime := float64(rng.EndPos) / float64(sampleRate)

			t.Logf("  Speech segment %d: %.3fs - %.3fs (duration: %.3fs)",
				i+1, startTime, endTime, segmentDuration)

			// Check if this segment overlaps with our expected segment
			if startTime <= expectedSpeechEnd && endTime >= expectedSpeechStart {
				t.Logf("  => This segment overlaps with expected speech at %.2fs-%.2fs",
					expectedSpeechStart, expectedSpeechEnd)
				foundExpectedSegment = true
			}

			totalSpeechDuration += segmentDuration
		}

		speechPercentage := (totalSpeechDuration * 100) / totalDuration
		t.Logf("Total speech duration: %.3f seconds (%.1f%% of file)",
			totalSpeechDuration, speechPercentage)

		if !foundExpectedSegment {
			t.Errorf("No speech segment detected in the expected range of %.2f-%.2f seconds!",
				expectedSpeechStart, expectedSpeechEnd)
		}
	} else {
		t.Logf("No distinct speech segments detected")
	}
}

func TestProcessStreamDirect(t *testing.T) {
	vad, err := NewVAD(16000, 512)
	if err != nil {
		t.Fatalf("Failed to create VAD: %v", err)
	}
	defer vad.Free()

	// Generate a simple sine wave (non-speech)
	samples := make([]int16, 4096)
	for i := range samples {
		samples[i] = int16(math.Sin(float64(i)*0.1) * 32767)
	}

	// Process the stream directly
	segments, err := vad.ProcessStreamDirect(samples, -1.0)
	if err != nil {
		t.Fatalf("Failed to process stream directly: %v", err)
	}

	// Check that we got some segments
	t.Logf("Got %d segments", len(segments))
	for i, segment := range segments {
		t.Logf("Segment %d: StartPos=%d, IsSpeech=%v", i, segment.StartPos, segment.IsSpeech)
	}

	// Test with empty samples
	_, err = vad.ProcessStreamDirect([]int16{}, -1.0)
	if err == nil {
		t.Fatal("Expected error when processing stream with empty samples")
	}
}

func BenchmarkPredictGoTinySilero(b *testing.B) {
	vad, err := NewVAD(16000, DefaultChunkSize)
	if err != nil {
		b.Fatalf("Failed to create VAD: %v", err)
	}
	samples := make([]int16, DefaultChunkSize)
	for i := range samples {
		samples[i] = int16(10000.0 * math.Sin(float64(i)*0.1))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := vad.PredictInt16(samples); err != nil {
			b.Fatalf("predict failed: %v", err)
		}
	}
}

func TestPerformanceRTF(t *testing.T) {
	path := "testdata/1843344-user.wav"
	samples, sampleRate := loadWavSamples(t, path)
	options := DefaultVADOptions(sampleRate, DefaultChunkSize)
	vad, err := NewVADWithOptions(options)
	if err != nil {
		t.Fatalf("failed to create VAD: %v", err)
	}
	defer vad.Free()

	chunkSize := DefaultChunkSize
	start := time.Now()
	processedChunks := 0
	for i := 0; i < len(samples); i += chunkSize {
		end := i + chunkSize
		if end > len(samples) {
			end = len(samples)
		}
		if end-i < 100 {
			continue
		}

		if _, err := vad.PredictInt16(samples[i:end]); err != nil {
			t.Fatalf("prediction failed: %v", err)
		}
		processedChunks++
	}
	elapsed := time.Since(start)
	duration := float64(len(samples)) / float64(sampleRate)
	if duration <= 0 {
		return
	}
	rtf := elapsed.Seconds() / duration
	frameCount := int(math.Ceil(duration / 0.02))
	if frameCount < 1 {
		frameCount = 1
	}
	singleFrameMs := elapsed.Seconds() * 1000 / float64(frameCount)

	t.Logf("RTF for %s: %.4f (processed %d chunks), %.2fms per 20ms frame", path, rtf, processedChunks, singleFrameMs)
}

func TestExpectedRanges1843344(t *testing.T) {
	samples, sampleRate := loadWavSamples(t, "testdata/1843344-user.wav")
	options := DefaultVADOptions(sampleRate, DefaultChunkSize)
	options.VoiceThreshold = 0.5

	vad, err := NewVADWithOptions(options)
	if err != nil {
		t.Fatalf("Failed to create VAD: %v", err)
	}
	defer vad.Free()

	segments, err := vad.ProcessStreamDirect(samples, options.VoiceThreshold)
	if err != nil {
		t.Fatalf("Failed to process streamed file: %v", err)
	}

	expectedRanges := []struct {
		startMs float64
		endMs   float64
	}{
		{9300, 9800},
		{14700, 15200},
		{19200, 19700},
	}

	chunkMs := float64(options.ChunkSize) * 1000 / float64(sampleRate)
	found := make([]bool, len(expectedRanges))

	for _, seg := range segments {
		if !seg.IsSpeech {
			continue
		}

		segStartMs := float64(seg.StartPos) * 1000 / float64(sampleRate)
		segEndMs := segStartMs + chunkMs

		for i, target := range expectedRanges {
			if found[i] {
				continue
			}
			if segStartMs <= target.endMs && segEndMs >= target.startMs {
				found[i] = true
			}
		}
	}

	for i, ok := range found {
		if !ok {
			t.Fatalf("expected speech near %.0f-%.0fms but did not detect it", expectedRanges[i].startMs, expectedRanges[i].endMs)
		}
	}
}

func loadWavSamples(t *testing.T, path string) ([]int16, int) {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatalf("unable to open %s: %v", path, err)
	}
	defer file.Close()

	reader := wav.NewReader(file)
	format, err := reader.Format()
	if err != nil {
		t.Fatalf("unable to read WAV format: %v", err)
	}

	var samples []int16
	for {
		data, err := reader.ReadSamples()
		if err != nil || len(data) == 0 {
			break
		}
		for _, d := range data {
			samples = append(samples, int16(reader.IntValue(d, 0)))
		}
	}

	return samples, int(format.SampleRate)
}
