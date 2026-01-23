package gosilero

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

const (
	tinyChunkSize    = 512
	DefaultChunkSize = tinyChunkSize
	contextSize      = 64
	stftPadding      = 64
	stftWindowSize   = 256
	stftStride       = 128
	freqBins         = stftWindowSize/2 + 1
	lstmHiddenSize   = 128
	sigmoidTableSize = 1024
	sigmoidTableMin  = -8.0
	sigmoidTableMax  = 8.0
	tanhTableSize    = 1024
	tanhTableMin     = -5.0
	tanhTableMax     = 5.0
)

//go:embed silero_weights.bin
var sileroWeights []byte

var (
	errInvalidWeights = errors.New("silero weights are malformed")
	sigmoidTable      []float32
	tanhTable         []float32
	sigmoidMin        float32
	sigmoidMax        float32
	sigmoidInvStep    float32
	tanhMin           float32
	tanhMax           float32
	tanhInvStep       float32
	fftTwiddleTable   [32][]complex64 // log2(256)=8, but 32 is safer
)

type conv1dLayer struct {
	weights     []float32
	bias        []float32
	inChannels  int
	outChannels int
	kernelSize  int
	stride      int
	padding     int
	relu        bool
}

func newConv1dLayer(inChannels, outChannels, kernelSize, stride, padding int, relu bool) *conv1dLayer {
	return &conv1dLayer{
		inChannels:  inChannels,
		outChannels: outChannels,
		kernelSize:  kernelSize,
		stride:      stride,
		padding:     padding,
		relu:        relu,
	}
}

func (c *conv1dLayer) loadWeights(data []float32) error {
	expected := c.outChannels * c.inChannels * c.kernelSize
	if len(data) != expected {
		return fmt.Errorf("unexpected conv weights: got %d expected %d", len(data), expected)
	}

	if cap(c.weights) < expected {
		c.weights = make([]float32, expected)
	} else {
		c.weights = c.weights[:expected]
	}
	for oc := 0; oc < c.outChannels; oc++ {
		for ic := 0; ic < c.inChannels; ic++ {
			for k := 0; k < c.kernelSize; k++ {
				c.weights[(ic*c.kernelSize+k)*c.outChannels+oc] = data[(oc*c.inChannels+ic)*c.kernelSize+k]
			}
		}
	}
	return nil
}

func (c *conv1dLayer) loadBias(data []float32) error {
	if len(data) != c.outChannels {
		return fmt.Errorf("unexpected conv bias size: got %d expected %d", len(data), c.outChannels)
	}
	c.bias = append(c.bias[:0], data...)
	return nil
}

func (c *conv1dLayer) forward(input []float32, inputLen int, output []float32, padBuf []float32) {
	outputLen := (inputLen+2*c.padding-(c.kernelSize-1)-1)/c.stride + 1
	total := outputLen * c.outChannels
	if len(output) < total {
		panic("conv1d output buffer too small")
	}
	paddedLen := inputLen + 2*c.padding
	padSize := paddedLen * c.inChannels
	if len(padBuf) < padSize {
		panic("conv1d padBuf too small")
	}
	padBuf = padBuf[:padSize]
	for i := range padBuf {
		padBuf[i] = 0
	}
	for ic := 0; ic < c.inChannels; ic++ {
		srcStart := ic * inputLen
		dstStart := ic*paddedLen + c.padding
		copy(padBuf[dstStart:dstStart+inputLen], input[srcStart:srcStart+inputLen])
	}

	weights := c.weights
	bias := c.bias
	inChannels := c.inChannels
	kernelSize := c.kernelSize
	stride := c.stride
	outChannels := c.outChannels

	// Initialize output with bias
	for oc := 0; oc < outChannels; oc++ {
		biasVal := float32(0)
		if bias != nil {
			biasVal = bias[oc]
		}
		base := oc * outputLen
		for t := 0; t < outputLen; t++ {
			output[base+t] = biasVal
		}
	}

	for ic := 0; ic < inChannels; ic++ {
		inBaseBase := ic * paddedLen
		for k := 0; k < kernelSize; k++ {
			wBase := (ic*kernelSize + k) * outChannels
			wSlice := weights[wBase : wBase+outChannels]
			_ = wSlice[outChannels-1]

			for t := 0; t < outputLen; t++ {
				v := padBuf[inBaseBase+t*stride+k]
				if v == 0 {
					continue
				}
				for oc := 0; oc < outChannels; oc++ {
					output[oc*outputLen+t] += v * wSlice[oc]
				}
			}
		}
	}

	if c.relu {
		for i := 0; i < total; i++ {
			if output[i] < 0 {
				output[i] = 0
			}
		}
	}
}

type weightReader struct {
	data   []byte
	offset int
}

func (r *weightReader) readUint32() (uint32, error) {
	if r.offset+4 > len(r.data) {
		return 0, errInvalidWeights
	}
	value := binary.LittleEndian.Uint32(r.data[r.offset : r.offset+4])
	r.offset += 4
	return value, nil
}

func (r *weightReader) readBytes(count int) ([]byte, error) {
	if count < 0 || r.offset+count > len(r.data) {
		return nil, errInvalidWeights
	}
	result := r.data[r.offset : r.offset+count]
	r.offset += count
	return result, nil
}

func (r *weightReader) readFloat32s(byteCount int, out []float32) ([]float32, error) {
	if byteCount%4 != 0 || r.offset+byteCount > len(r.data) {
		return nil, errInvalidWeights
	}
	count := byteCount / 4
	if cap(out) < count {
		out = make([]float32, count)
	} else {
		out = out[:count]
	}
	for i := 0; i < count; i++ {
		bits := binary.LittleEndian.Uint32(r.data[r.offset : r.offset+4])
		out[i] = math.Float32frombits(bits)
		r.offset += 4
	}
	return out, nil
}

type tinySileroModel struct {
	window        []float32
	fftBitReverse []int

	enc0 *conv1dLayer
	enc1 *conv1dLayer
	enc2 *conv1dLayer
	enc3 *conv1dLayer
	out  *conv1dLayer

	lstmWIH []float32
	lstmWHH []float32
	lstmBIH []float32
	lstmBHH []float32
}

var (
	defaultModel *tinySileroModel
	modelInitErr error
)

func getSharedModel() (*tinySileroModel, error) {
	if defaultModel != nil {
		return defaultModel, nil
	}
	if modelInitErr != nil {
		return nil, modelInitErr
	}

	m := &tinySileroModel{
		window:        generateHannWindow(stftWindowSize),
		fftBitReverse: makeBitReverse(stftWindowSize),

		enc0: newConv1dLayer(freqBins, 128, 3, 1, 1, true), // stride=1
		enc1: newConv1dLayer(128, 64, 3, 2, 1, true),       // stride=2 (fixed)
		enc2: newConv1dLayer(64, 64, 3, 2, 1, true),        // stride=2
		enc3: newConv1dLayer(64, 128, 3, 1, 1, true),       // stride=1 (fixed)
		out:  newConv1dLayer(128, 1, 1, 1, 0, false),
	}

	if err := m.loadWeights(sileroWeights); err != nil {
		modelInitErr = err
		return nil, err
	}
	defaultModel = m
	return m, nil
}

type tinySileroEngine struct {
	model *tinySileroModel

	fftInput       []complex64
	bufMag         []float32
	bufEnc0Out     []float32
	bufEnc1Out     []float32
	bufEnc2Out     []float32
	bufEnc3Out     []float32
	bufGates       []float32
	lstmInput      []float32
	h              []float32
	c              []float32
	scratchPad     []float32
	bufContext     []float32
	bufWithContext []float32
	bufPadded      []float32
}

func newTinySileroEngine() (*tinySileroEngine, error) {
	model, err := getSharedModel()
	if err != nil {
		return nil, err
	}

	engine := &tinySileroEngine{
		model:          model,
		fftInput:       make([]complex64, stftWindowSize),
		bufMag:         make([]float32, freqBins*4), // 4 frames instead of 3
		bufEnc0Out:     make([]float32, 128*4),      // stride=1: 4->4
		bufEnc1Out:     make([]float32, 64*2),       // stride=2: 4->2
		bufEnc2Out:     make([]float32, 64*1),       // stride=2: 2->1
		bufEnc3Out:     make([]float32, 128*1),      // stride=1: 1->1
		bufGates:       make([]float32, 4*lstmHiddenSize),
		lstmInput:      make([]float32, lstmHiddenSize),
		h:              make([]float32, lstmHiddenSize),
		c:              make([]float32, lstmHiddenSize),
		scratchPad:     make([]float32, 1024), // Large enough for any layer padBuf
		bufContext:     make([]float32, contextSize),
		bufWithContext: make([]float32, contextSize+tinyChunkSize),
		bufPadded:      make([]float32, contextSize+tinyChunkSize+stftPadding),
	}

	return engine, nil
}

func (m *tinySileroModel) loadWeights(data []byte) error {
	reader := weightReader{data: data}
	numTensors, err := reader.readUint32()
	if err != nil {
		return err
	}

	var tmpFloats []float32
	for i := 0; i < int(numTensors); i++ {
		nameLen, err := reader.readUint32()
		if err != nil {
			return err
		}
		nameBytes, err := reader.readBytes(int(nameLen))
		if err != nil {
			return err
		}
		name := string(nameBytes)

		shapeLen, err := reader.readUint32()
		if err != nil {
			return err
		}
		for j := 0; j < int(shapeLen); j++ {
			if _, err := reader.readUint32(); err != nil {
				return err
			}
		}

		dataLen, err := reader.readUint32()
		if err != nil {
			return err
		}

		var target *[]float32
		needsTransform := false
		switch name {
		case "lstm0_w_ih":
			target = &m.lstmWIH
			needsTransform = true
		case "lstm0_w_hh":
			target = &m.lstmWHH
			needsTransform = true
		case "lstm0_b_ih":
			target = &m.lstmBIH
		case "lstm0_b_hh":
			target = &m.lstmBHH
		}

		if target != nil {
			tmpFloats, err = reader.readFloat32s(int(dataLen), tmpFloats)
			if err != nil {
				return err
			}

			if needsTransform {
				h := lstmHiddenSize
				transformed := make([]float32, len(tmpFloats))
				for i := 0; i < 4*h; i++ {
					for j := 0; j < h; j++ {
						transformed[j*4*h+i] = tmpFloats[i*h+j]
					}
				}
				*target = transformed
			} else {
				*target = append((*target)[:0], tmpFloats...)
			}
			continue
		}

		tmpFloats, err = reader.readFloat32s(int(dataLen), tmpFloats)
		if err != nil {
			return err
		}

		switch name {
		case "enc0_weight":
			if err := m.enc0.loadWeights(tmpFloats); err != nil {
				return err
			}
		case "enc0_bias":
			if err := m.enc0.loadBias(tmpFloats); err != nil {
				return err
			}
		case "enc1_weight":
			if err := m.enc1.loadWeights(tmpFloats); err != nil {
				return err
			}
		case "enc1_bias":
			if err := m.enc1.loadBias(tmpFloats); err != nil {
				return err
			}
		case "enc2_weight":
			if err := m.enc2.loadWeights(tmpFloats); err != nil {
				return err
			}
		case "enc2_bias":
			if err := m.enc2.loadBias(tmpFloats); err != nil {
				return err
			}
		case "enc3_weight":
			if err := m.enc3.loadWeights(tmpFloats); err != nil {
				return err
			}
		case "enc3_bias":
			if err := m.enc3.loadBias(tmpFloats); err != nil {
				return err
			}
		case "out_weight":
			if err := m.out.loadWeights(tmpFloats); err != nil {
				return err
			}
		case "out_bias":
			if err := m.out.loadBias(tmpFloats); err != nil {
				return err
			}
		default:
			continue
		}
	}

	if len(m.lstmWIH) == 0 || len(m.lstmWHH) == 0 || len(m.lstmBIH) == 0 || len(m.lstmBHH) == 0 {
		return errInvalidWeights
	}
	if len(m.enc0.weights) == 0 || len(m.enc1.weights) == 0 || len(m.enc2.weights) == 0 || len(m.enc3.weights) == 0 {
		return errInvalidWeights
	}
	if len(m.out.weights) == 0 || len(m.out.bias) == 0 {
		return errInvalidWeights
	}

	return nil
}

func generateHannWindow(size int) []float32 {
	if size <= 0 {
		return nil
	}
	window := make([]float32, size)
	denom := float64(size - 1)
	for i := 0; i < size; i++ {
		window[i] = 0.5 - 0.5*float32(math.Cos(2*math.Pi*float64(i)/denom))
	}
	return window
}

func makeBitReverse(size int) []int {
	table := make([]int, size)
	bits := 0
	for l := size; l > 1; l >>= 1 {
		bits++
	}
	for i := 0; i < size; i++ {
		rev := 0
		for j := 0; j < bits; j++ {
			if (i>>j)&1 == 1 {
				rev |= 1 << (bits - 1 - j)
			}
		}
		table[i] = rev
	}
	return table
}

func init() {
	sigmoidTable = buildActivationTable(float64(sigmoidTableMin), float64(sigmoidTableMax), sigmoidTableSize, func(x float64) float64 {
		return 1 / (1 + math.Exp(-x))
	})
	sigmoidMin = float32(sigmoidTableMin)
	sigmoidMax = float32(sigmoidTableMax)
	sigmoidInvStep = float32(float64(sigmoidTableSize-1) / (sigmoidTableMax - sigmoidTableMin))

	tanhTable = buildActivationTable(float64(tanhTableMin), float64(tanhTableMax), tanhTableSize, math.Tanh)
	tanhMin = float32(tanhTableMin)
	tanhMax = float32(tanhTableMax)
	tanhInvStep = float32(float64(tanhTableSize-1) / (tanhTableMax - tanhTableMin))

	buildFFTTwiddles(stftWindowSize)
}

func fftInPlace(data []complex64, bitReverse []int) {
	n := len(data)
	for i := 0; i < n; i++ {
		j := bitReverse[i]
		if j > i {
			data[i], data[j] = data[j], data[i]
		}
	}

	bits := 0
	for length := 2; length <= n; length <<= 1 {
		bits++
		half := length / 2
		factors := fftTwiddleTable[bits]
		for i := 0; i < n; i += length {
			for j := 0; j < half; j++ {
				w := factors[j]
				u := data[i+j]
				v := data[i+j+half] * w
				data[i+j] = u + v
				data[i+j+half] = u - v
			}
		}
	}
}

func buildActivationTable(min, max float64, size int, fn func(float64) float64) []float32 {
	if size < 2 {
		return nil
	}
	step := (max - min) / float64(size-1)
	table := make([]float32, size)
	for i := 0; i < size; i++ {
		x := min + float64(i)*step
		table[i] = float32(fn(x))
	}
	return table
}

func buildFFTTwiddles(size int) {
	for length := 2; length <= size; length <<= 1 {
		bits := 0
		for t := length; t > 1; t >>= 1 {
			bits++
		}
		half := length / 2
		angle := -2 * math.Pi / float64(length)
		wLen := complex(float32(math.Cos(angle)), float32(math.Sin(angle)))
		factors := make([]complex64, half)
		w := complex(float32(1), float32(0))
		for j := 0; j < half; j++ {
			factors[j] = w
			w *= wLen
		}
		fftTwiddleTable[bits] = factors
	}
}

func sigmoid(x float32) float32 {
	if x <= sigmoidMin {
		return sigmoidTable[0]
	}
	if x >= sigmoidMax {
		return sigmoidTable[len(sigmoidTable)-1]
	}
	idx := (x - sigmoidMin) * sigmoidInvStep
	base := int(idx)
	fraction := idx - float32(base)
	return sigmoidTable[base] + fraction*(sigmoidTable[base+1]-sigmoidTable[base])
}

func tanh32(x float32) float32 {
	if x <= tanhMin {
		return tanhTable[0]
	}
	if x >= tanhMax {
		return tanhTable[len(tanhTable)-1]
	}
	idx := (x - tanhMin) * tanhInvStep
	base := int(idx)
	fraction := idx - float32(base)
	return tanhTable[base] + fraction*(tanhTable[base+1]-tanhTable[base])
}

func (e *tinySileroEngine) reset() {
	for i := range e.h {
		e.h[i] = 0
		e.c[i] = 0
	}
	for i := range e.bufContext {
		e.bufContext[i] = 0
	}
}

func (e *tinySileroEngine) Predict(audio []float32) float32 {
	if len(audio) < tinyChunkSize {
		return 0
	}

	model := e.model

	copy(e.bufWithContext[:contextSize], e.bufContext)
	copy(e.bufWithContext[contextSize:], audio[:tinyChunkSize])

	inputLen := contextSize + tinyChunkSize // 576
	copy(e.bufPadded[:inputLen], e.bufWithContext)

	for i := 0; i < stftPadding; i++ {
		srcIdx := inputLen - 2 - i // -2 to skip the last element and go backwards
		e.bufPadded[inputLen+i] = e.bufWithContext[srcIdx]
	}

	for frame := 0; frame < 4; frame++ {
		start := frame * stftStride
		input := e.fftInput
		win := model.window

		for i := 0; i < stftWindowSize; i += 4 {
			input[i] = complex(e.bufPadded[start+i]*win[i], 0)
			input[i+1] = complex(e.bufPadded[start+i+1]*win[i+1], 0)
			input[i+2] = complex(e.bufPadded[start+i+2]*win[i+2], 0)
			input[i+3] = complex(e.bufPadded[start+i+3]*win[i+3], 0)
		}
		fftInPlace(e.fftInput, model.fftBitReverse)
		for bin := 0; bin < freqBins; bin++ {
			c := e.fftInput[bin]
			re := real(c)
			im := imag(c)
			e.bufMag[bin*4+frame] = float32(math.Sqrt(float64(re*re + im*im)))
		}
	}

	// Save last 64 samples of current chunk as context for next chunk
	copy(e.bufContext, audio[tinyChunkSize-contextSize:tinyChunkSize])

	model.enc0.forward(e.bufMag, 4, e.bufEnc0Out, e.scratchPad)
	model.enc1.forward(e.bufEnc0Out, 4, e.bufEnc1Out, e.scratchPad)
	model.enc2.forward(e.bufEnc1Out, 2, e.bufEnc2Out, e.scratchPad)
	model.enc3.forward(e.bufEnc2Out, 1, e.bufEnc3Out, e.scratchPad)

	copy(e.lstmInput, e.bufEnc3Out)

	H := lstmHiddenSize
	bih := model.lstmBIH
	bhh := model.lstmBHH
	wih := model.lstmWIH
	whh := model.lstmWHH
	input := e.lstmInput
	h := e.h
	gates := e.bufGates

	// Initialize gates with bias
	copy(gates, bih)
	for i := 0; i < 4*H; i++ {
		gates[i] += bhh[i]
	}

	// Apply W_ih (Layout: [H, 4*H]) - Broadcasting input x
	for j := 0; j < H; j++ {
		x := input[j]
		if x == 0 {
			continue
		}
		wStart := j * 4 * H
		for i := 0; i < 4*H; i++ {
			gates[i] += wih[wStart+i] * x
		}
	}

	// Apply W_hh (Layout: [H, 4*H]) - Broadcasting hidden state h
	for j := 0; j < H; j++ {
		hVal := h[j]
		if hVal == 0 {
			continue
		}
		wStart := j * 4 * H
		for i := 0; i < 4*H; i++ {
			gates[i] += whh[wStart+i] * hVal
		}
	}

	// CRITICAL: Gate order is IFGO (PyTorch standard)
	for j := 0; j < H; j++ {
		iGate := sigmoid(gates[0*H+j]) // Position 0: I (Input)
		fGate := sigmoid(gates[1*H+j]) // Position 1: F (Forget)
		gGate := tanh32(gates[2*H+j])  // Position 2: G (Cell/Gate)
		oGate := sigmoid(gates[3*H+j]) // Position 3: O (Output)

		cNew := fGate*e.c[j] + iGate*gGate
		e.c[j] = cNew
		e.h[j] = oGate * tanh32(cNew)
	}

	bias := model.out.bias[0]
	sum := bias
	for j := 0; j < lstmHiddenSize; j++ {
		val := e.h[j]
		if val < 0 {
			val = 0
		}
		sum += model.out.weights[j] * val
	}

	return sigmoid(sum)
}
