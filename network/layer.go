package network

import "fmt"

var layerID int = 0

type Layer struct {
	ID           int
	NPerceptrons int
	Perceptrons  []*Perceptron
	LastInputs   []float64
}

func NewLayer(perceptrons []*Perceptron) *Layer {
	layer := new(Layer)
	layer.ID = layerID
	layerID += 1
	layer.NPerceptrons = len(perceptrons)
	layer.Perceptrons = perceptrons
	return layer
}

func (layer *Layer) Reset() {
	for _, per := range layer.Perceptrons {
		per.Reset()
	}
	layer.LastInputs = make([]float64, 0)
}

func (layer *Layer) GetActivations() []float64 {
	activations := make([]float64, layer.NPerceptrons)
	for i, per := range layer.Perceptrons {
		activations[i] = per.Activation
	}
	return activations
}

func (layer *Layer) Activate(inputs []float64) []float64 {
	layer.LastInputs = inputs
	activations := make([]float64, layer.NPerceptrons)
	for i, per := range layer.Perceptrons {
		activations[i] = per.Activate(inputs)
	}
	return activations
}

func (layer *Layer) Backpropagate(errors [][]float64, eta float64) [][]float64 {
	errorSignals := make([][]float64, layer.NPerceptrons)
	var error float64
	for i, per := range layer.Perceptrons {
		error = 0
		for j := 0; j < len(errors); j++ {
			error += errors[j][i]
		}
		errorSignals[i] = per.Backpropagate(error, eta, layer.LastInputs)
	}
	return errorSignals
}

func (layer *Layer) String() string {
	output := fmt.Sprintf("Layer %v\n", layer.ID)
	for _, per := range layer.Perceptrons {
		output += fmt.Sprintf("\n%v", per)
	}
	return output
}
