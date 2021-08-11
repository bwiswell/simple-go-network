package network

import (
	"fmt"
	"math"
)

type Network struct {
	Layers  []*Layer
	NLayers int
	Epoch   int
}

func NewNetwork(layers []*Layer) *Network {
	network := new(Network)
	network.Layers = layers
	network.NLayers = len(layers)
	network.Epoch = 0
	return network
}

func (network *Network) Reset() {
	for _, layer := range network.Layers {
		layer.Reset()
	}
	network.Epoch = 0
}

func (network *Network) Output() []float64 {
	return network.Layers[len(network.Layers)-1].GetActivations()
}

func (network *Network) FeedForward(inputs []float64) []float64 {
	outputs := inputs
	for _, layer := range network.Layers {
		outputs = layer.Activate(outputs)
	}
	return outputs
}

func (network *Network) CalcErrors(target []float64) []float64 {
	actual := network.Output()
	errors := make([]float64, len(target))
	for i := 0; i < len(target); i++ {
		errors[i] = target[i] - actual[i]
	}
	return errors
}

func (network *Network) BigE(target []float64) float64 {
	littleE := network.CalcErrors(target)[0]
	return math.Pow(littleE, 2) * 0.5
}

func (network *Network) Backpropogate(target []float64, eta float64) {
	errorSignals := [][]float64{network.CalcErrors(target)}
	for i := network.NLayers - 1; i >= 0; i-- {
		errorSignals = network.Layers[i].Backpropagate(errorSignals, eta)
	}
}

func (network *Network) Ffbp(inputs, target []float64, eta float64) {
	network.FeedForward(inputs)
	network.Backpropogate(target, eta)
}

func (network *Network) BatchTrain(inputs, targets [][]float64, eta float64, epochs int) {
	n := len(inputs)
	for i := 0; i < epochs; i++ {
		network.Epoch += 1
		for j := 0; j < n; j++ {
			network.Ffbp(inputs[i], targets[i], eta)
		}
	}
}

func (network *Network) OnlineTrain(inputs, targets [][]float64, eta float64, epochs int) {
	n := len(inputs)
	network.Epoch += epochs
	for i := 0; i < n; i++ {
		for j := 0; j < epochs; j++ {
			network.Ffbp(inputs[i], targets[i], eta)
		}
	}
}

func (network *Network) String() string {
	output := fmt.Sprintf("Epoch %v", network.Epoch)
	for _, layer := range network.Layers {
		output += fmt.Sprintf("\n%v", layer)
	}
	return output
}
