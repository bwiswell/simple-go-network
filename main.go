package main

import (
	"fmt"

	"github.com/bwiswell/simple-go-network/network"

	"github.com/bwiswell/simple-go-network/util"
)

func main() {
	weights := []float64{0.5, 0.5}
	perA := network.NewPerceptron(weights[:], 0.5, util.Sigmoid)
	perB := network.NewPerceptron(weights[:], 0.5, util.Sigmoid)
	layerAPers := []*network.Perceptron{perA, perB}
	layerA := network.NewLayer(layerAPers)
	perC := network.NewPerceptron(weights[:], 0.5, util.Sigmoid)
	layerBPers := []*network.Perceptron{perC}
	layerB := network.NewLayer(layerBPers)
	networkLayers := []*network.Layer{layerA, layerB}
	net := network.NewNetwork(networkLayers)

	inputA := []float64{0, 0}
	inputB := []float64{0, 1}
	inputC := []float64{1, 0}
	inputD := []float64{1, 1}
	inputs := [][]float64{inputA, inputB, inputC, inputD}
	outputA := []float64{0}
	outputB := []float64{0.5}
	outputC := []float64{0.5}
	outputD := []float64{1}
	outputs := [][]float64{outputA, outputB, outputC, outputD}
	net.BatchTrain(inputs, outputs, 0.1, 1000)
	showGuesses(net, inputs)
	net.BatchTrain(inputs, outputs, 0.1, 1000)
	showGuesses(net, inputs)
	net.BatchTrain(inputs, outputs, 0.1, 1000)
	showGuesses(net, inputs)
}

func showGuesses(net *network.Network, inputs [][]float64) {
	fmt.Printf("--------------------\nEpoch: %v\n", net.Epoch)
	for _, inputData := range inputs {
		outputData := net.FeedForward(inputData)
		fmt.Printf("Input: %v; Output: %v\n", inputData, outputData)
	}
}
