package network

import "fmt"

var currID int

func init() {
	currID = 0
}

func zeroArray(array []float64) []float64 {
	for i, _ := range array {
		array[i] = 0.0
	}
	return array
}

type Perceptron struct {
	ID             int
	NInputs        int
	InitialWeights []float64
	Weights        []float64
	InitialBias    float64
	Bias           float64
	ActivationFn   func(activity float64) float64
	DeltaWeights   []float64
	DeltaBias      float64
	Activity       float64
	Activation     float64
}

func NewPerceptron(weights []float64, bias float64, activationFn func(activity float64) float64) *Perceptron {
	perceptron := new(Perceptron)
	perceptron.ID = currID
	currID += 1
	perceptron.NInputs = len(weights)
	perceptron.InitialWeights = weights
	perceptron.Weights = make([]float64, perceptron.NInputs)
	copy(perceptron.Weights, weights)
	perceptron.InitialBias = bias
	perceptron.Bias = bias
	perceptron.ActivationFn = activationFn
	perceptron.DeltaWeights = make([]float64, perceptron.NInputs)
	perceptron.DeltaBias = 0.0
	perceptron.Activity = 0.0
	perceptron.Activation = 0.0
	return perceptron
}

func (per *Perceptron) Reset() {
	copy(per.Weights, per.InitialWeights)
	per.Bias = per.InitialBias
	per.DeltaWeights = zeroArray(per.DeltaWeights)
	per.DeltaBias = 0.0
	per.Activity = 0.0
	per.Activation = 0.0
}

func (per *Perceptron) Activate(inputs []float64) float64 {
	total := per.Bias
	for i := 0; i < per.NInputs; i++ {
		total += per.Weights[i] * inputs[i]
	}
	per.Activity = total
	per.Activation = per.ActivationFn(total)
	return per.Activation
}

func (per *Perceptron) Backpropagate(error, eta float64, lastInputs []float64) []float64 {
	delta := error * (1 - per.Activation) * per.Activation
	error_signals := make([]float64, per.NInputs)
	for i := 0; i < per.NInputs; i++ {
		per.DeltaWeights[i] = eta * delta * lastInputs[i]
		error_signals[i] = delta * per.Weights[i]
		per.Weights[i] += per.DeltaWeights[i]
	}
	per.DeltaBias = eta * delta
	per.Bias += per.DeltaBias
	return error_signals
}

func (per *Perceptron) String() string {
	output := fmt.Sprintf("Perceptron %v\nActivity: %v\nActivation: %v", per.ID, per.Activity, per.Activation)
	return fmt.Sprintf("---------------\n%v\n---------------", output)
}
