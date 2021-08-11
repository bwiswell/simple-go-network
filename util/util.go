package util

import "math"

func Sigmoid(activity float64) float64 {
	return 1 / (1 + math.Pow(math.E, -activity))
}
