namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    public class SigmoidOperation : Operation
    {
        private double[][] _input;

        public SigmoidOperation() : base()
        {

        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new SigmoidOperation();
        }

        public double[][] Forward(double[][] input)
        {
            _input = input;
            int numRows = input.Length;
            int numCols = input[0].Length;

            _output = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                _output[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    _output[i][j] = 1.0 / (1.0 + Math.Exp(-input[i][j]));
                }
            }

            return _output;
        }

        public override (double[][]?, double[][]?) Backward(double[][] dOutput)
        {
            int numRows = _input.Length;
            int numCols = _input[0].Length;

            double[][] dInput = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                dInput[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    double sigmoidDerivative = _output[i][j] * (1 - _output[i][j]);
                    dInput[i][j] = dOutput[i][j] * sigmoidDerivative;
                }
            }

            return (dInput, dInput);
        }
    }
}
