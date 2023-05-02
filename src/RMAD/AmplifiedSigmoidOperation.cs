namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    public class AmplifiedSigmoidOperation : Operation
    {
        private double[][] _input;

        public AmplifiedSigmoidOperation() : base()
        {

        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new AmplifiedSigmoidOperation();
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
                    _output[i][j] = 1.0 / (1.0 + Math.Pow((Math.PI - 2), -input[i][j]));
                }
            }

            return _output;
        }

        public override (double[][]?, double[][]?) Backward(double[][] dLdOutput)
        {
            int numRows = dLdOutput.Length;
            int numCols = dLdOutput[0].Length;
            double[][] dLdInput = new double[numRows][];

            for (int i = 0; i < numRows; i++)
            {
                dLdInput[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    double x = _input[i][j];
                    double dx = Math.Pow((Math.PI - 2), -x) / Math.Pow(1 + Math.Pow((Math.PI - 2), -x), 2);
                    dLdInput[i][j] = dLdOutput[i][j] * dx;
                }
            }

            return (dLdInput, dLdInput);
        }
    }
}
