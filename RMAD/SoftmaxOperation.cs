namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;

    public class SoftmaxOperation : Operation
    {
        private double[][] _input;

        public SoftmaxOperation() : base()
        {
        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new SoftmaxOperation();
        }

        public double[][] Forward(double[][] input)
        {
            _input = input;
            _output = Softmax(input);
            return _output;
        }

        public override (double[][]?, double[][]?) Backward(double[][] dLdOutput)
        {
            int numRows = _output.Length;
            int numCols = _output[0].Length;

            double[][] dLdInput = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                dLdInput[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    for (int k = 0; k < numCols; k++)
                    {
                        if (j == k)
                        {
                            dLdInput[i][j] += dLdOutput[i][k] * _output[i][j] * (1 - _output[i][j]);
                        }
                        else
                        {
                            dLdInput[i][j] -= dLdOutput[i][k] * _output[i][j] * _output[i][k];
                        }
                    }
                }
            }
            return (dLdInput, dLdInput);
        }

        private double[][] Softmax(double[][] input)
        {
            int numRows = input.Length;
            int numCols = input[0].Length;

            double[][] output = new double[numRows][];
            for (int i = 0; i < numRows; i++)
            {
                output[i] = new double[numCols];
                double max = input[i].Max();
                double sum = 0;
                for (int j = 0; j < numCols; j++)
                {
                    double exp = Math.Exp(input[i][j] - max);
                    sum += exp;
                    output[i][j] = exp;
                }

                for (int j = 0; j < numCols; j++)
                {
                    output[i][j] /= sum;
                }
            }
            return output;
        }
    }
}
