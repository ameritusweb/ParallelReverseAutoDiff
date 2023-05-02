namespace ParallelReverseAutoDiff.RMAD
{
    using System;

    public class ApplyDropoutOperation : Operation
    {
        private double[][] _input;
        private double[][] _dropoutMask;
        private double _dropoutRate;
        private Random _random;

        public ApplyDropoutOperation(double dropoutRate) : base()
        {
            _dropoutRate = dropoutRate;
            _random = new Random();
        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new ApplyDropoutOperation(net.GetDropoutRate());
        }

        public double[][] Forward(double[][] input)
        {
            _input = input;
            int numRows = input.Length;
            int numCols = input[0].Length;

            _output = new double[numRows][];
            _dropoutMask = new double[numRows][];

            for (int i = 0; i < numRows; i++)
            {
                _output[i] = new double[numCols];
                _dropoutMask[i] = new double[numCols];

                for (int j = 0; j < numCols; j++)
                {
                    double randomValue = _random.NextDouble();
                    _dropoutMask[i][j] = randomValue < _dropoutRate ? 0 : 1;
                    _output[i][j] = _dropoutMask[i][j] * input[i][j];
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
                    dLdInput[i][j] = dLdOutput[i][j] * _dropoutMask[i][j];
                }
            }
            return (dLdInput, dLdInput);
        }
    }
}
