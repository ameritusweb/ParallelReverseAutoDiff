namespace ParallelReverseAutoDiff.RMAD
{
    public class LeakyReLUOperation : Operation
    {
        private double[][] _input;
        private double _alpha;

        public LeakyReLUOperation(double alpha = 0.01) : base()
        {
            _alpha = alpha;
        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new LeakyReLUOperation();
        }

        public double[][] Forward(double[][] input)
        {
            _input = input;
            int rows = input.Length;
            int cols = input[0].Length;
            _output = new double[rows][];

            for (int i = 0; i < rows; i++)
            {
                _output[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    double x = input[i][j];
                    _output[i][j] = x > 0 ? x : _alpha * x;
                }
            }

            return _output;
        }

        public override (double[][]?, double[][]?) Backward(double[][] dLdOutput)
        {
            int rows = dLdOutput.Length;
            int cols = dLdOutput[0].Length;
            double[][] dLdInput = new double[rows][];

            for (int i = 0; i < rows; i++)
            {
                dLdInput[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    double x = _input[i][j];
                    double gradient = x > 0 ? 1.0 : _alpha;
                    dLdInput[i][j] = dLdOutput[i][j] * gradient;
                }
            }

            return (dLdInput, dLdInput);
        }
    }

}
