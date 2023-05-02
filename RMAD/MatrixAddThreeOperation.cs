namespace ParallelReverseAutoDiff.RMAD
{
    public class MatrixAddThreeOperation : Operation
    {
        private double[][] _inputA;
        private double[][] _inputB;
        private double[][] _bias;
        private double _learningRate;

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixAddThreeOperation(net.GetLearningRate());
        }

        private MatrixAddThreeOperation(double learningRate) : base()
        {
            _learningRate = learningRate;
        }

        public double[][] Forward(double[][] inputA, double[][] inputB, double[][] bias)
        {
            _inputA = inputA;
            _inputB = inputB;
            _bias = bias;
            int numRows = inputA.Length;
            int numCols = inputA[0].Length;
            _output = new double[numRows][];

            for (int i = 0; i < numRows; i++)
            {
                _output[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    _output[i][j] = inputA[i][j] + inputB[i][j] + bias[i][j];
                }
            }

            return _output;
        }

        public override (double[][]?, double[][]?) Backward(double[][] dOutput)
        {
            int numRows = dOutput.Length;
            int numCols = dOutput[0].Length;
            double[][] dInputA = new double[numRows][];
            double[][] dInputB = new double[numRows][];

            for (int i = 0; i < numRows; i++)
            {
                dInputA[i] = new double[numCols];
                dInputB[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    dInputA[i][j] = dOutput[i][j];
                    dInputB[i][j] = dOutput[i][j];
                }
            }

            return (dInputA, dInputB); // You can return either dInputA or dInputB, as they are identical.
        }
    }

}
