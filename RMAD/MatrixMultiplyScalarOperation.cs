namespace ParallelReverseAutoDiff.RMAD
{
    using System.Threading.Tasks;

    public class MatrixMultiplyScalarOperation : Operation
    {
        private double[][] _input;
        private double _scalar;

        public MatrixMultiplyScalarOperation() : base()
        {
            
        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixMultiplyScalarOperation();
        }

        public double[][] Forward(double[][] input, double scalar)
        {
            _scalar = scalar;
            _input = input;
            int rows = input.Length;
            int cols = input[0].Length;
            _output = new double[rows][];

            // Parallelize the outer loop
            Parallel.For(0, rows, i =>
            {
                _output[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    _output[i][j] = input[i][j] * _scalar;
                }
            });

            return _output;
        }

        public override (double[][]?, double[][]?) Backward(double[][] dLdOutput)
        {
            int rows = dLdOutput.Length;
            int cols = dLdOutput[0].Length;
            double[][] dLdInput = new double[rows][];

            // Parallelize the outer loop
            Parallel.For(0, rows, i =>
            {
                dLdInput[i] = new double[cols];
                for (int j = 0; j < cols; j++)
                {
                    dLdInput[i][j] = dLdOutput[i][j] * _scalar;
                }
            });

            return (dLdInput, dLdInput);
        }
    }

}
