namespace ParallelReverseAutoDiff.RMAD
{
    public class MatrixTransposeOperation : Operation
    {
        private double[][] _input;

        public MatrixTransposeOperation() : base()
        {

        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixTransposeOperation();
        }

        public double[][] Forward(double[][] input)
        {
            _input = input;
            int inputRows = input.Length;
            int inputCols = input[0].Length;

            _output = new double[inputCols][];
            for (int i = 0; i < inputCols; i++)
            {
                _output[i] = new double[inputRows];
                for (int j = 0; j < inputRows; j++)
                {
                    _output[i][j] = input[j][i];
                }
            }

            return _output;
        }

        public override (double[][]?, double[][]?) Backward(double[][] dOutput)
        {
            int dOutputRows = dOutput.Length;
            int dOutputCols = dOutput[0].Length;

            double[][] dInput = new double[dOutputCols][];
            for (int i = 0; i < dOutputCols; i++)
            {
                dInput[i] = new double[dOutputRows];
                for (int j = 0; j < dOutputRows; j++)
                {
                    dInput[i][j] = dOutput[j][i];
                }
            }

            return (dInput, dInput);
        }
    }
}
