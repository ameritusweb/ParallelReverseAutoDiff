namespace ParallelReverseAutoDiff.RMAD
{
    using System.Threading.Tasks;

    public class HadamardProductOperation : Operation
    {
        private double[][] _input1;
        private double[][] _input2;

        public HadamardProductOperation() : base()
        {

        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new HadamardProductOperation();
        }

        public double[][] Forward(double[][] input1, double[][] input2)
        {
            _input1 = input1;
            _input2 = input2;
            int numRows = _input1.Length;
            int numCols = _input1[0].Length;

            _output = new double[numRows][];
            // Parallelize the outer loop
            Parallel.For(0, numRows, i =>
            {
                _output[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    _output[i][j] = _input1[i][j] * _input2[i][j];
                }
            });

            return _output ;
        }

        public override (double[][]?, double[][]?) Backward(double[][] dOutput)
        {
            int numRows = _input1.Length;
            int numCols = _input1[0].Length;

            // Calculate gradient w.r.t. input1
            double[][] dInput1 = new double[numRows][];
            // Parallelize the outer loop
            Parallel.For(0, numRows, i =>
            {
                dInput1[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    dInput1[i][j] = dOutput[i][j] * _input2[i][j];
                }
            });

            // Calculate gradient w.r.t. input2
            double[][] dInput2 = new double[numRows][];
            // Parallelize the outer loop
            Parallel.For(0, numRows, i =>
            {
                dInput2[i] = new double[numCols];
                for (int j = 0; j < numCols; j++)
                {
                    dInput2[i][j] = dOutput[i][j] * _input1[i][j];
                }
            });

            return (dInput1, dInput2);
        }
    }
}
