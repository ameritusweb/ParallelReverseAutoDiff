namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Linq;
    using System.Threading.Tasks;

    public class LayerNormalizationOperation : Operation
    {

        private const double _epsilon = 1E-6;
        private double[][] _input;
        private double[] _mean;
        private double[] _stdDev;
        private int _numRows;
        private int _numCols;

        public LayerNormalizationOperation() : base()
        {

        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new LayerNormalizationOperation();
        }

        public double[][] Forward(double[][] input)
        {
            _input = input;
            _numRows = input.Length;
            _numCols = input[0].Length;

            _mean = new double[_numRows];
            _stdDev = new double[_numRows];

            // Compute the mean and standard deviation for each row
            for (int i = 0; i < _numRows; i++)
            {
                _mean[i] = input[i].Average();
                _stdDev[i] = Math.Sqrt(input[i].Select(x => Math.Pow(x - _mean[i], 2)).Sum() / _numCols);
            }

            // Normalize the input
            _output = new double[_numRows][];
            // Parallelize the outer loop
            Parallel.For(0, _numRows, i =>
            {
                _output[i] = new double[_numCols];
                for (int j = 0; j < _numCols; j++)
                {
                    _output[i][j] = (input[i][j] - _mean[i]) / (_stdDev[i] + _epsilon);
                }
            });

            return _output;
        }

        public override (double[][]?, double[][]?) Backward(double[][] gradOutput)
        {
            double[][] gradient = new double[_numRows][];

            // Parallelize the outer loop
            Parallel.For(0, _numRows, i =>
            {
                gradient[i] = new double[_numCols];
                double invStdDev = 1 / (_stdDev[i] + _epsilon);
                double exp1 = (1 - 1.0 / _numCols) * invStdDev;

                for (int j = 0; j < _numCols; j++)
                {
                    var exp2 = Math.Sqrt(Math.Pow(_input[i][j] - _mean[i], 2)) / (_numCols * Math.Pow(_stdDev[i] + _epsilon, 2));
                    var exp3 = exp1 - exp2;
                    // Multiply the computed gradient by the upstream gradient
                    gradient[i][j] = gradOutput[i][j] * exp3;
                }
            });

            return (gradient, gradient);
        }
    }
}