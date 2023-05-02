namespace ParallelReverseAutoDiff.RMAD
{
    using System;
    using System.Threading.Tasks;

    public class MatrixMultiplyOperation : Operation
    {
        private double[][] _input1;
        private double[][] _input2;
        private int _weightsToUpdateIndex;

        public MatrixMultiplyOperation(int weightsToUpdateIndex = -1) : base()
        {
            _weightsToUpdateIndex = weightsToUpdateIndex;
        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new MatrixMultiplyOperation();
        }

        public double[][] Forward(double[][] input1, double[][] input2)
        {
            _input1 = input1;
            _input2 = input2;
            int input1Rows = input1.Length;
            int input1Cols = input1[0].Length;
            int input2Rows = _input2.Length;
            int input2Cols = _input2[0].Length;

            if (input1Cols != input2Rows)
            {
                throw new Exception("Input 1 columns do not match Input 2 rows");
            }

            _output = new double[input1Rows][];
            // Parallelize the outer loop
            Parallel.For(0, input1Rows, i =>
            {
                _output[i] = new double[input2Cols];
                for (int j = 0; j < input2Cols; j++)
                {
                    _output[i][j] = 0;
                    for (int k = 0; k < input1Cols; k++)
                    {
                        _output[i][j] += input1[i][k] * input2[k][j];
                        if (double.IsNaN(_output[i][j]))
                        {

                        }
                    }
                }
            });

            return _output;
        }

        public override (double[][]?, double[][]?) Backward(double[][] dOutput)
        {
            int input1Rows = _input1.Length;
            int input1Cols = _input1[0].Length;
            int input2Rows = _input2.Length;
            int input2Cols = _input2[0].Length;

            // Calculate gradient w.r.t. input1
            double[][] dInput1 = new double[input1Rows][];
            // Parallelize the outer loop
            Parallel.For(0, input1Rows, i =>
            {
                dInput1[i] = new double[input1Cols];
                for (int j = 0; j < input1Cols; j++)
                {
                    dInput1[i][j] = 0;
                    for (int k = 0; k < input2Cols; k++)
                    {
                        dInput1[i][j] += dOutput[i][k] * _input2[j][k];
                    }
                }
            });

            // Calculate gradient w.r.t. input2
            double[][] dInput2 = new double[input2Rows][];
            // Parallelize the outer loop
            Parallel.For(0, input2Rows, i =>
            {
                dInput2[i] = new double[input2Cols];
                for (int j = 0; j < input2Cols; j++)
                {
                    dInput2[i][j] = 0;
                    for (int k = 0; k < input1Rows; k++)
                    {
                        dInput2[i][j] += _input1[k][i] * dOutput[k][j];
                    }
                }
            });

            if (_weightsToUpdateIndex == 0)
            {
                // Update weights
                double learningRate = 0.01;
                for (int i = 0; i < input1Rows; i++)
                {
                    for (int j = 0; j < input1Cols; j++)
                    {
                        _input1[i][j] -= learningRate * dInput1[i][j];
                    }
                }
            }
            else if (_weightsToUpdateIndex == 1)
            {
                // Update weights
                double learningRate = 0.01;
                for (int i = 0; i < input2Rows; i++)
                {
                    for (int j = 0; j < input2Cols; j++)
                    {
                        _input2[i][j] -= learningRate * dInput2[i][j];
                    }
                }
            }

            return (dInput1, dInput2);
        }
    }

}
