//------------------------------------------------------------------------------
// <copyright file="LeakyReLUOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.RMAD
{
    public class LeakyReLUOperation : Operation
    {
        private Matrix input;
        private double alpha;

        public LeakyReLUOperation(double alpha = 0.01)
            : base()
        {
            this.alpha = alpha;
        }

        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new LeakyReLUOperation();
        }

        public Matrix Forward(Matrix input)
        {
            this.input = input;
            int rows = input.Length;
            int cols = input[0].Length;
            this.output = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double x = input[i][j];
                    this.output[i][j] = x > 0 ? x : this.alpha * x;
                }
            }

            return this.output;
        }

        public override (Matrix?, Matrix?) Backward(Matrix dLdOutput)
        {
            int rows = dLdOutput.Length;
            int cols = dLdOutput[0].Length;
            Matrix dLdInput = new Matrix(rows, cols);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    double x = this.input[i][j];
                    double gradient = x > 0 ? 1.0 : this.alpha;
                    dLdInput[i][j] = dLdOutput[i][j] * gradient;
                }
            }

            return (dLdInput, dLdInput);
        }
    }
}
