using ParallelReverseAutoDiff.RMAD;
using ParallelReverseAutoDiff.Test.Convolutional;
using ParallelReverseAutoDiff.Test.ForwardMode;
using Xunit;

namespace ParallelReverseAutoDiff.Test
{
    public class LayerNormalizationTest
    {

        private const double EPSILON = 1e-9;

        [Theory]
        [InlineData(22, 22)]
        [InlineData(33, 33)]
        [InlineData(44, 44)]
        [InlineData(222, 222)]
        [InlineData(333, 333)]
        [InlineData(444, 444)]
        public async Task GivenLayerNormalization_TheBackwardGradientMatchesTheDualNumberGradient(int numRows, int numCols)
        {
            Random rand = new Random(Guid.NewGuid().GetHashCode());
            LayerNormalizationOperation layerNorm = new LayerNormalizationOperation();
            var input = new DualNumberMatrix(numRows, numCols);
            var upstreamGradient = new Matrix(numRows, numCols);  // fill this with your random values

            for (int row = 0; row < numRows; row++)
            {
                for (int col = 0; col < numCols; col++)
                {
                    upstreamGradient[row][col] = rand.NextDouble() * 10d;
                    // Initialize the input matrix with real values
                    input[row, col] = new DualNumber((row * 2d + 1) * (col * 2d + 5) * rand.NextDouble(), 0.0);  // replace with your own values
                }
            }

            for (int inputRow = 0; inputRow < numRows; inputRow++)
            {
                for (int inputCol = 0; inputCol < numCols; inputCol++)
                {
                    // Reset the dual parts of the input matrix
                    for (int row = 0; row < numRows; row++)
                    {
                        for (int col = 0; col < numCols; col++)
                        {
                            input[row, col].Dual = 0.0;
                        }
                    }

                    // Set the dual part of the current input element to 1
                    input[inputRow, inputCol].Dual = 1.0;

                    // Execution
                    var forwardOutput = layerNorm.Forward(input, EPSILON);

                    for (int i = 0; i < forwardOutput.Length; i++)
                    {
                        for (int j = 0; j < forwardOutput[0].Length; j++)
                        {
                            forwardOutput[i, j].Dual *= upstreamGradient[i, j];
                        }
                    }

                    layerNorm.Forward(input.ToMatrix());

                    // Compute the expected output using the backward function
                    var output = layerNorm.Backward(upstreamGradient); /* the result of your backward function given the same input and upstreamGradient */
                    var expectedOutput = output.Item1 as Matrix;

                    // Validation
                    double diff = Math.Abs(expectedOutput[inputRow, inputCol] - forwardOutput[inputRow, inputCol].Dual);
                    Assert.True(diff < EPSILON);
                }

                if (numRows > 100 || numCols > 100)
                {
                    break;
                }
            }
        }
    }
}