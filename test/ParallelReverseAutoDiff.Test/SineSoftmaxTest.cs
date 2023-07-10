using ParallelReverseAutoDiff.RMAD;
using Xunit;

namespace ParallelReverseAutoDiff.Test
{
    public class SineSoftmaxTest
    {
        [Fact]
        public void GivenASineSoftmaxOperation_TheRowsAddUpToMoreThanPointEightOnAverage()
        {
            SineSoftmaxOperation op = new SineSoftmaxOperation();
            List<double> averages = new List<double>();
            for (int i = 5; i < 1000; ++i)
            {
                Matrix input = new Matrix(i, i);
                input.Initialize(InitializationType.Xavier);
                var output = op.Forward(input);
                Assert.NotNull(output);
                var sums = output.ToArray().ToList().Select(x => x.Sum()).ToList();
                var average = output.ToArray().ToList().Select(x => x.Sum()).Average();
                Assert.True(average > 0.8d);
                averages.Add(average);
            }
        }
    }
}