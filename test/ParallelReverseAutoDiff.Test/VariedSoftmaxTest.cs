using ParallelReverseAutoDiff.RMAD;
using Xunit;

namespace ParallelReverseAutoDiff.Test
{
    public class VariedSoftmaxTest
    {
        [Fact]
        public void GivenAVariesSoftmaxOperation_SuccessfullyStaysBetween0And1()
        {
            VariedSoftmaxOperation op = new VariedSoftmaxOperation();
            Matrix test = new Matrix(1, 6);
            test[0] = new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };
            Matrix temp = new Matrix(1, 1);
            temp[0, 0] = 0.01d;
            var res = op.Forward(test, temp);
            for (int i = 0; i < res[0].Length; ++i)
            {
                res[0, i] *= res[0, i];
            }
            SigmoidOperation sig = new SigmoidOperation();
            var sig1 = sig.Forward(res);
        }
    }
}