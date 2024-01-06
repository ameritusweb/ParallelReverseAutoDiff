using ParallelReverseAutoDiff.RMAD;
using Xunit;

namespace ParallelReverseAutoDiff.Test
{
    public class IterativeSoftmaxTest
    {
        [Fact]
        public void GivenAnIterativeSoftmaxOperation_PerformsSuccessfullyOnTheGpu()
        {
            try
            {
                CudaBlas.Instance.Initialize();
                Thread.Sleep(1000);
                VariedMaskedIterativeSoftmaxOperation op = new VariedMaskedIterativeSoftmaxOperation();
                Matrix input = new Matrix(1, 10000);
                input.Initialize(InitializationType.Xavier);
                LeakyReLUOperation relu = new LeakyReLUOperation(0.1d);
                input = relu.Forward(input);
                PiecewiseActivationOperation act = new PiecewiseActivationOperation();
                input = act.Forward(input);
                input = input * 10d;
                VariedSoftmaxOperation softmaxOp = new VariedSoftmaxOperation();
                Matrix beforeTemp = new Matrix(1, 10000);
                beforeTemp[0][0] = 0.01d;
                var previousSoftmax = softmaxOp.Forward(input, beforeTemp);
                Matrix temp = new Matrix(1, 10000);
                temp[0][0] = 0.001d;
                
                var res = op.Forward(input, temp, previousSoftmax, 0.01d);
                Matrix dOutput = new Matrix(1, 10000);
                dOutput.Initialize(InitializationType.Xavier);
                var backwardRest = op.Backward(dOutput);
                Assert.True(res[0][0] > 0.0d);
            } catch (Exception e)
            {
                Assert.True(false, e.Message);
            } finally
            {
                CudaBlas.Instance.Dispose();
            }
        }
    }
}