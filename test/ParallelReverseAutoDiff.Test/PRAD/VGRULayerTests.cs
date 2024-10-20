using Emgu.CV.Shape;
using ParallelReverseAutoDiff.PRAD;
using ParallelReverseAutoDiff.PRAD.Layers;
using Xunit;

namespace ParallelReverseAutoDiff.Test.PRAD
{
    public class VGRULayerTests
    {
        [Fact]
        public void TestLayer()
        {
            int[] shape = new int[] { 100, 200 };
            PradOp opInput1 = new PradOp(Tensor.XavierUniform(shape));
            PradOp opAngles = new PradOp(Tensor.XavierUniform(shape));
            int[] shapeD1 = new int[] { shape[0], shape[1] * 6 };
            int[] shapeD2 = new int[] { shape[0], shape[1] };
            PradOp opD1 = new PradOp(Tensor.XavierUniform(shapeD1));
            PradOp opD2 = new PradOp(Tensor.XavierUniform(shapeD2));
            int[] shapeP = new int[] { shape[0], shape[1] * 12 };
            PradOp previousHiddenState = new PradOp(Tensor.XavierUniform(shapeP));
            int[] shapeGate = new int[] { shape[1] * 12, shape[1] * 12 };
            int[] shapeB = new int[] { shape[0], shape[1] * 12 };
            int[] shapeV = new int[] { shape[0], shape[1] * 12 };
            int[] shapeW = new int[] { shape[0], shape[1] * 6 };
            PradOp[][] updateWeights = GateWeights(shapeGate, shapeV, shapeW, shapeB);
            PradOp[][] resetWeights = GateWeights(shapeGate, shapeV, shapeW, shapeB);
            PradOp[][] candidateWeights = new PradOp[2][];
            for (int i = 0; i < 2; ++i)
            {
                candidateWeights[i] = new PradOp[4];
                candidateWeights[i][0] = new PradOp(Tensor.XavierUniform(shapeW));
                candidateWeights[i][1] = new PradOp(Tensor.XavierUniform(shapeGate));
                candidateWeights[i][2] = new PradOp(Tensor.XavierUniform(shapeB));
                candidateWeights[i][3] = new PradOp(Tensor.XavierUniform(shapeW));
            }

            PradOp[][] hiddenWeights = new PradOp[2][];
            for (int i = 0; i < 2; ++i)
            {
                hiddenWeights[i] = new PradOp[1];
                hiddenWeights[i][0] = new PradOp(Tensor.XavierUniform(shapeW));
            }
            PradOp convolutionFilter = new PradOp(Tensor.XavierUniform(new int[] { 1, 2, 2, 2 }));

            VGRULayer layer = new VGRULayer(opInput1, opAngles, opD1, opD2, previousHiddenState, updateWeights, resetWeights, candidateWeights, hiddenWeights, convolutionFilter);
            var computeResult = layer.Compute();

            var upstream = new Tensor(new int[] { 4 }, 1d);

            computeResult.Back(upstream);
        
        }

        private PradOp[][] GateWeights(int[] shapeGate, int[] shapeV, int[] shapeW, int[] shapeB)
        {
            PradOp[][] updateWeights = new PradOp[2][];
            for (int i = 0; i < 2; ++i)
            {
                updateWeights[i] = new PradOp[7];
                updateWeights[i][0] = new PradOp(Tensor.XavierUniform(shapeW));
                updateWeights[i][1] = new PradOp(Tensor.XavierUniform(shapeW));
                updateWeights[i][2] = new PradOp(Tensor.XavierUniform(shapeV));
                updateWeights[i][3] = new PradOp(Tensor.XavierUniform(shapeV));
                updateWeights[i][4] = new PradOp(Tensor.XavierUniform(shapeW));
                updateWeights[i][5] = new PradOp(Tensor.XavierUniform(shapeGate));
                updateWeights[i][6] = new PradOp(Tensor.XavierUniform(shapeB));
            }

            return updateWeights;
        }
    }
}
