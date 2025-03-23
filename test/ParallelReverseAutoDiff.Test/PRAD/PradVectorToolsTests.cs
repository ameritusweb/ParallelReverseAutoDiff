using ParallelReverseAutoDiff.PRAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace ParallelReverseAutoDiff.Test.PRAD
{
    public class PradVectorToolsTests
    {
        [Fact]
        public void OrderingTest()
        {
            PradVectorTools vectorTools = new PradVectorTools();

            double[] dataA = new double[] { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
            double[] dataZ = new double[] { 0.5, -0.5, 0.5, -0.5, 0.5, -0.5 };
            Tensor mid1 = Tensor.XavierUniform(new int[] { 1, 6 });
            Tensor mid2 = Tensor.XavierUniform(new int[] { 1, 6 });
            Tensor full = new Tensor(new int[] { 4, 6 }, dataA.Concat(mid1.Data).Concat(mid2.Data).Concat(dataZ).ToArray());
            PradOp fullOp = new PradOp(full);

            PradResult result = vectorTools.ComputeOrderingLoss(fullOp);

            Tensor upstream = new Tensor(new int[] { 1, 3 }, 1d);

            result.Back(upstream);
        }

        [Fact]
        public void OrderingTest2()
        {
            PradVectorTools vectorTools = new PradVectorTools();

            double[] dataA = new double[] { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
            double[] dataZ = new double[] { 0.5, 0.5, 0.5, -0.5,-0.5, -0.5 };
            Tensor mid1 = Tensor.XavierUniform(new int[] { 1, 6 });
            Tensor mid2 = Tensor.XavierUniform(new int[] { 1, 6 });
            Tensor full = new Tensor(new int[] { 4, 6 }, dataA.Concat(mid1.Data).Concat(mid2.Data).Concat(dataZ).ToArray());
            PradOp fullOp = new PradOp(full);

            PradResult result = vectorTools.ComputeOrderingLoss2(fullOp);

            Tensor upstream = new Tensor(new int[] { 1, 1 }, 1d);

            result.Back(upstream);
        }

        [Fact]
        public void OrderingTest3()
        {
            PradVectorTools vectorTools = new PradVectorTools();

            double[] dataA = new double[] { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
            double[] dataZ = new double[] { 0.5, 0.5, 0.5, -0.5, -0.5, -0.5 };
            Tensor mid1 = Tensor.XavierUniform(new int[] { 1, 6 });
            Tensor mid2 = Tensor.XavierUniform(new int[] { 1, 6 });
            Tensor mid3 = Tensor.XavierUniform(new int[] { 1, 6 });
            Tensor mid4 = Tensor.XavierUniform(new int[] { 1, 6 });

            Tensor fullmid = new Tensor(new int[] { 4, 6 }, mid1.Data.Concat(mid2.Data).Concat(mid3.Data).Concat(mid4.Data).ToArray());
            PradOp fullMidOp = new PradOp(fullmid);

            Tensor mult1 = Tensor.XavierUniform(new int[] { 3, 6 });
            PradOp mult1Op = new PradOp(mult1);

            Tensor weights = Tensor.XavierUniform(new int[] { 3, 3 });
            PradOp weightsOp = new PradOp(weights);

            double loss = 0d;
            List<double> losses = new List<double>();

            for (int i = 0; i < 400; ++i)
            {

                PradResult multResult = vectorTools.VectorBasedMatrixMultiplication(fullMidOp, mult1Op, weightsOp);
                PradOp multResultOp = multResult.PradOp;
                Tensor multResultT = multResultOp.CurrentTensor;
                double[] data = dataA.Concat(multResultT.Data).Concat(dataZ).ToArray();

                PradOp newMultResult = new PradOp(new Tensor(new int[] { 6, 6 }, data));

                PradResult result = vectorTools.ComputeOrderingLoss2(newMultResult);
                loss = result.Result[0, 0];
                losses.Add(loss);

                Tensor upstream = new Tensor(new int[] { 1, 1 }, 1d);

                result.Back(upstream);

                Tensor newUpstream = new Tensor(new int[] { 4, 6 }, newMultResult.SeedGradient.Data.Skip(6).Take(24).ToArray());
                multResultOp.Back(newUpstream);

                var newOps = vectorTools.GradientDescent(mult1Op, weightsOp);
                mult1Op = newOps[0];
                weightsOp = newOps[1];
                fullMidOp = new PradOp(fullmid);
            }

            File.WriteAllLines("result.txt", losses.Select(x => x.ToString()).ToArray());
        }

        [Fact]
        public void OrderingTest_ScaledUp()
        {
            PradVectorTools vectorTools = new PradVectorTools();
            vectorTools.LearningRate = 0.0002; // You could reduce this slightly if instability shows

            int vectorSize = 48; // increased from 6 to 24
            int numMidVectors = 8; // increased from 4 to 8
            int halfVectorSize = vectorSize / 2;

            // Anchors (A and Z)
            double[] dataA = Enumerable.Repeat(0.5, vectorSize / 2)
                                .Concat(Enumerable.Repeat(0.5, vectorSize / 2)).ToArray();
            double[] dataZ = Enumerable.Repeat(0.5, vectorSize / 2)
                                .Concat(Enumerable.Repeat(-0.5, vectorSize / 2)).ToArray();

            // Middle vectors
            Tensor[] midTensors = Enumerable.Range(0, numMidVectors)
                .Select(_ => Tensor.XavierUniform(new int[] { 1, vectorSize }))
                .ToArray();

            Tensor fullMid = new Tensor(
                new int[] { numMidVectors, vectorSize },
                midTensors.SelectMany(t => t.Data).ToArray()
            );
            PradOp fullMidOp = new PradOp(fullMid);

            // Matrix multiplication parameters: increase complexity
            Tensor mult1 = Tensor.XavierUniform(new int[] { halfVectorSize, vectorSize }); // was 3x6
            PradOp mult1Op = new PradOp(mult1);

            Tensor weights = Tensor.XavierUniform(new int[] { halfVectorSize, halfVectorSize }); // was 3x3
            PradOp weightsOp = new PradOp(weights);

            double loss = 0d;
            List<double> losses = new List<double>();

            List<string> output = new List<string>();

            for (int i = 0; i < 500; ++i)
            {
                // Forward pass
                PradResult multResult = vectorTools.VectorBasedMatrixMultiplication(fullMidOp, mult1Op, weightsOp);
                PradOp multResultOp = multResult.PradOp;
                Tensor multResultT = multResultOp.CurrentTensor;

                // Create complete sequence: A + intermediate + Z
                double[] fullSequence = dataA.Concat(multResultT.Data).Concat(dataZ).ToArray();
                int totalRows = 2 + numMidVectors;
                PradOp sequenceOp = new PradOp(new Tensor(new int[] { totalRows, vectorSize }, fullSequence));

                // Loss computation
                PradResult result = vectorTools.ComputeOrderingLoss2(sequenceOp);
                loss = result.Result[0, 0];
                losses.Add(loss);

                if (i % 10 == 9)
                {
                    output.Add("Iteration " + (i + 1) + ": with loss of: " + loss);
                    output.Add(multResultT.PrintCode());
                }

                // Backprop
                Tensor upstream = new Tensor(new int[] { 1, 1 }, 1d);
                result.Back(upstream);

                int midValuesCount = numMidVectors * vectorSize;
                Tensor newUpstream = new Tensor(new int[] { numMidVectors, vectorSize }, sequenceOp.SeedGradient.Data.Skip(vectorSize).Take(midValuesCount).ToArray());
                multResultOp.Back(newUpstream);

                // Update params
                var updated = vectorTools.GradientDescent(mult1Op, weightsOp);
                mult1Op = updated[0];
                weightsOp = updated[1];

                // Reset mid vector op (not updated directly)
                fullMidOp = new PradOp(fullMid);
            }

            File.WriteAllLines("result_output.txt", output.ToArray());
            File.WriteAllLines("result_scaled.txt", losses.Select(x => x.ToString()).ToArray());
        }

    }
}
