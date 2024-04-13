// ------------------------------------------------------------------------------
// <copyright file="SpatialNetTrainer.cs" author="ameritusweb" date="12/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.VGruExample
{
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// A gated recurrent network trainer.
    /// </summary>
    public class SpatialNetTrainer
    {
        /// <summary>
        /// Train the network.
        /// </summary>
        /// <returns>A task.</returns>
        public async Task Train()
        {
            try
            {
                CudaBlas.Instance.Initialize();
                int numTimeSteps = 12;
                SpatialNet net = new SpatialNet(numTimeSteps, 11, 110, 2, 0.01d, 14d);
                await net.Initialize();

                SineWaveVectorGenerator vectorGenerator = new SineWaveVectorGenerator(1d, 1d, 0d);
                Random random = new Random(2);

                // net.ApplyWeights();
                for (int i = 0; i < 1000; ++i)
                {
                    int samples = random.Next(101, 201);
                    int cycles = random.Next((int)(samples / 20d), (int)(samples / 5d));
                    var vectors = vectorGenerator.GenerateWaveWithVectors(samples, cycles);
                    var sectionCount = 0;

                    for (int j = 0; j < (samples - numTimeSteps - 1); j += numTimeSteps)
                    {
                        sectionCount++;
                        var matrices = new Matrix[numTimeSteps];
                        for (int k = 0; k < numTimeSteps; ++k)
                        {
                            matrices[k] = VectorToMatrix.CreateLine(vectors[j + k].Vector.Direction, 11);
                        }

                        var input = new DeepMatrix(matrices);
                        var lastVector = vectors[j + numTimeSteps].Vector;

                        var (gradient, output, loss) = net.Forward(input, lastVector.Direction, lastVector.Magnitude);

                        double x = output[0, 0];
                        double y = output[0, 1];
                        double resultMagnitude = Math.Sqrt((x * x) + (y * y));
                        double resultAngle = Math.Atan2(y, x);
                        double oLoss = loss[0, 0];

                        Console.WriteLine($"Iteration {i}_{sectionCount}, Loss: {loss[0, 0]}, Result: [{resultMagnitude}, {resultAngle}], Last: [{lastVector.Magnitude}, {lastVector.Direction}]");

                        var inputGradient = await net.Backward(gradient);

                        net.ApplyGradients();

                        await net.Reset();

                        Thread.Sleep(1000);
                        if (i % 4 == 0 && j == 0)
                        {
                            net.SaveWeights();
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
            finally
            {
                CudaBlas.Instance.Dispose();
            }
        }
    }
}
