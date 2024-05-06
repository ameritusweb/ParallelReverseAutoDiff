// ------------------------------------------------------------------------------
// <copyright file="MusicNetTrainer.cs" author="ameritusweb" date="12/18/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

namespace ParallelReverseAutoDiff.VGruExample
{
    using ParallelReverseAutoDiff.RMAD;
    using ParallelReverseAutoDiff.VGruExample.VGruNetwork;
    using ParallelReverseAutoDiff.VGruExample.VGruNetwork.RMAD;

    /// <summary>
    /// A music network trainer.
    /// </summary>
    public class MusicNetTrainer
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
                int numTimeSteps = 6;
                MazeNet net = new MazeNet(81, 11, 110, 2, 0.001d, 4d);
                await net.Initialize();

                SineWaveVectorGenerator vectorGenerator = new SineWaveVectorGenerator(1d, 1d, 0d);
                Random random = new Random(12);

                Matrix lastMatrix = new Matrix(11, 22);
                List<Matrix> outputMatrices = new List<Matrix>();
                List<Matrix> fullOutputMatrices = new List<Matrix>();
                List<double> targetAngles = new List<double>();
                List<double[,]> correlations = new List<double[,]>();

                net.ApplyWeights();
                for (int i = 0; i < 1000; ++i)
                {
                    int samples = random.Next(101, 201);
                    int cycles = random.Next((int)(samples / 20d), (int)(samples / 5d));
                    var vectors = vectorGenerator.GenerateWaveWithVectors(samples, cycles);
                    var sectionCount = 0;
                    double lastTargetAngle = double.MinValue;

                    for (int j = 0; j < (samples - numTimeSteps - 1); j += numTimeSteps)
                    {
                        sectionCount++;
                        var matrices = new Matrix[numTimeSteps];
                        for (int k = 0; k < numTimeSteps; ++k)
                        {
                            matrices[k] = VectorToMatrix.CreateLine(vectors[j + k].Vector.Direction, 11);
                            if (k > 0)
                            {
                                targetAngles.Add((3d * Math.PI / 4d) / (1 + Math.Exp(-vectors[j + k].Vector.Direction)));
                            }
                        }

                        var input = new DeepMatrix(matrices);
                        var lastVector = vectors[j + numTimeSteps].Vector;
                        var targetAngle = (3d * Math.PI / 4d) / (1 + Math.Exp(-lastVector.Direction));
                        targetAngles.Add(targetAngle);

                        Console.WriteLine($"Iteration {i}_{sectionCount}, Target Angle: {targetAngle}");

                        await net.Train(input, targetAngles.ToArray());

                        // var (gradient, output, loss) = net.Forward(input, targetAngle, lastVector.Magnitude);

                        // Console.WriteLine($"Iteration {i}_{sectionCount}, Loss: {loss[0, 0]}, Last: [{lastVector.Magnitude}, {lastVector.Direction}]");

                        // var inputGradient = await net.Backward(gradient);
                        // net.ApplyGradients();
                        lastTargetAngle = lastVector.Direction;

                        // net.ApplyGradients();
                        await net.Reset();

                        targetAngles.Clear();

                        Thread.Sleep(1000);

                        if (sectionCount % 4 == 3)
                        {
                            // net.SaveWeights();
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
