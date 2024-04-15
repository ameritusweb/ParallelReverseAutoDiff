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
                Random random = new Random(6);

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

                        if (targetAngles.Count > 9)
                        {
                            targetAngles.RemoveAt(0);
                        }

                        targetAngles.Add(lastVector.Direction);

                        var (gradient, output, loss) = net.Forward(input, lastVector.Direction, lastVector.Magnitude);

                        var compare = VectorToMatrix.CreateLine(lastVector.Direction, 11);
                        double x = output[0, 0];
                        double y = output[0, 1];
                        double resultMagnitude = Math.Sqrt((x * x) + (y * y));
                        double resultAngle = Math.Atan2(y, x);
                        double oLoss = loss[0, 0];

                        Matrix magMatrix = new Matrix(11, 1);
                        Matrix mMatrix = new Matrix(11, 11);
                        for (int k = 0; k < 11; ++k)
                        {
                            for (int l = 0; l < 1; ++l)
                            {
                                magMatrix[k, l] = output[k].Sum();
                            }

                            for (int l = 0; l < 11; ++l)
                            {
                                mMatrix[k, l] = output[k, l];
                            }
                        }

                        if (outputMatrices.Count > 9)
                        {
                            outputMatrices.RemoveAt(0);
                        }

                        outputMatrices.Add(magMatrix);

                        if (fullOutputMatrices.Count > 9)
                        {
                            fullOutputMatrices.RemoveAt(0);
                        }

                        fullOutputMatrices.Add(mMatrix);

                        Matrix? cGradient = null;
                        double cLoss = 0d;

                        if (outputMatrices.Count >= 10)
                        {
                            if (correlations.Count > 9)
                            {
                                correlations.RemoveAt(0);
                            }

                            var cc1 = CorrelationCalculator.PearsonCorrelationLoss(outputMatrices.ToArray(), targetAngles.ToArray());
                            var cc = cc1.Correlations;

                            var cc2 = CorrelationCalculator.PearsonCorrelationLoss(fullOutputMatrices.ToArray(), targetAngles.ToArray());
                            cGradient = cc2.Gradient;
                            cLoss = cc2.Loss;

                            correlations.Add(cc);

                            double max = 0d;
                            int r = 0;
                            int c = 0;
                            double totalCC = 0d;
                            double min = 1000d;
                            int rMin = 0;
                            int cMin = 0;
                            for (int k = 0; k < 11; ++k)
                            {
                                for (int l = 0; l < 1; ++l)
                                {
                                    if (cc[k, l] > max)
                                    {
                                        max = cc[k, l];
                                        r = k;
                                        c = l;
                                    }

                                    if (cc[k, l] < min)
                                    {
                                        min = cc[k, l];
                                        rMin = k;
                                        cMin = l;
                                    }

                                    if (cc[k, l] >= 0.4d || cc[k, l] <= -0.4d)
                                    {
                                        Console.WriteLine($"Correlation {k} {l}: {cc[k, l]}");
                                    }

                                    totalCC += cc[k, l];
                                }
                            }

                            double avgCC = totalCC / 11d;
                            Console.WriteLine($"Average Correlation: {avgCC}");

                            Console.WriteLine($"Max Correlation: {max} {r} {c}");

                            Console.WriteLine($"Min Correlation: {min} {rMin} {cMin}");
                        }

                        Matrix diffMatrix = new Matrix(11, 22);
                        for (int k = 0; k < 11; ++k)
                        {
                            for (int l = 0; l < 22; ++l)
                            {
                                diffMatrix[k, l] = output[k, l] - lastMatrix[k, l];
                            }
                        }

                        Console.WriteLine($"Iteration {i}_{sectionCount}, Loss: {cLoss}, Result: [{resultMagnitude}, {resultAngle}], Last: [{lastVector.Magnitude}, {lastVector.Direction}]");

                        lastMatrix.Replace(output.ToArray());

                        // var inputGradient = await net.Backward(gradient, cGradient);

                        // net.ApplyGradients();
                        await net.Reset();

                        Thread.Sleep(1000);
                        if (i % 4 == 3 && j == 0)
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
