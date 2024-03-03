using Newtonsoft.Json;
using ParallelReverseAutoDiff.GravNetExample.Common;
using ParallelReverseAutoDiff.RMAD;
using System.Numerics;

namespace ParallelReverseAutoDiff.GravNetExample
{
    public class TiledNetTrainer
    {
        public async Task Train()
        {
            try
            {
                CudaBlas.Instance.Initialize();
                TiledNet net = new TiledNet(512, 6144, 3, 0.01d, 4d);
                await net.Initialize();
                //net.ApplyWeights();

                var pngFiles = Directory.GetFiles(@"E:\images\inputs\svg", "*.png");

                Random random = new Random(8);
                var files = pngFiles.OrderBy(x => random.Next()).ToArray();
                uint i = 0;
                await files.WithRepeatAsync(async (pngFile, token) =>
                {

                    if (!pngFile.Contains("_"))
                    {
                        token.Continue();
                        return;
                    }

                    Node[,] nodes = ImageSerializer.DeserializeImageWithAntiAlias(pngFile);
                    Node[,] interpolated = ImageSerializer.BilinearInterpolation(nodes, 512, 512);
                    List<List<double>> data = new List<List<double>>();
                    int count = 0;
                    int totalCount = 0;
                    double[,] values = new double[512, 512];
                    for (int i = 0; i < 512; ++i)
                    {
                        var subvect = new List<double>();
                        for (int j = 0; j < 512; ++j)
                        {
                            var value = (256d - interpolated[i, j].GrayValue) / (255d);
                            if (value > 0.5d)
                            {
                                count++;
                            }
                            values[i, j] = value;
                            subvect.Add(value);
                            totalCount++;
                        }
                        data.Add(subvect);
                    }
                    var perc = (double)count / (double)totalCount * 100d;

                    var file = pngFile.Substring(pngFile.LastIndexOf('\\') + 1);
                    int uIndex = file.IndexOf("_");
                    var prefix = file.Substring(0, uIndex);

                    var percentages = CalculatePercentagesAboveThreshold(values);

                    //var glyphFile = pngFile.Replace("\\" + prefix, "\\" + prefix + "_glyph").Replace("svg\\", "svg-glyph\\");
                    //Node[,] glyphNodes = ImageSerializer.DeserializeImageWithoutAntiAlias(glyphFile);
                    //Matrix rotationTargets = new Matrix(15, 15);
                    //Vector3[] glyphs = new Vector3[225];
                    //int m = 0;
                    //for (int k = 0; k < 15; ++k)
                    //{
                    //    for (int l = 0; l < 15; ++l)
                    //    {
                    //        rotationTargets[k, l] = glyphNodes[k, l].IsForeground ? 1d : 0d;
                    //        glyphs[m] = new Vector3(0f, 0f, (float)rotationTargets[k, l]);
                    //        m++;
                    //    }
                    //}

                    Matrix matrix = new Matrix(data.Count, data[0].Count);
                    for (int j = 0; j < data.Count; j++)
                    {
                        for (int k = 0; k < data[0].Count; k++)
                        {
                            matrix[j, k] = data[j][k];
                        }
                    }

                    i++;

                    var res = net.Forward(matrix, percentages);
                    var gradient = res.Item1;
                    var output = res.Item2;
                    var loss = res.Item3;
                    //var gradient0 = res.Item2;
                    //var gradient1 = res.Item3;
                    //var output = res.Item4;
                    //var o0 = res.Item5;
                    //var o1 = res.Item6;
                    //var glyph = res.Item7;
                    //var loss = res.Item8;
                    //var loss0 = res.Item9;
                    //var loss1 = res.Item10;

                    //for (int j = 0; j < 225; ++j)
                    //{
                    //    glyphs[j].X = (float)glyph[j, 0];
                    //    glyphs[j].Y = (float)glyph[j, 1];
                    //}


                    Console.WriteLine($"Iteration {i} Output X: {output[0, 0]}, Output Y: {output[0, 1]}, Grad: {gradient[0, 0]}, {gradient[0, 1]}");
                    Console.WriteLine($"Loss: {loss[0, 0]}, Perc: {perc}");
                    //Console.WriteLine($"O1 X: {o1[0, 0]}, O1 Y: {o1[0, 1]}, Loss: {loss[0, 0]}, {loss0[0, 0]}, {loss1[0, 0]}");
                    await net.Backward(gradient);
                    net.ApplyGradients();
                    //}

                    await net.Reset();
                    Thread.Sleep(1000);
                    if (i % 11 == 10)
                    {
                        net.SaveWeights();
                    }

                    //if (token.UsageCount == 0)
                    //{
                    //    token.Repeat(2);
                    //}
                });
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            } finally
            {
                CudaBlas.Instance.Dispose();
            }
        }

        private double[,] CalculatePercentagesAboveThreshold(double[,] input, double threshold = 0.5)
        {
            int inputWidth = input.GetLength(0); // 512
            int inputHeight = input.GetLength(1); // 512
            int sectionSize = inputWidth / 8; // Assuming the input is always 512x512 and output grid is 8x8

            double[,] percentages = new double[8, 8];

            for (int sectionX = 0; sectionX < 8; sectionX++)
            {
                for (int sectionY = 0; sectionY < 8; sectionY++)
                {
                    int aboveThresholdCount = 0;
                    for (int x = sectionX * sectionSize; x < (sectionX + 1) * sectionSize; x++)
                    {
                        for (int y = sectionY * sectionSize; y < (sectionY + 1) * sectionSize; y++)
                        {
                            if (input[x, y] > threshold)
                            {
                                aboveThresholdCount++;
                            }
                        }
                    }

                    double totalValues = sectionSize * sectionSize;
                    percentages[sectionX, sectionY] = (double)aboveThresholdCount / totalValues;
                }
            }

            return percentages;
        }
    }
}
