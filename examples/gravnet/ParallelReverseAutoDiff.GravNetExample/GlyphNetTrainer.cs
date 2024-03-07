using Newtonsoft.Json;
using ParallelReverseAutoDiff.GravNetExample.Common;
using ParallelReverseAutoDiff.RMAD;
using System.Numerics;

namespace ParallelReverseAutoDiff.GravNetExample
{
    public class GlyphNetTrainer
    {
        public async Task Train()
        {
            try
            {
                CudaBlas.Instance.Initialize();
                GlyphNet net = new GlyphNet(512, 6144, 3, 0.01d, 4d);
                await net.Initialize();
                //net.ApplyWeights();

                var pngFiles = Directory.GetFiles(@"E:\images\inputs\svg", "*.png");

                Random random = new Random(11);
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
                    for (int i = 0; i < 512; ++i)
                    {
                        var subvect = new List<double>();
                        for (int j = 0; j < 512; ++j)
                        {
                            subvect.Add((256d - interpolated[i, j].GrayValue) / (255d));
                        }
                        data.Add(subvect);
                    }

                    var file = pngFile.Substring(pngFile.LastIndexOf('\\') + 1);
                    int uIndex = file.IndexOf("_");
                    var prefix = file.Substring(0, uIndex);

                    //var glyphFile = pngFile.Replace("\\" + prefix, "\\" + prefix + "_glyph").Replace("svg\\", "svg-glyph\\");
                    //Node[,] glyphNodes = ImageSerializer.DeserializeImageWithoutAntiAlias(glyphFile);

                    var glyphNodes = AnalyzeImageSections(interpolated);
                    
                    Matrix rotationTargets = new Matrix(8, 8);
                    Vector3[] glyphs = new Vector3[64];
                    int m = 0;
                    for (int k = 0; k < 8; ++k)
                    {
                        for (int l = 0; l < 8; ++l)
                        {
                            rotationTargets[k, l] = glyphNodes[k, l];
                            glyphs[m] = new Vector3(0f, 0f, (float)rotationTargets[k, l]);
                            m++;
                        }
                    }

                    Matrix matrix = new Matrix(data.Count, data[0].Count);
                    for (int j = 0; j < data.Count; j++)
                    {
                        for (int k = 0; k < data[0].Count; k++)
                        {
                            matrix[j, k] = data[j][k];
                        }
                    }

                    i++;

                    var res = net.Forward(matrix, rotationTargets);
                    var glyph = res.Item1;
                    var gradients = res.Item2;
                    //var gradient1 = res.Item3;
                    //var output = res.Item4;
                    //var o0 = res.Item5;
                    //var o1 = res.Item6;
                    //var glyph = res.Item7;
                    //var loss = res.Item8;
                    //var loss0 = res.Item9;
                    //var loss1 = res.Item10;

                    for (int j = 0; j < 64; ++j)
                    {
                        glyphs[j].X = (float)glyph[j, 0];
                        glyphs[j].Y = (float)glyph[j, 1];
                    }


                    //Console.WriteLine($"Iteration {i} Output X: {output[0, 0]}, Output Y: {output[0, 1]}, Grad: {gradient[0, 0]}, {gradient[0, 1]}");
                    Console.WriteLine($"Iteration {i} Glyph: {glyphs[0].X}, {glyphs[0].Y}");
                    Console.WriteLine($"Loss: {gradients.Sum(x => x.Loss[0, 0])}");
                    //Console.WriteLine($"O0 X: {o0[0, 0]}, O0 Y: {o0[0, 1]}, Loss 0: {loss0[0, 0]}");
                    //Console.WriteLine($"O1 X: {o1[0, 0]}, O1 Y: {o1[0, 1]}, Loss 1: {loss1[0, 0]}");
                    await net.Backward(gradients);
                    net.ApplyGradients();
                    //}

                    await net.Reset();
                    Thread.Sleep(1000);
                    if (i % 11 == 5)
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

        private int[,] AnalyzeImageSections(Node[,] interpolated)
        {
            // Assuming Node is a custom type with a GrayValue property.
            // Size of the output matrix.
            const int outputSize = 8;
            // Calculate section size based on the input image dimensions and desired output matrix size.
            int sectionWidth = interpolated.GetLength(0) / outputSize;
            int sectionHeight = interpolated.GetLength(1) / outputSize;

            // Initialize the output matrix.
            int[,] sectionAnalysis = new int[outputSize, outputSize];

            // Process each section.
            for (int sectionX = 0; sectionX < outputSize; sectionX++)
            {
                for (int sectionY = 0; sectionY < outputSize; sectionY++)
                {
                    // Calculate the start and end indices for the section.
                    int startX = sectionX * sectionWidth;
                    int startY = sectionY * sectionHeight;
                    int endX = startX + sectionWidth;
                    int endY = startY + sectionHeight;

                    // Flatten the section into a single collection for easier analysis.
                    var sectionPixels = Enumerable.Range(startX, sectionWidth).SelectMany(
                        x => Enumerable.Range(startY, sectionHeight),
                        (x, y) => (256d - interpolated[x, y].GrayValue) / (255d));

                    // Determine if 40% or more of the pixels in the section are greater than 0.5.
                    sectionAnalysis[sectionX, sectionY] = sectionPixels.Count(val => val > 0.5) >= sectionPixels.Count() * 0.4 ? 1 : 0;
                }
            }

            return sectionAnalysis;
        }
    }
}
