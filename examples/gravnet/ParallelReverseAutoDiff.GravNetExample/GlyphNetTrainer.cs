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

                Random random = new Random(1);
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

                    var glyphFile = pngFile.Replace(prefix, prefix + "_glyph").Replace("svg\\", "svg-glyph\\");
                    Node[,] glyphNodes = ImageSerializer.DeserializeImageWithoutAntiAlias(glyphFile);
                    Matrix rotationTargets = new Matrix(15, 15);
                    Vector3[] glyphs = new Vector3[225];
                    int m = 0;
                    for (int k = 0; k < 15; ++k)
                    {
                        for (int l = 0; l < 15; ++l)
                        {
                            rotationTargets[k, l] = glyphNodes[k, l].IsForeground ? 1d : 0d;
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

                    var res = net.Forward(matrix, rotationTargets, Math.PI / 4d);
                    var gradient = res.Item1;
                    var output = res.Item2;
                    var glyph = res.Item3;
                    var loss = res.Item4;

                    for (int j = 0; j < 225; ++j)
                    {
                        glyphs[j].X = (float)glyph[j, 0];
                        glyphs[j].Y = (float)glyph[j, 1];
                    }


                    Console.WriteLine($"Iteration {i} Output X: {res.Item2[0, 0]}, Output Y: {res.Item2[0, 1]}, Grad: {res.Item1[0, 0]}, {res.Item1[0, 1]}");

                    await net.Backward(gradient);
                    net.ApplyGradients();
                    //}

                    await net.Reset();
                    Thread.Sleep(1000);
                    if (i % 11 == 2)
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
    }
}
