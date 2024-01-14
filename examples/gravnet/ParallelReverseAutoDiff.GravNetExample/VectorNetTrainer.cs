using Newtonsoft.Json;
using ParallelReverseAutoDiff.RMAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.GravNetExample
{
    public class VectorNetTrainer
    {
        public async Task Train()
        {
            try
            {
                CudaBlas.Instance.Initialize();
                VectorNet net = new VectorNet(17, 446, 3, 0.0002d, 4d);
                await net.Initialize();
                // net.ApplyWeights();

                var jsonFiles = Directory.GetFiles(@"E:\images\inputs\ocr", "*.json");

                Random random = new Random(Guid.NewGuid().GetHashCode());
                var files = jsonFiles.OrderBy(x => random.Next()).ToArray();
                int i = 0;
                foreach (var jsonFile in files)
                {
                   
                    var json = File.ReadAllText(jsonFile);
                    var data = JsonConvert.DeserializeObject<List<List<double>>>(json);
                    var file = jsonFile.Substring(jsonFile.LastIndexOf('\\') + 1);
                    var sub = file.Substring(16, 1);

                    if (sub != "A" && sub != "B")
                    {
                        continue;
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

                    double targetAngle = sub == "A" ? Math.PI / 4d : ((Math.PI / 2) + (Math.PI / 4));
                    var res = net.Forward(matrix, targetAngle);
                    var gradient = res.Item1;
                    var output = res.Item2;
                    var loss = res.Item3;
                    var x = output[0][0];
                    var y = output[0][1];
                    double resultMagnitude = Math.Sqrt((x * x) + (y * y));
                    double resultAngle = Math.Atan2(y, x);

                    Console.WriteLine($"Iteration {i} {sub} Mag: {resultMagnitude}, Angle: {resultAngle}, TargetAngle: {targetAngle}, Gradient: {gradient[0][0]}, {gradient[0][1]} Loss: {loss[0][0]}");

                    await net.Backward(gradient);
                    net.ApplyGradients();
                    await net.Reset();
                    Thread.Sleep(1000);
                    if (i % 25 == 24)
                    {
                        net.SaveWeights();
                    }
                }
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
