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
                VectorNet net = new VectorNet(17, 446, 3, 0.0001d, 4d);
                await net.Initialize();
                net.ApplyWeights();

                var jsonFiles = Directory.GetFiles(@"E:\images\inputs\ocr", "*.json");

                double sumResultAngleA = 0d;
                double numResultAngleA = 0d;
                double sumResultAngleB = 0d;
                double numResultAngleB = 0d;
                double sumLoss = 0d;
                double numLoss = 0d;
                Random random = new Random(11);
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
                    var layer = net.GetModelLayer();
                    if (layer != null)
                    {
                        //var angles = layer.WeightMatrix("SummationWeights");
                        //angles[0, 0] += 0.001d;
                    }

                    i++;

                    double targetAngle = sub == "A" ? Math.PI / 4d : ((Math.PI / 2) + (Math.PI / 4));
                    double oppositeAngle = sub == "A" ? ((Math.PI / 2) + (Math.PI / 4)) : Math.PI / 4d;
                    var res = net.Forward(matrix, targetAngle, oppositeAngle);
                    var gradient = res.Item1;
                    var output = res.Item2;
                    var loss = res.Item3;
                    var absloss = Math.Abs(loss[0][0]) + Math.Abs(loss[0][1]);
                    sumLoss += absloss;
                    numLoss += 1d;
                    var x = output[0][0];
                    var y = output[0][1];
                    double resultMagnitude = Math.Sqrt((x * x) + (y * y));
                    double resultAngle = Math.Atan2(y, x);
                    if (sub == "A")
                    {
                        sumResultAngleA += resultAngle;
                        numResultAngleA += 1d;
                    } else if (sub == "B")
                    {
                        sumResultAngleB += resultAngle;
                        numResultAngleB += 1d;
                    }
                    double avgloss = sumLoss / (numLoss + 1E-9);

                    Console.WriteLine($"Iteration {i} {sub} Mag: {resultMagnitude}, Angle: {resultAngle}, TargetAngle: {targetAngle}, Gradient: {gradient[0][0]}, {gradient[0][1]} Loss: {absloss}");
                    Console.WriteLine($"Average Result Angle A: {sumResultAngleA / (numResultAngleA + 1E-9)}");
                    Console.WriteLine($"Average Result Angle B: {sumResultAngleB / (numResultAngleB + 1E-9)}");

                    //if (Math.Abs(loss[0][0]) >= 200d)
                    //{
                        Console.WriteLine($"Average loss: {avgloss}");
                        await net.Backward(gradient);
                        net.ApplyGradients();
                    //}

                    await net.Reset();
                    Thread.Sleep(1000);
                    if (i % 100 == 90)
                    {
                        sumResultAngleA = 0d;
                        numResultAngleA = 0d;
                        sumResultAngleB = 0d;
                        numResultAngleB = 0d;
                        sumLoss = 0d;
                        numLoss = 0d;
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
