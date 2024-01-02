using Newtonsoft.Json;
using ParallelReverseAutoDiff.RMAD;

namespace ParallelReverseAutoDiff.GatExample.OpticalCharacterRecognition
{
    public class OpticalCharacterRecognitionNetworkTrainer
    {
        public OpticalCharacterRecognitionNetworkTrainer()
        {

        }

        public async Task Train3()
        {
            try
            {
                CudaBlas.Instance.Initialize();
                OpticalCharacterRecognitionNetwork network = new OpticalCharacterRecognitionNetwork(34, 223, 3, 0.00002d, 4);
                await network.Initialize();
                network.ApplyWeights();
                RandomNumberGenerator generator = new RandomNumberGenerator();
                var jsonFiles = Directory.GetFiles(@"E:\images\inputs\ocr", "*.json");
                var pairs = RandomPairGenerator.GenerateRandomPairs(jsonFiles.Length);
                int i = 0;
                List<double> targets = new List<double>();
                foreach (var pair in pairs)
                {
                    i++;
                    var i1 = pair.Item1;
                    var i2 = pair.Item2;
                    var json1 = File.ReadAllText(jsonFiles[i1]);
                    var data1 = JsonConvert.DeserializeObject<List<List<double>>>(json1);
                    var json2 = File.ReadAllText(jsonFiles[i2]);
                    var data2 = JsonConvert.DeserializeObject<List<List<double>>>(json2);
                    var data = data1.Concat(data2).ToList();
                    var file1 = jsonFiles[i1].Substring(jsonFiles[i1].LastIndexOf('\\') + 1);
                    var file2 = jsonFiles[i2].Substring(jsonFiles[i2].LastIndexOf('\\') + 1);
                    var sub1 = file1.Substring(16, 1);
                    var sub2 = file2.Substring(16, 1);
                    double targetMax = sub1 == sub2 ? 0.75d : 0.2d;
                    Matrix matrix = new Matrix(data.Count, data[0].Count);
                    for (int j = 0; j < data.Count; j++)
                    {
                        for (int k = 0; k < data[0].Count; k++)
                        {
                            matrix[j, k] = data[j][k];
                        }
                    }
                    if (targetMax == 0.2d && targetMax == targets.LastOrDefault())
                    {
                        continue;
                    }

                    targets.Add(targetMax);
                   
                    var (gradient, output, sorted) = network.Forward(matrix, targetMax, sub1, sub2);

                    Console.WriteLine("Target: " + targetMax + " " + sub1 + " " + sub2 + " " + sorted.Max() + ", Grad: " + gradient[0].Max());

                    //var inputGradient = await network.Backward(gradient);
                    //var randLearning = generator.GetRandomNumber(0.00001d, 0.0001d);
                    //network.AdjustLearningRate(randLearning);
                    //network.ApplyGradients();
                    await network.Reset();
                    Thread.Sleep(1000);
                    if (i % 21 == 20)
                    {
                        //network.SaveWeights();
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            finally
            {
                CudaBlas.Instance.Dispose();
            }
        }

        public async Task Train2()
        {
            try
            {
                CudaBlas.Instance.Initialize();
                OpticalCharacterRecognitionNetwork network = new OpticalCharacterRecognitionNetwork(34, 223, 3, 0.00001d, 4);
                await network.Initialize();
                var jsonFiles = Directory.GetFiles(@"E:\gatstore", "*.json");
                for (int i = 0; i < 100; i++)
                {
                    var json1 = File.ReadAllText(jsonFiles[0]);
                    var data1 = JsonConvert.DeserializeObject<List<List<double>>>(json1);
                    var json2 = File.ReadAllText(jsonFiles[1]);
                    var data2 = JsonConvert.DeserializeObject<List<List<double>>>(json2);
                    var data = data1.Concat(data2).ToList();
                    Matrix matrix = new Matrix(data.Count, data[0].Count);
                    for (int j = 0; j < data.Count; j++)
                    {
                        for (int k = 0; k < data[0].Count; k++)
                        {
                            matrix[j, k] = data[j][k];
                        }
                    }
                    var (gradient, output, sorted) = network.Forward(matrix, 0.75d, "A", "A");
                    var inputGradient = await network.Backward(gradient);
                    network.ApplyGradients();
                    await network.Reset();
                    Thread.Sleep(5000);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            finally
            {
                CudaBlas.Instance.Dispose();
            }
        }

        public async Task Train()
        {
            try
            {
                CudaBlas.Instance.Initialize();
                OpticalCharacterRecognitionNetwork network = new OpticalCharacterRecognitionNetwork(17, 223, 3, 0.0000001d, 4);
                await network.Initialize();
                var jsonFiles = Directory.GetFiles(@"E:\gatstore", "*.json");
                foreach (var jsonFile in jsonFiles)
                {
                    var json = File.ReadAllText(jsonFile);
                    var data = JsonConvert.DeserializeObject<List<List<double>>>(json);
                    Matrix matrix = new Matrix(data.Count, data[0].Count);
                    for (int i = 0; i < data.Count; i++)
                    {
                        for (int j = 0; j < data[0].Count; j++)
                        {
                            matrix[i, j] = data[i][j];
                        }
                    }

                    OnlineStatisticsCalculator calculator = new OnlineStatisticsCalculator();

                    for (int j = 0; j < 500; ++j)
                    {
                        var res = network.Forward(matrix, 0.25d, "A", "A");

                        // calculator.AddDataPoint(res);
                        // calculator.AddDataPoints(array);

                        Console.WriteLine($"Mean: {calculator.GetMean()}, StdDev: {calculator.GetStandardDeviation()}, Var: {calculator.GetVariance()}, Min: {calculator.GetMin()}, Max: {calculator.GetMax()}");
                        network.RandomizeWeights();
                        Thread.Sleep(10);
                    }

                    File.WriteAllText(jsonFile.Replace(".json", ".txt"), $"Mean: {calculator.GetMean()}, StdDev: {calculator.GetStandardDeviation()}, Var: {calculator.GetVariance()}, Min: {calculator.GetMin()}, Max: {calculator.GetMax()}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            finally
            {
                CudaBlas.Instance.Dispose();
            }
        }
    }
}
