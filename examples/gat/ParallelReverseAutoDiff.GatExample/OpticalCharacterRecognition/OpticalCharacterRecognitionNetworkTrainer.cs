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
                OpticalCharacterRecognitionNetwork network = new OpticalCharacterRecognitionNetwork(34, 223, 3, 0.0002d, 4);
                await network.Initialize();
                network.ApplyWeights();
                RandomNumberGenerator generator = new RandomNumberGenerator();
                var jsonFiles = Directory.GetFiles(@"E:\images\inputs\ocr", "*.json");
                var pairsAll = RandomPairGenerator.GenerateRandomPairs(jsonFiles.Length);
                int i = 0;
                List<double> targets = new List<double>();
                //var pairs2 = pairsAll.Select((x, ind) => (ind, jsonFiles[x.Item1].Substring(jsonFiles[x.Item1].LastIndexOf('\\') + 1).Substring(16) + " "
                //                       + jsonFiles[x.Item2].Substring(jsonFiles[x.Item2].LastIndexOf('\\') + 1).Substring(16))).Where(x => x.Item2.StartsWith("A_Inter_30px_pos1.", StringComparison.Ordinal)).ToList();
                //var pairs3 = pairs2.Where(x => x.Item2.Substring(6).Contains("A_Inter", StringComparison.Ordinal)).ToList();
                //var pairs = pairs3.Select(x => pairsAll[x.ind]).ToList();
                foreach (var pair in pairsAll)
                {
                    var i1 = pair.Item1;
                    var i2 = pair.Item2;
                    var json1 = File.ReadAllText(jsonFiles[i1]);
                    var data1 = JsonConvert.DeserializeObject<List<List<double>>>(json1);
                    var json2 = File.ReadAllText(jsonFiles[i2]);
                    var data2 = JsonConvert.DeserializeObject<List<List<double>>>(json2);
                    var data = data1.Concat(data2).ToList();
                    var file1 = jsonFiles[i1].Substring(jsonFiles[i1].LastIndexOf('\\') + 1);
                    var file2 = jsonFiles[i2].Substring(jsonFiles[i2].LastIndexOf('\\') + 1);
                    var sub1 = file1.Substring(16);
                    var sub2 = file2.Substring(16);
                    double targetMax = sub1.Substring(0, 1) == sub2.Substring(0, 1) ? 65d : 1d;
                    Matrix matrix = new Matrix(data.Count, data[0].Count);
                    for (int j = 0; j < data.Count; j++)
                    {
                        for (int k = 0; k < data[0].Count; k++)
                        {
                            matrix[j, k] = data[j][k];
                        }
                    }
                    if (targetMax == 1.0d && targetMax == targets.LastOrDefault())
                    {
                        continue;
                    }
                    i++;

                    targets.Add(targetMax);
                   
                    var (gradient, output, sorted) = network.Forward(matrix, targetMax, sub1, sub2);

                    Console.WriteLine("Target: " + targetMax + " " + sub1 + " " + sub2 + " " + (sorted.Any() ? sorted.Max() : "") + ", Grad: " + gradient[0].Max());

                    var inputGradient = await network.Backward(gradient, !sorted.Any());
                    //var randLearning = generator.GetRandomNumber(0.00001d, 0.0001d);
                    //network.AdjustLearningRate(randLearning);
                    network.ApplyGradients();
                    await network.Reset();
                    Thread.Sleep(1000);
                    if (i % 25 == 24)
                    {
                        network.SaveWeights();
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
