using Newtonsoft.Json;
using ParallelReverseAutoDiff.RMAD;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelReverseAutoDiff.GatExample.OpticalCharacterRecognition
{
    public class OpticalCharacterRecognitionNetworkTrainer
    {
        public OpticalCharacterRecognitionNetworkTrainer()
        {

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

                    var gradient = network.Forward(matrix);
                    await network.Backward(gradient);

                    await network.Reset();
                    Thread.Sleep(1000);
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
