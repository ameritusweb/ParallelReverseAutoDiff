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
            CudaBlas.Instance.Initialize();
            VectorNet net = new VectorNet(17, 446, 3, 0.0002d, 4d);
            await net.Initialize();

            var jsonFiles = Directory.GetFiles(@"E:\images\inputs\ocr", "*.json");

            Random random = new Random(Guid.NewGuid().GetHashCode());
            var files = jsonFiles.OrderBy(x => random.Next()).ToArray();
            foreach (var jsonFile in files)
            {
                var json = File.ReadAllText(jsonFile);
                var data = JsonConvert.DeserializeObject<List<List<double>>>(json);
                var file = jsonFile.Substring(jsonFile.LastIndexOf('\\') + 1);
                var sub = file.Substring(16, 1);

                Matrix matrix = new Matrix(data.Count, data[0].Count);
                for (int j = 0; j < data.Count; j++)
                {
                    for (int k = 0; k < data[0].Count; k++)
                    {
                        matrix[j, k] = data[j][k];
                    }
                }

                var res = net.Forward(matrix);

            }
        }
    }
}
