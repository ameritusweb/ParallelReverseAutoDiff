//------------------------------------------------------------------------------
// <copyright file="TrainingSetLoader.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    using System.IO.Compression;
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.RMAD;
    using ParallelReverseAutoDiff.Test.GraphAttentionPaths;

    /// <summary>
    /// A training set loader.
    /// </summary>
    public class TrainingSetLoader
    {
        private Random rand;

        /// <summary>
        /// Initializes a new instance of the <see cref="TrainingSetLoader"/> class.
        /// </summary>
        public TrainingSetLoader()
        {
            this.rand = new Random(Guid.NewGuid().GetHashCode());
        }

        /// <summary>
        /// Loads a mini-batch of training data.
        /// </summary>
        /// <returns>A task.</returns>
        public async Task LoadMiniBatch()
        {
            var graphFiles = Directory.GetFiles("G:\\My Drive\\graphs", "*.zip").ToList();
            var randomGraphFiles = graphFiles.OrderBy(x => this.rand.Next()).Take(4).ToList();
            List<GapGraph> graphs = new List<GapGraph>();
            for (int i = 0; i < randomGraphFiles.Count; ++i)
            {
                var file = randomGraphFiles[i];
                var jsons = this.ExtractFromZip(file);
                var randomJson = jsons.OrderBy(x => this.rand.Next()).First();
                var graph = JsonConvert.DeserializeObject<GapGraph>(randomJson) ?? throw new InvalidOperationException("Could not deserialize to graph.");
                graph.Populate();
                graphs.Add(graph);
            }

            var json = JsonConvert.SerializeObject(graphs);
            File.WriteAllText("minibatch.json", json);

            int batchSize = 4;
            try
            {
                CudaBlas.Instance.Initialize();
                GraphAttentionPathsNeuralNetwork neuralNetwork = new GraphAttentionPathsNeuralNetwork(graphs, batchSize, 16, 115, 10, 2, 4, 0.001d, 4d);
                await neuralNetwork.Initialize();
                DeepMatrix gradientOfLoss = neuralNetwork.Forward();
                await neuralNetwork.Backward(gradientOfLoss);
            }
            finally
            {
                CudaBlas.Instance.Dispose();
            }
        }

        private List<string> ExtractFromZip(string zipName)
        {
            List<string> jsons = new List<string>();

            using (FileStream fileStream = new FileStream(zipName, FileMode.Open))
            {
                using (ZipArchive archive = new ZipArchive(fileStream, ZipArchiveMode.Read))
                {
                    foreach (ZipArchiveEntry entry in archive.Entries)
                    {
                        using (Stream entryStream = entry.Open())
                        {
                            using (GZipStream gzipStream = new GZipStream(entryStream, CompressionMode.Decompress))
                            {
                                using (StreamReader reader = new StreamReader(gzipStream))
                                {
                                    string json = reader.ReadToEnd();
                                    jsons.Add(json);
                                }
                            }
                        }
                    }
                }
            }

            return jsons;
        }
    }
}
