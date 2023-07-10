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
        private GraphAttentionPathsNeuralNetwork neuralNetwork;

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
            CudaBlas.Instance.Initialize();
            try
            {
                var graphFiles = Directory.GetFiles("G:\\My Drive\\graphs", "*.zip").ToList();

                for (int i = 0; i < 1001; ++i)
                {
                    var randomGraphFiles = graphFiles.OrderBy(x => this.rand.Next()).ToList();
                    List<GapGraph> graphs = new List<GapGraph>();
                    for (int j = 0; j < randomGraphFiles.Count; ++j)
                    {
                        var file = randomGraphFiles[j];
                        var jsons = this.ExtractFromZip(file);
                        var randomJson = jsons.OrderBy(x => this.rand.Next()).First();
                        var graph = JsonConvert.DeserializeObject<GapGraph>(randomJson) ?? throw new InvalidOperationException("Could not deserialize to graph.");
                        graph.Populate();
                        if (!graph.GapPaths.Any(x => x.IsTarget))
                        {
                            j--;
                            continue;
                        }

                        if (graph.GapPaths.Where(x => x.IsTarget).Count() > 1)
                        {
                            j--;
                            continue;
                        }

                        graphs.Add(graph);

                        if (graphs.Count == 4)
                        {
                            break;
                        }
                    }

                    var json = JsonConvert.SerializeObject(graphs);
                    File.WriteAllText("minibatch.json", json);

                    await this.ProcessMiniBatch(graphs);
                    Thread.Sleep(5000);

                    if (i % 10 == 0 || i % 10 == 5)
                    {
                        try
                        {
                            this.neuralNetwork.SaveWeights();
                        }
                        catch (OutOfMemoryException ex)
                        {
                            Console.WriteLine(ex);
                        }
                    }
                }
            }
            finally
            {
                CudaBlas.Instance.Dispose();
            }
        }

        private async Task ProcessMiniBatch(List<GapGraph> graphs)
        {
            try
            {
                if (this.neuralNetwork == null)
                {
                    this.neuralNetwork = new GraphAttentionPathsNeuralNetwork(graphs, 16, 115, 7, 2, 4, 0.001d, 4d);
                    await this.neuralNetwork.Initialize();
                    this.neuralNetwork.ApplyWeights();
                }
                else
                {
                    this.neuralNetwork.Reinitialize(graphs);
                }

                DeepMatrix gradientOfLoss = this.neuralNetwork.Forward();
                await this.neuralNetwork.Backward(gradientOfLoss);
                this.neuralNetwork.ApplyGradients();
                await this.neuralNetwork.Reset();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
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
