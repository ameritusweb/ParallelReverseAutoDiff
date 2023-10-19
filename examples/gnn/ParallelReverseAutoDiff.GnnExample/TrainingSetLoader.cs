//------------------------------------------------------------------------------
// <copyright file="TrainingSetLoader.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    using System.Collections.Concurrent;
    using System.IO.Compression;
    using ManagedCuda.BasicTypes;
    using Newtonsoft.Json;
    using ParallelReverseAutoDiff.RMAD;
    using ParallelReverseAutoDiff.Test.GraphAttentionPaths;

    /// <summary>
    /// A training set loader.
    /// </summary>
    public class TrainingSetLoader
    {
        private readonly Random rand;
        private GraphAttentionPathsNeuralNetwork neuralNetwork;
        private ConcurrentBag<GapGraph> bagOfGraphs;
        private TrainingSetGenerator generator;

        /// <summary>
        /// Initializes a new instance of the <see cref="TrainingSetLoader"/> class.
        /// </summary>
        public TrainingSetLoader()
        {
            this.rand = new Random(Guid.NewGuid().GetHashCode());
            this.bagOfGraphs = new ConcurrentBag<GapGraph>();
            this.generator = new TrainingSetGenerator();
        }

        /// <summary>
        /// Loads mini-batch from FEN string.
        /// </summary>
        /// <returns>The task.</returns>
        public async Task LoadMiniBatchFromFen()
        {
            CudaBlas.Instance.Initialize();
            this.generator.Initialize();
            this.generator.AddToBag(
                this.bagOfGraphs,
                "4r1k1/1p1r1p2/p1p1qb1p/2Pp1b2/2nP4/PQB2pP1/4PPBP/3RR1K1 w - - 0 26",
                new Chess.Move(new Chess.Position("e2"), new Chess.Position("f3"), new Chess.Piece('P'), new Chess.Piece('p')),
                new Chess.Move(new Chess.Position("g4"), new Chess.Position("f3"), new Chess.Piece('p'), new Chess.Piece('N')));
            bool gresult = this.bagOfGraphs.TryTake(out var g);
            if (g != null && gresult)
            {
                bool result = false;
                for (int i = 0; i < 10; ++i)
                {
                    var miniBatchTask = this.ProcessMiniBatch(new GapGraph[] { g }.ToList());
                    await miniBatchTask;
                    result = miniBatchTask.Result;
                    Thread.Sleep(5000);
                    if (i == 2)
                    {
                        break;
                    }
                }

                if (result)
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

        /// <summary>
        /// Loads a mini-batch of training data from bag.
        /// </summary>
        /// <returns>A task.</returns>
        public async Task LoadMiniBatchFromBag()
        {
            CudaBlas.Instance.Initialize();
            this.generator.Initialize();
            await this.generator.AddToBag(this.bagOfGraphs, this.rand);

            try
            {
                for (int i = 0; i < 1001; ++i)
                {
                    List<GapGraph> graphs = new List<GapGraph>();

                    while (graphs.Count < 4)
                    {
                        bool gresult = this.bagOfGraphs.TryTake(out var g);
                        if (g != null && gresult)
                        {
                            if (!g.GapPaths.Any(x => x.IsTarget))
                            {
                                continue;
                            }

                            if (g.GapPaths.Where(x => x.IsTarget).Count() > 1)
                            {
                                continue;
                            }

                            graphs.Add(g);
                        }
                    }

                    var json = JsonConvert.SerializeObject(graphs);
                    File.WriteAllText("minibatch.json", json);

                    var bagTask = this.generator.AddToBag(this.bagOfGraphs, this.rand);
                    var miniBatchTask = this.ProcessMiniBatch(graphs);
                    await Task.WhenAll(bagTask, miniBatchTask);
                    var result = miniBatchTask.Result;
                    Thread.Sleep(5000);

                    if (result)
                    {
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
            }
            finally
            {
                CudaBlas.Instance.Dispose();
            }
        }

        /// <summary>
        /// Loads a mini-batch of training data.
        /// </summary>
        /// <returns>A task.</returns>
        public async Task LoadMiniBatch()
        {
            CudaBlas.Instance.Initialize();
            this.generator.Initialize();
            try
            {
                var graphFiles = Directory.GetFiles("G:\\My Drive\\graphs2", "*.zip").ToList();

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

                    var result = await this.ProcessMiniBatch(graphs);
                    Thread.Sleep(5000);

                    if (result)
                    {
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
            }
            finally
            {
                CudaBlas.Instance.Dispose();
            }
        }

        private async Task<bool> ProcessMiniBatch(List<GapGraph> graphs)
        {
            try
            {
                if (this.neuralNetwork == null)
                {
                    this.neuralNetwork = new GraphAttentionPathsNeuralNetwork(graphs, 18, 119, 5, 2, 4, 0.001d, 4d);
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
                return true;
            }
            catch (AggregateException ae)
            {
                Console.WriteLine(ae);
                await this.neuralNetwork.Reset();
                return false;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
                await this.neuralNetwork.Reset();
                return false;
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
