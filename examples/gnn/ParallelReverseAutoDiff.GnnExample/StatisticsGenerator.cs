//------------------------------------------------------------------------------
// <copyright file="StatisticsGenerator.cs" author="ameritusweb" date="5/21/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace ParallelReverseAutoDiff.GnnExample
{
    using System.Diagnostics;
    using Chess;
    using Newtonsoft.Json;

    /// <summary>
    /// A statistics generator.
    /// </summary>
    public class StatisticsGenerator
    {
        private readonly ChessBoardLoader loader = new ChessBoardLoader();

        private readonly Dictionary<GamePhase, Dictionary<string, int>> edgeFrequenciesByPhase;
        private long totalMoves;

        /// <summary>
        /// Initializes a new instance of the <see cref="StatisticsGenerator"/> class.
        /// </summary>
        public StatisticsGenerator()
        {
            this.edgeFrequenciesByPhase = new Dictionary<GamePhase, Dictionary<string, int>>();
        }

        /// <summary>
        /// Reads the statistics.
        /// </summary>
        public void Read()
        {
            var dir = Directory.GetCurrentDirectory();

            // Define the file path to read the JSON data
            string filePath = dir + "\\edge_frequencies_2773067.json";

            // Read the JSON data from the file
            string json = File.ReadAllText(filePath);

            // Deserialize the JSON data into a nested dictionary
            Dictionary<string, Dictionary<string, int>> edgeFrequenciesByPhase = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, int>>>(json) ?? throw new Exception(string.Empty);

            // Order the dictionary by the highest values
            var orderedEdgesByPhase = edgeFrequenciesByPhase
                .SelectMany(phase => phase.Value.Select(edge => new { Phase = phase.Key, Edge = edge.Key, Frequency = edge.Value }))
                .OrderByDescending(entry => entry.Frequency)
                .ToList();

            // Print the ordered edges by phase
            foreach (var entry in orderedEdgesByPhase.TakeLast(100))
            {
                Debug.WriteLine($"Phase: {entry.Phase}, Edge: {entry.Edge}, Frequency: {entry.Frequency}");
            }
        }

        /// <summary>
        /// Reads the statistics.
        /// </summary>
        public void Read2()
        {
            var dir = Directory.GetCurrentDirectory();

            // Define the file path to read the JSON data
            string filePath = dir + "\\move_frequencies_2773067.json";

            // Read the JSON data from the file
            string json = File.ReadAllText(filePath);

            // Deserialize the JSON data into a nested dictionary
            Dictionary<string, Dictionary<string, int>> edgeFrequenciesByPhase = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, int>>>(json) ?? throw new Exception(string.Empty);

            // Order the dictionary by the highest values
            var orderedEdgesByPhase = edgeFrequenciesByPhase
                .SelectMany(phase => phase.Value.Select(edge => new { Phase = phase.Key, Edge = edge.Key, Frequency = edge.Value }))
                .OrderByDescending(entry => entry.Frequency)
                .ToList();

            // Print the ordered edges by phase
            foreach (var entry in orderedEdgesByPhase.TakeLast(100))
            {
                Debug.WriteLine($"Phase: {entry.Phase}, Edge: {entry.Edge}, Frequency: {entry.Frequency}");
            }
        }

        /// <summary>
        /// Reads the statistics.
        /// </summary>
        public void Read2a()
        {
            var dir = Directory.GetCurrentDirectory();

            // Define the file path to read the JSON data
            string filePath = dir + "\\move_frequencies_2773067.json";

            // Read the JSON data from the file
            string json = File.ReadAllText(filePath);

            // Deserialize the JSON data into a nested dictionary
            Dictionary<string, Dictionary<string, int>> edgeFrequenciesByPhase = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, int>>>(json) ?? throw new Exception(string.Empty);

            // Order the dictionary by the highest values
            var orderedEdgesByPhase = edgeFrequenciesByPhase
                .SelectMany(phase => phase.Value.Select(edge => new { Phase = phase.Key, Edge = edge.Key, Frequency = edge.Value }))
                .OrderByDescending(entry => entry.Frequency)
                .ToList();
            HashSet<string> paths = new HashSet<string>();

            // Print the ordered edges by phase
            foreach (var entry in orderedEdgesByPhase)
            {
                var edge = entry.Edge;
                var move = new Move(edge);
                var path = GameState.GetPath(move);
                var positions = path.Select(p => p.ToString());
                string ss = string.Empty;
                foreach (var position in positions)
                {
                    ss += position;
                }

                paths.Add(ss);
                Debug.WriteLine($"Phase: {entry.Phase}, Edge: {entry.Edge}, Frequency: {entry.Frequency}");
            }

            File.WriteAllText("paths.txt", string.Join(Environment.NewLine, paths));
        }

        /// <summary>
        /// Reads the statistics.
        /// </summary>
        public void Read3()
        {
            var dir = Directory.GetCurrentDirectory();

            // Define the file path to read the JSON data
            string filePath = dir + "\\actualmove_frequencies_2773067.json";

            // Read the JSON data from the file
            string json = File.ReadAllText(filePath);

            // Deserialize the JSON data into a nested dictionary
            Dictionary<string, Dictionary<string, int>> edgeFrequenciesByPhase = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, int>>>(json) ?? throw new Exception(string.Empty);

            // Order the dictionary by the highest values
            var orderedEdgesByPhase = edgeFrequenciesByPhase
                .SelectMany(phase => phase.Value.Select(edge => new { Phase = phase.Key, Edge = edge.Key, Frequency = edge.Value }))
                .OrderByDescending(entry => entry.Frequency)
                .ToList();

            // Print the ordered edges by phase
            foreach (var entry in orderedEdgesByPhase.TakeLast(1000))
            {
                Debug.WriteLine($"Phase: {entry.Phase}, Edge: {entry.Edge}, Frequency: {entry.Frequency}");
            }
        }

        /// <summary>
        /// Generates the statistics.
        /// </summary>
        public void Generate()
        {
            GameState gameState = new GameState();
            var total = this.loader.GetTotal();
            for (int i = 0; i < total; i++)
            {
                try
                {
                    var moves = this.loader.LoadMoves(i);
                    foreach (var amove in moves)
                    {
                        var phase = gameState.GetGamePhase();
                        var allmoves = gameState.GetMoves();
                        foreach (var move in allmoves)
                        {
                            if (this.edgeFrequenciesByPhase.ContainsKey(phase))
                            {
                                if (this.edgeFrequenciesByPhase[phase].ContainsKey(move))
                                {
                                    this.edgeFrequenciesByPhase[phase][move]++;
                                }
                                else
                                {
                                    this.edgeFrequenciesByPhase[phase].Add(move, 1);
                                }
                            }
                            else
                            {
                                this.edgeFrequenciesByPhase.Add(phase, new Dictionary<string, int>());
                                this.edgeFrequenciesByPhase[phase].Add(move, 1);
                            }
                        }

                        this.totalMoves += 1;

                        gameState.Board.Move(amove);
                    }
                }
                catch (Exception e)
                {
                    // ignored
                    Console.WriteLine(e.Message);
                }

                gameState.Board.Clear();
            }

            // Serialize the nested dictionary to JSON
            string json = JsonConvert.SerializeObject(this.edgeFrequenciesByPhase, Formatting.Indented);

            // Define the file path to save the JSON data
            string filePath = "edge_frequencies_" + this.totalMoves + ".json";

            // Save the JSON data to the file
            File.WriteAllText(filePath, json);
        }

        /// <summary>
        /// Generates the statistics.
        /// </summary>
        public void Generate2()
        {
            GameState gameState = new GameState();
            var total = this.loader.GetTotal();
            for (int i = 0; i < total; i++)
            {
                try
                {
                    var moves = this.loader.LoadMoves(i);
                    foreach (var amove in moves)
                    {
                        var phase = gameState.GetGamePhase();
                        var allmoves = gameState.GetAllMoves();
                        foreach (var move in allmoves)
                        {
                            if (this.edgeFrequenciesByPhase.ContainsKey(phase))
                            {
                                if (this.edgeFrequenciesByPhase[phase].ContainsKey(move.ToString()))
                                {
                                    this.edgeFrequenciesByPhase[phase][move.ToString()]++;
                                }
                                else
                                {
                                    this.edgeFrequenciesByPhase[phase].Add(move.ToString(), 1);
                                }
                            }
                            else
                            {
                                this.edgeFrequenciesByPhase.Add(phase, new Dictionary<string, int>());
                                this.edgeFrequenciesByPhase[phase].Add(move.ToString(), 1);
                            }
                        }

                        this.totalMoves += 1;

                        gameState.Board.Move(amove);
                    }
                }
                catch (Exception e)
                {
                    // ignored
                    Console.WriteLine(e.Message);
                }

                gameState.Board.Clear();
            }

            // Serialize the nested dictionary to JSON
            string json = JsonConvert.SerializeObject(this.edgeFrequenciesByPhase, Formatting.Indented);

            // Define the file path to save the JSON data
            string filePath = "move_frequencies_" + this.totalMoves + ".json";

            // Save the JSON data to the file
            File.WriteAllText(filePath, json);
        }

        /// <summary>
        /// Generates the statistics.
        /// </summary>
        public void Generate3()
        {
            GameState gameState = new GameState();
            var total = this.loader.GetTotal();
            for (int i = 0; i < total; i++)
            {
                try
                {
                    var moves = this.loader.LoadMoves(i);
                    foreach (var amove in moves)
                    {
                        var phase = gameState.GetGamePhase();
                        if (this.edgeFrequenciesByPhase.ContainsKey(phase))
                        {
                            if (this.edgeFrequenciesByPhase[phase].ContainsKey(amove.ToString()))
                            {
                                this.edgeFrequenciesByPhase[phase][amove.ToString()]++;
                            }
                            else
                            {
                                this.edgeFrequenciesByPhase[phase].Add(amove.ToString(), 1);
                            }
                        }
                        else
                        {
                            this.edgeFrequenciesByPhase.Add(phase, new Dictionary<string, int>());
                            this.edgeFrequenciesByPhase[phase].Add(amove.ToString(), 1);
                        }

                        this.totalMoves += 1;

                        gameState.Board.Move(amove);
                    }
                }
                catch (Exception e)
                {
                    // ignored
                    Console.WriteLine(e.Message);
                }

                gameState.Board.Clear();
            }

            // Serialize the nested dictionary to JSON
            string json = JsonConvert.SerializeObject(this.edgeFrequenciesByPhase, Formatting.Indented);

            // Define the file path to save the JSON data
            string filePath = "actualmove_frequencies_" + this.totalMoves + ".json";

            // Save the JSON data to the file
            File.WriteAllText(filePath, json);
        }
    }
}
