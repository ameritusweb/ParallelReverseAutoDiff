using GradientExplorer.Helpers;
using GradientExplorer.Services;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace GradientExplorer.Mcts
{
    /// <summary>
    /// Main class for Monte Carlo Tree Search for algebraic expression simplification
    /// </summary>
    public class MctsEngine : IMctsEngine
    {
        private readonly IGameStateGenerator gameStateGenerator;
        private readonly ILogger logger;
        private readonly ConcurrentPruner pruner;
        private readonly Random rand;

        private readonly CancellationTokenSource cts = new CancellationTokenSource();

        private double highestScoreSoFar = double.MinValue;
        private int maxDepthSoFar = 0;  // To keep track of the maximum depth
        private int maxRolloutDepthSoFar = 0;
        private int numberOfNodes = 0;
        private readonly List<Task> allTasks = new List<Task>();
        private int maxVisitsReached = 0;  // To keep track of the maximum number of visits
        private bool isInitialized = false;

        public ITreeNode Root { get; set; }
        public int MaxConcurrentRollouts { get; set; } = 8;

        public double ExplorationConstant { get; set; } = Math.Sqrt(2); // The constant C in UCB1 formula

        // Constant defining the depth of a rollout
        public int RolloutDepth { get; set; } = 10;

        // Create a new ManualResetEventSlim that starts in the non-signaled state (false).
        readonly ManualResetEventSlim queueNotEmpty = new ManualResetEventSlim(false);

        public MctsEngine(IGameStateGenerator gameStateGenerator, ILogger logger)
        {
            this.gameStateGenerator = gameStateGenerator;
            this.logger = logger;
            pruner = new ConcurrentPruner(TimeSpan.FromSeconds(5), logger);
            rand = new Random(Guid.NewGuid().GetHashCode());
        }

        public void StartPruner()
        {
            pruner.Start();
        }

        public void StopPruner()
        {
            pruner.Stop();
        }

        // Initialize the root node
        public void Initialize(GameState gameState)
        {
            Root = new TreeNode(gameState);
            // Initialize Root's game state here
            isInitialized = true;
        }

        // Main function to perform Monte Carlo Tree Search
        public async Task<IReadOnlyList<SimplificationAction>> RunMCTS()
        {
            if (!isInitialized)
            {
                throw new InvalidOperationException("MCTS Engine has not been initialized.");
            }

            logger.Log(nameof(MctsEngine), "MCTS Run started.", Helpers.SeverityType.Information);

            StartPruner();

            // Concurrent queue to hold nodes that need to be expanded
            ConcurrentQueue<MctsAction> expandQueue = new ConcurrentQueue<MctsAction>();
            expandQueue.Enqueue(new MctsAction { TreeNode = Root, Depth = 0 });

            // Signal that the queue is not empty
            queueNotEmpty.Set();

            // Flag to control the termination of the MCTS loop
            bool shouldTerminate = false;

            int initialDepth = 0;

            // Keep track of all the tasks you've started so you can await them later if needed

            // Semaphore to limit the number of concurrent rollouts
            var semaphore = new SemaphoreSlim(MaxConcurrentRollouts);

            // Control the number of total tasks created
            int totalTasksCreated = 0;

            while (!shouldTerminate)
            {
                if (cts.Token.IsCancellationRequested)
                {
                    break;
                }

                // Wait until the queue has items
                queueNotEmpty.Wait();

                await semaphore.WaitAsync();

                // Reset the event if the queue is empty
                if (!expandQueue.Any())
                {
                    queueNotEmpty.Reset();
                    continue;
                }

                if (totalTasksCreated < MaxConcurrentRollouts)  // Limit the total number of tasks
                {

                    allTasks.Add(Task.Run(() => ProcessMctsActionAsync(expandQueue, semaphore, initialDepth)));

                    totalTasksCreated++;
                }
                else
                {
                    semaphore.Release();

                    // Wait for some task to complete before creating a new one
                    Task completedTask = await Task.WhenAny(allTasks);
                    allTasks.Remove(completedTask);
                    totalTasksCreated--;
                }
            }

            // Await all tasks to complete
            await Task.WhenAll(allTasks);

            // Find the best node based on your criteria (e.g., highest score)
            ITreeNode bestNode = SelectBestFinalNode();

            StopPruner();

            // Retrieve the best sequence of actions leading to that node
            return GetBestActionSequence(bestNode);
        }

        public async Task ProcessMctsActionAsync(ConcurrentQueue<MctsAction> expandQueue, SemaphoreSlim semaphore, int initialDepth)
        {
            MctsAction nodeToExpand;
            try
            {
                if (expandQueue.TryDequeue(out nodeToExpand))
                {
                    // Reset the event if the queue is empty
                    if (!expandQueue.Any())
                    {
                        queueNotEmpty.Reset();
                    }

                    // Increment VisitorsCount as a new task starts working on this node
                    nodeToExpand.TreeNode.AtomicIncrementVisitorsCount();

                    // Perform Selection
                    ITreeNode leafNode = SelectNode(nodeToExpand.TreeNode);

                    if (leafNode == null)
                    {
                        leafNode = nodeToExpand.TreeNode;
                    }

                    // Perform Expansion (asynchronously)
                    await ExpandNodeAsync(leafNode, nodeToExpand.Depth + 1);

                    if (leafNode.Children.Any())
                    {
                        // Enqueue an expansion event right before initiating a rollout
                        // Enqueue the leaf node
                        expandQueue.Enqueue(new MctsAction { TreeNode = leafNode, Depth = nodeToExpand.Depth + 1 });

                        // Signal that the queue is not empty
                        queueNotEmpty.Set();
                    }

                    // Enqueue an expansion event right before initiating a rollout
                    // Enqueue the root node for a new expansion phase
                    expandQueue.Enqueue(new MctsAction { TreeNode = Root, Depth = 0 });

                    // Signal that the queue is not empty
                    queueNotEmpty.Set();

                    // Perform Simulation (rollout)
                    double rolloutScore = await SimulateRandomRollout(leafNode, nodeToExpand.Depth + 1);

                    // Perform Backpropagation
                    Backpropagate(leafNode, rolloutScore);

                    nodeToExpand.TreeNode.AtomicDecrementVisitorsCount();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Exception occurred inside MCTS task: {ex.Message}");
            }
            finally
            {
                semaphore.Release();
            }
        }

        // Selection logic using UCB1 formula
        public ITreeNode SelectNode(ITreeNode node)
        {
            ITreeNode selected = null;
            double bestValue = double.MinValue;

            foreach (var child in node.Children.Where(x => !x.MarkForPruning))
            {
                // Compute UCB1 value
                double ucbValue = (child.Score / child.Visits) +
                    ExplorationConstant * Math.Sqrt(Math.Log(node.Visits) / child.Visits);

                ucbValue += (1E-6 * rand.NextDouble());

                // Update best node if needed
                if (ucbValue > bestValue)
                {
                    selected = child;
                    bestValue = ucbValue;
                }
            }

            return selected;
        }

        // Function to expand a node by adding child nodes
        public async Task ExpandNodeAsync(ITreeNode node, int currentDepth)
        {
            // If the node is already fully expanded, return immediately
            if (node.IsFullyExpanded)
            {
                return;
            }

            // Get unique game states to be expanded
            IQueue<GameState> uniqueGameStates = await gameStateGenerator.GenerateUniqueGameStates(node.GameState);

            int counter = 0; // Initialize a counter to keep track of how many nodes are expanded
            const int maxExpansions = 5; // Set the maximum number of expansions

            while (!uniqueGameStates.IsEmpty && counter < maxExpansions)
            {
                if (uniqueGameStates.TryDequeue(out GameState gameState))
                {
                    ITreeNode child = new TreeNode(gameState, node);

                    // Directly add to the existing Children collection
                    node.Children.Add(child);
                    if (node.Children.Count > 10)
                    {

                    }
                    counter++; // Increment the counter
                }
            }

            if (uniqueGameStates.IsEmpty)
            {
                node.IsFullyExpanded = true;
            }

            if (currentDepth > maxDepthSoFar)
            {
                maxDepthSoFar = currentDepth;
                logger.Log(nameof(MctsEngine), $"Tree expanded. New depth: {currentDepth}", Helpers.SeverityType.Information);
            }
        }

        // Function to perform a random rollout from a node and return the gained score
        public async Task<double> SimulateRandomRollout(ITreeNode node, int currentDepth)
        {
            ITreeNode currentNode = node;
            GameState gameState = node.GameState;
            double accumulatedScore = 0;

            for (int i = 0; i < RolloutDepth; ++i)
            {
                if (IsTerminalState(gameState))
                {
                    break;
                }

                // Randomly expand and advance game state and update score
                ITreeNode childNode = await RandomlyExpandAndAdvanceGameState(node);

                // Add the new child node to the current node's children
                currentNode.Children.Add(childNode);
                if (currentNode.Children.Count > 10)
                {

                }

                // Make the new child node the current node for the next iteration
                currentNode = childNode;

                Interlocked.Increment(ref numberOfNodes);

                accumulatedScore += EvaluateGameState(gameState); // Assuming you have a function to evaluate the game state
            }

            int rolloutDepth = currentDepth + RolloutDepth;  // Calculate the new depth after the rollout
            maxRolloutDepthSoFar = Math.Max(maxRolloutDepthSoFar, rolloutDepth);  // Update maxRolloutDepthSoFar

            logger.Log(nameof(MctsEngine), $"Rollout complete. Rollout Depth: {rolloutDepth}, Max Rollout Depth: {maxRolloutDepthSoFar}", Helpers.SeverityType.Information);


            return accumulatedScore;
        }

        public async Task<ITreeNode> RandomlyExpandAndAdvanceGameState(ITreeNode node)
        {
            // Use GameStateGenerator to get the next random GameState
            GameState newState = await gameStateGenerator.GetNextRandomGameState(node.GameState);

            // Create and return the new child node
            ITreeNode child = new TreeNode(newState, node);

            return child;
        }

        // Dummy function to check if a game state is terminal; replace with actual logic
        public bool IsTerminalState(GameState gameState)
        {
            return false; // Implement actual logic
        }

        // Dummy function to evaluate a game state; replace with actual logic
        public double EvaluateGameState(GameState gameState)
        {
            return rand.NextDouble();
        }

        // Function to backpropagate the score from a leaf node up to the root
        public void Backpropagate(ITreeNode node, double score)
        {
            ITreeNode currentNode = node;

            // Update the node and all its ancestors
            while (currentNode != null)
            {
                currentNode.AtomicIncrementVisits();
                currentNode.AtomicAddToScore(score); // Using the atomic AddToScore method

                // Update the maxVisitsReached in a thread-safe manner
                int currentVisits = currentNode.Visits;
                int originalMaxVisits;
                do
                {
                    originalMaxVisits = maxVisitsReached;
                    if (currentVisits <= originalMaxVisits)
                    {
                        break;
                    }
                } while (Interlocked.CompareExchange(ref maxVisitsReached, currentVisits, originalMaxVisits) != originalMaxVisits);

                // Log the new maximum of visits if it was updated
                if (currentVisits > originalMaxVisits)
                {
                    logger.Log(nameof(MctsEngine), $"New maximum number of visits reached: {currentVisits}", Helpers.SeverityType.Information);
                }

                currentNode.AtomicIncrementVisitedForPruning();

                CheckForPruning(currentNode);

                currentNode = currentNode.Parent;
            }

            if (node.Score > highestScoreSoFar)
            {
                highestScoreSoFar = node.Score;
                logger.Log(nameof(MctsEngine), $"New best score found: {highestScoreSoFar}", Helpers.SeverityType.Information);
            }
        }

        public IReadOnlyList<SimplificationAction> GetBestActionSequence(ITreeNode bestNode)
        {
            List<SimplificationAction> actions = new List<SimplificationAction>();
            ITreeNode currentNode = bestNode;

            while (currentNode.Parent != null)
            {
                actions.Add(currentNode.Action);
                currentNode = currentNode.Parent;
            }

            // Reverse the list because we collected the actions from leaf to root
            actions.Reverse();

            return actions.AsReadOnly();
        }

        public ITreeNode SelectBestFinalNode()
        {
            // Assuming that a higher score indicates a "better" or more simplified expression,
            // find the node with the highest score.
            // Note: You can start the search from the root or any other node depending on your use case.
            return FindHighestScoreNode(Root);
        }

        private ITreeNode FindHighestScoreNode(ITreeNode node)
        {
            // If this is a leaf node, return the node itself
            if (!node.Children.Any())
            {
                return node;
            }

            // Otherwise, recurse into each child and find the best node
            ITreeNode bestChild = null;
            double highestScore = double.MinValue;

            foreach (var child in node.Children)
            {
                ITreeNode bestNodeInSubtree = FindHighestScoreNode(child);
                if (bestNodeInSubtree.Score > highestScore)
                {
                    highestScore = bestNodeInSubtree.Score;
                    bestChild = bestNodeInSubtree;
                }
            }

            return bestChild;
        }

        public async Task<IReadOnlyList<SimplificationAction>> CancelAsync()
        {
            cts.Cancel();

            // Wait for all ongoing tasks to complete
            await Task.WhenAll(allTasks);

            // Find the best node and return the best sequence of actions
            ITreeNode bestNode = SelectBestFinalNode();

            StopPruner();

            return GetBestActionSequence(bestNode);
        }

        private void CheckForPruning(ITreeNode node)
        {
            if (node.Children.All(child => child.IsFullyExpanded) && node.VisitedForPruning >= node.Children.Count)
            {
                // Take a snapshot and sort it
                var sortedChildren = node.Children.OrderBy(child => child.Score).ToList();

                // Determine the index up to which nodes will be marked for pruning
                int markIndex = sortedChildren.Count / 2;  // 50% of the lowest scoring nodes

                // Mark the lowest-scoring children for pruning
                foreach (var child in sortedChildren.Take(markIndex))
                {
                    child.MarkForPruning = true;
                    pruner.Enqueue(child);
                }

                // Reset the VisitedForPruning counter
                node.VisitedForPruning = 0;
            }
        }
    }
}
