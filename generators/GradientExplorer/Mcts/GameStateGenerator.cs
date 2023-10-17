using GradientExplorer.Helpers;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Windows.Documents;

namespace GradientExplorer.Mcts
{
    public class GameStateGenerator : IGameStateGenerator
    {
        private ConcurrentDictionary<GameState, DualQueueCollection<GameState>> gameStateCache = new ConcurrentDictionary<GameState, DualQueueCollection<GameState>>();

        public GameStateGenerator()
        {
        }

        // Dummy function to generate possible game states; replace with actual logic
        public async Task<IQueue<GameState>> GenerateUniqueGameStates(GameState currentGameState)
        {

            await Task.Delay(100);
            // Implement game-specific logic to generate next possible game states
            DualQueueCollection<GameState> coll = gameStateCache.GetOrAdd(currentGameState, (x) => new DualQueueCollection<GameState>());
            if (coll.InitialFill == null)
            {
                coll.InitialFill = () =>
                {
                    List<GameState> states = new List<GameState>();
                    states.Add(new GameState(new Model.GradientGraph()));
                    states.Add(new GameState(new Model.GradientGraph()));
                    states.Add(new GameState(new Model.GradientGraph()));
                    states.Add(new GameState(new Model.GradientGraph()));
                    states.Add(new GameState(new Model.GradientGraph()));
                    states.Add(new GameState(new Model.GradientGraph()));
                    states.Add(new GameState(new Model.GradientGraph()));
                    return states;
                };
            }
            return coll;
        }

        public async Task<GameState> GetNextRandomGameState(GameState currentGameState)
        {
            await Task.Delay(100);
            return new GameState(new Model.GradientGraph());
        }
    }
}
