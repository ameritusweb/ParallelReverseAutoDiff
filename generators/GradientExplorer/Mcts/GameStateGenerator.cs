using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Mcts
{
    public class GameStateGenerator : IGameStateGenerator
    {
        private ConcurrentDictionary<GameState, ConcurrentQueue<GameState>> gameStateCache = new ConcurrentDictionary<GameState, ConcurrentQueue<GameState>>();
        
        // Dummy function to generate possible game states; replace with actual logic
        public async Task<ConcurrentQueue<GameState>> GenerateUniqueGameStates(GameState currentGameState)
        {

            await Task.Delay(100);
            // Implement game-specific logic to generate next possible game states
            if (!gameStateCache.ContainsKey(currentGameState))
            {
                var queue = new ConcurrentQueue<GameState>();
                queue.Enqueue(currentGameState);
                queue.Enqueue(currentGameState);
                queue.Enqueue(currentGameState);
                queue.Enqueue(currentGameState);
                queue.Enqueue(currentGameState);
                gameStateCache.TryAdd(currentGameState, queue);
            }
            

        }

        public async Task<GameState> GetNextRandomGameState(GameState currentGameState)
        {
            await Task.Delay(100);
            return currentGameState;
        }
    }
}
