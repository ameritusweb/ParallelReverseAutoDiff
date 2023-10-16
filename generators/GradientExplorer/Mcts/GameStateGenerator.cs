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
        // Dummy function to generate possible game states; replace with actual logic
        public async Task<ConcurrentQueue<GameState>> GenerateUniqueGameStates(GameState currentGameState)
        {

            await Task.Delay(100);
            // Implement game-specific logic to generate next possible game states
            return new ConcurrentQueue<GameState>();
        }

        public async Task<GameState> GetNextRandomGameState(GameState currentGameState)
        {
            await Task.Delay(100);
            return currentGameState;
        }
    }
}
