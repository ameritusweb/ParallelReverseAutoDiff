using GradientExplorer.Helpers;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Mcts
{
    public interface IGameStateGenerator
    {
        Task<IQueue<GameState>> GenerateUniqueGameStates(GameState currentGameState);

        Task<GameState> GetNextRandomGameState(GameState currentGameState);
    }
}
