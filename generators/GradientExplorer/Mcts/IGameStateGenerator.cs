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
        Task<ConcurrentQueue<GameState>> GenerateUniqueGameStates(GameState currentGameState);
    }
}
