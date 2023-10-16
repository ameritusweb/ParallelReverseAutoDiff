using GradientExplorer.Commands;
using GradientExplorer.Helpers;
using GradientExplorer.Mcts;
using GradientExplorer.Model;
using GradientExplorer.Services;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Input;

namespace GradientExplorer.ViewModels
{
    public class SimplificationToolViewModel : IViewModel, INotifyPropertyChanged
    {

        private ILogger logger;

        public ILogger Logger
        {
            get
            {
                return logger;
            }
        }

        public ICommand SimplifyCommand { get; set; }

        public SimplificationToolViewModel(GradientToolboxViewModel parent, ILogger logger) {

            this.Parent = parent;
            this.logger = logger;
            SimplifyCommand = new AsyncRelayCommand(SimplifyAsync, CanSimplify);
        }

        public GradientToolboxViewModel Parent { get; set; }

        private async Task SimplifyAsync()
        {
            GameStateGenerator generator = new GameStateGenerator();
            MctsEngine engine = new MctsEngine(generator, logger);
            var simplification = await engine.RunMCTS();
        }

        private bool CanSimplify()
        {
            return true;
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
