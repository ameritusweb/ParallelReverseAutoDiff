using Autofac;
using GradientExplorer.Parsers;
using GradientExplorer.ViewModels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientExplorer.Helpers
{
    public class AutofacContainerProvider
    {
        private static IContainer _container;
        private static readonly object _lock = new object();

        public static IContainer Container
        {
            get
            {
                if (_container == null)
                {
                    lock (_lock)
                    {
                        if (_container == null)
                        {
                            var builder = new ContainerBuilder();
                            // Register your types here
                            builder.RegisterType<MethodParser>().As<IMethodParser>();
                            // Register your IEventAggregator as a single instance
                            builder.RegisterType<EventAggregator>()
                                .As<IEventAggregator>()
                                .SingleInstance();
                            builder.RegisterType<GradientExplorerViewModel>().AsSelf();
                            builder.RegisterType<GradientToolboxViewModel>().AsSelf();
                            _container = builder.Build();
                        }
                    }
                }
                return _container;
            }
        }
    }

}
