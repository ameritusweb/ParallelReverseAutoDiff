using Autofac;
using GradientExplorer.Parsers;
using GradientExplorer.Services;
using GradientExplorer.ViewModels;
using System.Collections.Concurrent;
using System.Collections.Generic;

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
                            // Register ConcurrentDictionary for sync and async subscriptions
                            builder.Register(ctx => new ConcurrentDictionary<EventType, ConcurrentDictionary<int, List<SubscriptionBase>>>())
                                   .AsSelf()
                                   .InstancePerLifetimeScope();
                            builder.RegisterType<MethodParser>().As<IMethodParser>();
                            builder.RegisterType<NodeFactory>().As<INodeFactory>();
                            builder.RegisterType<NodeTypeFactory>().As<INodeTypeFactory>();
                            builder.RegisterType<ExpressionDecomposer>()
                                .As<IExpressionDecomposer>()
                                .SingleInstance();
                            builder.RegisterType<NodeBuilderPool>()
                                .As<INodeBuilderPool>()
                                .SingleInstance();
                            builder.RegisterType<ExpressionDifferentiator>().As<IExpressionDifferentiator>();
                            builder.RegisterType<GradientGraphFactory>().As<IGradientGraphFactory>();
                            builder.RegisterType<LaTeXBuilder>().As<ILaTeXBuilder>();
                            // Register your IEventAggregator as a single instance
                            builder.RegisterType<EventAggregator>()
                                .As<IEventAggregator>()
                                .SingleInstance();
                            builder.RegisterType<EnvironmentProvider>()
                                .As<IEnvironmentProvider>()
                                .SingleInstance();
                            builder.RegisterType<DateTimeProvider>()
                                .As<IDateTimeProvider>()
                                .SingleInstance();
                            builder.RegisterType<PaneCreator>()
                                .As<IPaneCreator>()
                                .SingleInstance();
                            builder.RegisterType<Logger>()
                                .As<ILogger>()
                                .SingleInstance();
                            builder.RegisterType<LaTeXBuilder>()
                                .As<ILaTeXBuilder>()
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
