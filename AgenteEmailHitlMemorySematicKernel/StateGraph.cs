namespace AgenteEmailHitlMemorySematicKernel;

public static class GraphEnd { public const string Node = "__END__"; }

public class NodeEvent<TState>
{
    public string NodeName { get; init; } = string.Empty;
    public TState State { get; init; } = default!;
    public bool IsEnd { get; init; } = false;
    public string Log { get; init; } = string.Empty;
}

public class StateGraph<TState>
{
    private readonly Dictionary<string, Func<TState, Task<(TState, string, string)>>> _nodes = new();
    private string _entryPoint = string.Empty;

    public StateGraph<TState> AddNode(string name, Func<TState, Task<(TState, string, string)>> handler)
    {
        _nodes[name] = handler;
        return this;
    }

    public StateGraph<TState> SetEntryPoint(string name) { _entryPoint = name; return this; }

    public CompiledGraph<TState> Compile() => new(_nodes, _entryPoint);
}

public class CompiledGraph<TState>
{
    private readonly Dictionary<string, Func<TState, Task<(TState, string, string)>>> _nodes;
    private readonly string _entryPoint;

    internal CompiledGraph(
        Dictionary<string, Func<TState, Task<(TState, string, string)>>> nodes,
        string entryPoint)
    {
        _nodes = nodes;
        _entryPoint = entryPoint;
    }

    public async IAsyncEnumerable<NodeEvent<TState>> StreamAsync(TState initialState)
    {
        var state = initialState;
        var currentNode = _entryPoint;
        var step = 0;

        while (currentNode != GraphEnd.Node && step++ < 30)
        {
            if (!_nodes.TryGetValue(currentNode, out var handler))
                throw new InvalidOperationException($"Nó '{currentNode}' não encontrado.");

            var (nextState, nextNode, log) = await handler(state);
            state = nextState;

            yield return new NodeEvent<TState>
            {
                NodeName = currentNode,
                State = state,
                Log = log,
                IsEnd = false
            };

            currentNode = nextNode;
        }

        yield return new NodeEvent<TState>
        {
            NodeName = GraphEnd.Node,
            State = state,
            IsEnd = true
        };
    }
}
