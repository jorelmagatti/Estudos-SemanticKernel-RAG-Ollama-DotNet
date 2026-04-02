namespace EssayWriterBlazor.Graph;

public static class GraphEnd { public const string Node = "__END__"; }

public class NodeEvent<TState>
{
    public string  NodeName { get; init; } = string.Empty;
    public TState  State    { get; init; } = default!;
    public bool    IsEnd    { get; init; } = false;
}

public class StateGraph<TState>
{
    private readonly Dictionary<string, Func<TState, Task<TState>>>       _nodes = new();
    private readonly Dictionary<string, string>                            _edges = new();
    private readonly Dictionary<string, (Func<TState, string> Router,
                                          Dictionary<string, string> Map)> _cond  = new();
    private string _entry = string.Empty;

    public StateGraph<TState> AddNode(string name, Func<TState, Task<TState>> fn)
        { _nodes[name] = fn; return this; }

    public StateGraph<TState> AddEdge(string from, string to)
        { _edges[from] = to; return this; }

    public StateGraph<TState> AddConditionalEdge(
        string from, Func<TState, string> router, Dictionary<string, string> map)
        { _cond[from] = (router, map); return this; }

    public StateGraph<TState> SetEntryPoint(string name)
        { _entry = name; return this; }

    public CompiledGraph<TState> Compile() => new(_nodes, _edges, _cond, _entry);
}

public class CompiledGraph<TState>
{
    private readonly Dictionary<string, Func<TState, Task<TState>>>       _nodes;
    private readonly Dictionary<string, string>                            _edges;
    private readonly Dictionary<string, (Func<TState, string>, Dictionary<string, string>)> _cond;
    private readonly string _entry;

    internal CompiledGraph(
        Dictionary<string, Func<TState, Task<TState>>> nodes,
        Dictionary<string, string> edges,
        Dictionary<string, (Func<TState, string>, Dictionary<string, string>)> cond,
        string entry)
    { _nodes = nodes; _edges = edges; _cond = cond; _entry = entry; }

    public async IAsyncEnumerable<NodeEvent<TState>> StreamAsync(TState state)
    {
        var current = _entry;
        var step    = 0;

        while (current != GraphEnd.Node && step++ < 50)
        {
            if (!_nodes.TryGetValue(current, out var fn))
                throw new InvalidOperationException($"Nó '{current}' não encontrado.");

            state = await fn(state);

            yield return new NodeEvent<TState> { NodeName = current, State = state };

            // Resolve próximo nó
            if (_cond.TryGetValue(current, out var cond))
            {
                var key = cond.Item1(state);
                current = cond.Item2.TryGetValue(key, out var mapped) ? mapped : GraphEnd.Node;
            }
            else
            {
                current = _edges.TryGetValue(current, out var next) ? next : GraphEnd.Node;
            }
        }

        yield return new NodeEvent<TState>
        {
            NodeName = GraphEnd.Node,
            State    = state,
            IsEnd    = true
        };
    }
}
