namespace MultAgentConsultaRagGrafoSemanticKernel;

// ════════════════════════════════════════════════════════════════════════════
//  StateGraph<TState> — motor de grafo com suporte a conditional edges
//
//  Python → C#:
//  builder = StateGraph(AgentState)          StateGraph<AgentState>
//  builder.add_node("planner", fn)      →    .AddNode("planner", fn)
//  builder.add_edge("a", "b")           →    .AddEdge("a", "b")
//  builder.add_conditional_edges(..)    →    .AddConditionalEdge(..)
//  builder.set_entry_point("planner")   →    .SetEntryPoint("planner")
//  builder.compile()                    →    .Compile()
//  graph.stream({state}, thread)        →    compiled.StreamAsync(state)
// ════════════════════════════════════════════════════════════════════════════
public static class GraphEnd
{
    public const string Node = "__END__";
}

public class NodeEvent<TState>
{
    public string NodeName { get; init; } = string.Empty;
    public TState State { get; init; } = default!;
    public bool IsEnd { get; init; } = false;
}

public class StateGraph<TState>
{
    private readonly Dictionary<string, Func<TState, Task<TState>>> _nodes = new();
    private readonly Dictionary<string, string> _edges = new();
    private readonly Dictionary<string, (Func<TState, string> Router,
                                         Dictionary<string, string> Map)> _conditionalEdges = new();
    private string _entryPoint = string.Empty;

    public StateGraph<TState> AddNode(string name, Func<TState, Task<TState>> handler)
    {
        _nodes[name] = handler;
        return this;
    }

    public StateGraph<TState> AddEdge(string from, string to)
    {
        _edges[from] = to;
        return this;
    }

    public StateGraph<TState> AddConditionalEdge(
        string from,
        Func<TState, string> router,
        Dictionary<string, string> routeMap)
    {
        _conditionalEdges[from] = (router, routeMap);
        return this;
    }

    public StateGraph<TState> SetEntryPoint(string nodeName)
    {
        _entryPoint = nodeName;
        return this;
    }

    public CompiledGraph<TState> Compile()
    {
        if (string.IsNullOrEmpty(_entryPoint))
            throw new InvalidOperationException("Chame SetEntryPoint() antes de Compile().");

        return new CompiledGraph<TState>(_nodes, _edges, _conditionalEdges, _entryPoint);
    }
}

public class CompiledGraph<TState>
{
    private readonly Dictionary<string, Func<TState, Task<TState>>> _nodes;
    private readonly Dictionary<string, string> _edges;
    private readonly Dictionary<string, (Func<TState, string> Router,
                                          Dictionary<string, string> Map)> _conditionalEdges;
    private readonly string _entryPoint;

    internal CompiledGraph(
        Dictionary<string, Func<TState, Task<TState>>> nodes,
        Dictionary<string, string> edges,
        Dictionary<string, (Func<TState, string>, Dictionary<string, string>)> conditionalEdges,
        string entryPoint)
    {
        _nodes = nodes;
        _edges = edges;
        _conditionalEdges = conditionalEdges;
        _entryPoint = entryPoint;
    }

    /// <summary>
    /// Executa o grafo emitindo um NodeEvent por nó.
    /// Equivalente ao: for s in graph.stream({state}, thread): print(s)
    /// </summary>
    public async IAsyncEnumerable<NodeEvent<TState>> StreamAsync(TState initialState)
    {
        var state = initialState;
        var currentNode = _entryPoint;
        var maxSteps = 50;
        var step = 0;

        while (currentNode != GraphEnd.Node && step++ < maxSteps)
        {
            if (!_nodes.TryGetValue(currentNode, out var handler))
                throw new InvalidOperationException($"Nó '{currentNode}' não encontrado.");

            state = await handler(state);

            yield return new NodeEvent<TState>
            {
                NodeName = currentNode,
                State = state,
                IsEnd = false
            };

            currentNode = ResolveNext(currentNode, state);
        }

        yield return new NodeEvent<TState>
        {
            NodeName = GraphEnd.Node,
            State = state,
            IsEnd = true
        };
    }

    private string ResolveNext(string current, TState state)
    {
        if (_conditionalEdges.TryGetValue(current, out var cond))
        {
            var key = cond.Router(state);
            return cond.Map.TryGetValue(key, out var mapped) ? mapped : GraphEnd.Node;
        }
        return _edges.TryGetValue(current, out var next) ? next : GraphEnd.Node;
    }
}
