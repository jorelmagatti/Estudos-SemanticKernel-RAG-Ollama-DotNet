namespace AgenteConsultaRagHitl;

// ════════════════════════════════════════════════════════════════════════════
//  StateGraph<TState> — mini-framework de grafo com suporte a HITL
//
//  Equivalência com LangGraph:
//  ─────────────────────────────────────────────────────────────────────────
//  StateGraph(AgentState)           →  StateGraph<AgentState>
//  graph.add_node("llm", fn)        →  .AddNode("llm", fn)
//  graph.add_edge("a","b")          →  .AddEdge("a","b")
//  graph.add_conditional_edges(..)  →  .AddConditionalEdge(..)
//  graph.set_entry_point("llm")     →  .SetEntryPoint("llm")
//  graph.compile(                   →  .Compile(
//      checkpointer=memory,              interruptBefore: ["action"])
//      interrupt_before=["action"])
//  compiled.stream(state, cfg)      →  compiled.StreamAsync(state)
//  compiled.stream(None, cfg)       →  compiled.ResumeAsync(state)
// ════════════════════════════════════════════════════════════════════════════

public static class GraphConstants
{
    public const string END = "__END__";
}

/// <summary>
/// Evento emitido pelo grafo durante o streaming.
/// Cada evento representa a saída de um nó.
/// </summary>
public class NodeEvent<TState>
{
    public string NodeName { get; init; } = string.Empty;
    public TState State { get; init; } = default!;

    /// <summary>Tokens gerados em tempo real (nós com streaming).</summary>
    public IAsyncEnumerable<string>? StreamTokens { get; init; }

    /// <summary>True quando o grafo foi interrompido por interrupt_before.</summary>
    public bool IsInterrupted { get; init; } = false;
}

/// <summary>
/// Construtor do grafo — equivalente ao StateGraph() do LangGraph.
/// </summary>
public class StateGraph<TState>
{
    // Nó normal (sem streaming)
    private readonly Dictionary<string, Func<TState, Task<TState>>> _nodes = new();

    // Nó com streaming de tokens
    private readonly Dictionary<string, Func<TState, IAsyncEnumerable<(TState State, string? Token)>>>
        _streamingNodes = new();

    private readonly Dictionary<string, string> _edges = new();
    private readonly Dictionary<string, (Func<TState, string> Router,
                                         Dictionary<string, string> Map)> _conditionalEdges = new();

    private string _entryPoint = string.Empty;
    private List<string> _interruptBefore = new();

    // ── fluent API ────────────────────────────────────────────────────────────

    public StateGraph<TState> AddNode(string name, Func<TState, Task<TState>> handler)
    {
        _nodes[name] = handler;
        return this;
    }

    public StateGraph<TState> AddStreamingNode(
        string name,
        Func<TState, IAsyncEnumerable<(TState, string?)>> handler)
    {
        _streamingNodes[name] = handler;
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

    /// <summary>
    /// Compila o grafo.
    /// Equivalente a: graph.compile(checkpointer=memory, interrupt_before=["action"])
    /// </summary>
    public CompiledGraph<TState> Compile(IEnumerable<string>? interruptBefore = null)
    {
        if (string.IsNullOrEmpty(_entryPoint))
            throw new InvalidOperationException("Chame SetEntryPoint() antes de Compile().");

        return new CompiledGraph<TState>(
            _nodes,
            _streamingNodes,
            _edges,
            _conditionalEdges,
            _entryPoint,
            interruptBefore?.ToList() ?? new List<string>());
    }
}

/// <summary>
/// Grafo compilado com suporte a interrupt e resume.
/// </summary>
public class CompiledGraph<TState>
{
    private readonly Dictionary<string, Func<TState, Task<TState>>>
        _nodes;

    private readonly Dictionary<string, Func<TState, IAsyncEnumerable<(TState, string?)>>>
        _streamingNodes;

    private readonly Dictionary<string, string> _edges;
    private readonly Dictionary<string, (Func<TState, string> Router,
                                          Dictionary<string, string> Map)> _conditionalEdges;
    private readonly string _entryPoint;
    private readonly List<string> _interruptBefore;

    internal CompiledGraph(
        Dictionary<string, Func<TState, Task<TState>>> nodes,
        Dictionary<string, Func<TState, IAsyncEnumerable<(TState, string?)>>> streamingNodes,
        Dictionary<string, string> edges,
        Dictionary<string, (Func<TState, string>, Dictionary<string, string>)> conditionalEdges,
        string entryPoint,
        List<string> interruptBefore)
    {
        _nodes = nodes;
        _streamingNodes = streamingNodes;
        _edges = edges;
        _conditionalEdges = conditionalEdges;
        _entryPoint = entryPoint;
        _interruptBefore = interruptBefore;
    }

    /// <summary>
    /// Executa o grafo a partir do estado inicial, emitindo eventos por nó.
    ///
    /// Quando alcança um nó listado em interruptBefore, emite um NodeEvent
    /// com IsInterrupted=true e para — aguardando chamada a ResumeAsync().
    ///
    /// Equivalente a:
    ///   for event in abot.graph.stream({messages}, thread_config): ...
    /// </summary>
    public async IAsyncEnumerable<NodeEvent<TState>> StreamAsync(TState initialState)
    {
        var state = initialState;
        var currentNode = _entryPoint;
        var maxSteps = 20;
        var step = 0;

        while (currentNode != GraphConstants.END && step++ < maxSteps)
        {
            // ── interrupt_before: pausa ANTES de executar o nó ───────────────
            // Equivalente ao interrupt_before=["action"] do LangGraph
            if (_interruptBefore.Contains(currentNode))
            {
                yield return new NodeEvent<TState>
                {
                    NodeName = currentNode,
                    State = state,
                    IsInterrupted = true
                };
                yield break; // para o stream — resume será chamado depois
            }

            // ── Executa o nó ──────────────────────────────────────────────────
            if (_streamingNodes.TryGetValue(currentNode, out var streamHandler))
            {
                // Nó com streaming de tokens
                TState? finalState = default;
                var tokenChannel = System.Threading.Channels.Channel.CreateUnbounded<string>();

                var producer = Task.Run(async () =>
                {
                    await foreach (var (partial, token) in streamHandler(state))
                    {
                        finalState = partial;
                        if (token != null)
                            await tokenChannel.Writer.WriteAsync(token);
                    }
                    tokenChannel.Writer.Complete();
                });

                yield return new NodeEvent<TState>
                {
                    NodeName = currentNode,
                    State = state,
                    StreamTokens = ReadChannel(tokenChannel.Reader)
                };

                await producer;
                state = finalState!;
            }
            else if (_nodes.TryGetValue(currentNode, out var handler))
            {
                // Nó normal
                state = await handler(state);

                yield return new NodeEvent<TState>
                {
                    NodeName = currentNode,
                    State = state
                };
            }
            else
            {
                throw new InvalidOperationException($"Nó '{currentNode}' não registrado no grafo.");
            }

            // ── Resolve próximo nó ────────────────────────────────────────────
            currentNode = ResolveNextNode(currentNode, state);
        }
    }

    /// <summary>
    /// Retoma a execução após uma interrupção HITL, a partir do nó que foi pausado.
    ///
    /// Equivalente a:
    ///   for event in abot.graph.stream(None, thread_config): ...
    ///                                  ^^^^ None = retomar do checkpoint
    /// </summary>
    public async IAsyncEnumerable<NodeEvent<TState>> ResumeAsync(
        TState state,
        string resumeFromNode)
    {
        var currentNode = resumeFromNode;
        var maxSteps = 20;
        var step = 0;

        while (currentNode != GraphConstants.END && step++ < maxSteps)
        {
            if (_streamingNodes.TryGetValue(currentNode, out var streamHandler))
            {
                TState? finalState = default;
                var tokenChannel = System.Threading.Channels.Channel.CreateUnbounded<string>();

                var producer = Task.Run(async () =>
                {
                    await foreach (var (partial, token) in streamHandler(state))
                    {
                        finalState = partial;
                        if (token != null)
                            await tokenChannel.Writer.WriteAsync(token);
                    }
                    tokenChannel.Writer.Complete();
                });

                yield return new NodeEvent<TState>
                {
                    NodeName = currentNode,
                    State = state,
                    StreamTokens = ReadChannel(tokenChannel.Reader)
                };

                await producer;
                state = finalState!;
            }
            else if (_nodes.TryGetValue(currentNode, out var handler))
            {
                state = await handler(state);
                yield return new NodeEvent<TState>
                {
                    NodeName = currentNode,
                    State = state
                };
            }

            currentNode = ResolveNextNode(currentNode, state);
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    private string ResolveNextNode(string current, TState state)
    {
        if (_conditionalEdges.TryGetValue(current, out var cond))
        {
            var key = cond.Router(state);
            return cond.Map.TryGetValue(key, out var mapped) ? mapped : GraphConstants.END;
        }
        return _edges.TryGetValue(current, out var next) ? next : GraphConstants.END;
    }

    private static async IAsyncEnumerable<string> ReadChannel(
        System.Threading.Channels.ChannelReader<string> reader)
    {
        await foreach (var item in reader.ReadAllAsync())
            yield return item;
    }
}
