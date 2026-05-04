using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Ollama;
using ModelContextProtocol.Client;

namespace McpOllamaClient;

/// <summary>
/// Encapsula toda a lógica de:
///   1. Conectar ao servidor MCP via HTTP (SSE/Streamable)
///   2. Importar as tools MCP como KernelFunctions do Semantic Kernel
///   3. Rodar um loop de chat com Ollama usando function-calling
/// </summary>
public sealed class McpAgent
{
    private readonly string _mcpServerUrl;
    private readonly string _ollamaModel;

    private Kernel _kernel = null!;
    private IChatCompletionService _chat = null!;
    private ChatHistory _history = null!;
    private McpClient _mcpClient = null!;

    public McpAgent(string mcpServerUrl, string ollamaModel)
    {
        _mcpServerUrl = mcpServerUrl;
        _ollamaModel = ollamaModel;
    }

    // ─────────────────────────────────────────────────────────
    //  Inicialização
    // ─────────────────────────────────────────────────────────
    public async Task InitializeAsync()
    {
        // 1. Conecta ao servidor MCP via HTTP (transport Streamable/SSE stateless)
        Console.Write("🔌 Conectando ao servidor MCP ... ");
        _mcpClient = await ConnectToMcpServerAsync();
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("Conectado ✅");
        Console.ResetColor();

        // 2. Lista as tools disponíveis no servidor
        var tools = await _mcpClient.ListToolsAsync();

        Console.ForegroundColor = ConsoleColor.Magenta;
        Console.WriteLine($"\n📦 Tools MCP disponíveis ({tools.Count}):");
        Console.ResetColor();
        foreach (var tool in tools)
        {
            Console.WriteLine($"   • {tool.Name} — {tool.Description}");
        }
        Console.WriteLine();

        // 3. Monta o Kernel do Semantic Kernel com Ollama
        Console.Write($"🦙 Inicializando Ollama (modelo: {_ollamaModel}) ... ");
        _kernel = BuildKernel(tools);
        _chat = _kernel.GetRequiredService<IChatCompletionService>();
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("Pronto ✅");
        Console.ResetColor();
        Console.WriteLine();

        // 4. System prompt explicando ao modelo o que ele pode fazer
        _history = new ChatHistory();
        _history.AddSystemMessage(BuildSystemPrompt(tools));
    }

    // ─────────────────────────────────────────────────────────
    //  Loop de chat interativo
    // ─────────────────────────────────────────────────────────
    public async Task RunChatLoopAsync()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("💬 Chat iniciado! Digite sua mensagem (ou 'sair' para encerrar).");
        Console.WriteLine("   Dicas de prompts:");
        Console.WriteLine("   → \"Olá, meu nome é João\"");
        Console.WriteLine("   → \"Quanto é 42 + 58?\"");
        Console.WriteLine("   → \"Some 123 e 456 por favor\"");
        Console.ResetColor();
        Console.WriteLine();

        while (true)
        {
            // Lê entrada do usuário
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write("Você: ");
            Console.ResetColor();

            var userInput = Console.ReadLine()?.Trim();

            if (string.IsNullOrEmpty(userInput))
                continue;

            if (userInput.Equals("sair", StringComparison.OrdinalIgnoreCase) ||
                userInput.Equals("exit", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine("\n👋 Encerrando. Até logo!");
                break;
            }

            _history.AddUserMessage(userInput);

            // Ollama costuma devolver tool_calls no primeiro turno; é preciso executar as
            // funções e chamar o modelo de novo (mesmo padrão ReAct dos outros agentes no repo).
#pragma warning disable SKEXP0070
            var executionSettings = new OllamaPromptExecutionSettings
            {
                Temperature = 0.2f,
                FunctionChoiceBehavior = FunctionChoiceBehavior.Auto()
            };
#pragma warning restore SKEXP0070

            try
            {
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.Write("\n🤔 Pensando");

                const int maxIterations = 8;
                var iteration = 0;
                string assistantReply = string.Empty;

                while (iteration++ < maxIterations)
                {
                    var response = await _chat.GetChatMessageContentAsync(
                        _history,
                        executionSettings,
                        _kernel);

                    var functionCalls = FunctionCallContent.GetFunctionCalls(response).ToList();
                    if (functionCalls.Count > 0)
                    {
                        _history.Add(response);
                        var toolResults = new ChatMessageContentItemCollection();
                        foreach (var fc in functionCalls)
                        {
                            var result = await fc.InvokeAsync(_kernel);
                            toolResults.Add(result);
                        }

                        _history.Add(new ChatMessageContent(AuthorRole.Tool, toolResults));
                        continue;
                    }

                    // Llama 3.1 (e similares) costumam colocar a "tool call" como JSON no texto, não em tool_calls.
                    if (TryParsePlainTextToolJson(response.Content, out var jsonTool, out var jsonArgs) &&
                        TryResolveMcpFunction(jsonTool, out var resolvedJson))
                    {
                        _history.Add(response);
                        await AppendManualToolResultsAsync(resolvedJson, jsonArgs);
                        continue;
                    }

                    // Llama 3.2 frequentemente responde em prosa sem disparar tools; forçamos helloworld quando a intenção é clara.
                    // Só na 1ª ida ao LLM deste turno (iteration já foi incrementado no while): iteration == 1.
                    if (iteration == 1 &&
                        TryInferNomeApresentacao(userInput, out var nomeInferido) &&
                        TryResolveMcpFunction("helloworld", out var resolvedHello))
                    {
                        await AppendManualToolResultsAsync(resolvedHello, new KernelArguments { ["nome"] = nomeInferido });
                        continue;
                    }

                    _history.Add(response);
                    assistantReply = response.Content ?? string.Empty;
                    if (string.IsNullOrWhiteSpace(assistantReply))
                        assistantReply = "(sem resposta)";
                    break;
                }

                Console.Write("\r                    \r"); // limpa "Pensando..."
                Console.ResetColor();

                Console.ForegroundColor = ConsoleColor.Green;
                Console.Write("Assistente: ");
                Console.ResetColor();
                Console.WriteLine(assistantReply);
                Console.WriteLine();
            }
            catch (Exception ex)
            {
                Console.Write("\r                    \r");
                Console.ResetColor();
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"\n❌ Erro ao chamar o modelo: {ex.Message}");
                Console.ResetColor();

                if (ex.Message.Contains("connect", StringComparison.OrdinalIgnoreCase) ||
                    ex.Message.Contains("refused", StringComparison.OrdinalIgnoreCase))
                {
                    Console.WriteLine("   Verifique se o Ollama está rodando: ollama serve");
                    Console.WriteLine($"   E se o modelo está disponível: ollama pull {_ollamaModel}");
                }
                Console.WriteLine();
            }
        }

        // Cleanup
        if (_mcpClient is IAsyncDisposable d)
            await d.DisposeAsync();
    }

    // ─────────────────────────────────────────────────────────
    //  Helpers privados
    // ─────────────────────────────────────────────────────────

    /// <summary>
    /// Conecta ao servidor MCP via HTTP transport (Streamable/SSE stateless).
    /// </summary>
    private async Task<McpClient> ConnectToMcpServerAsync()
    {
        // O servidor usa app.MapMcp() que expõe o endpoint em /mcp por padrão
        var mcpEndpoint = new Uri($"{_mcpServerUrl}/mcp");

        var transport = new HttpClientTransport(new HttpClientTransportOptions
        {
            Endpoint = mcpEndpoint,
            Name = "MCP Server Local",
            // Servidor usa Streamable HTTP stateless (só POST /mcp)
            // AutoDetect faz GET /mcp primeiro (SSE probe) → 404
            // Forçar StreamableHttp evita o GET desnecessário
            TransportMode = HttpTransportMode.StreamableHttp,
        });

        var client = await McpClient.CreateAsync(transport);
        return client;
    }

    /// <summary>
    /// Constrói o Kernel do Semantic Kernel com:
    ///   - Ollama como backend de chat
    ///   - Tools MCP importadas como KernelPlugin
    /// </summary>
    private Kernel BuildKernel(IList<ModelContextProtocol.Client.McpClientTool> mcpTools)
    {
        var builder = Kernel.CreateBuilder();

        // Configura logging (só warnings para não poluir o console)
        builder.Services.AddLogging(l =>
            l.AddConsole().SetMinimumLevel(LogLevel.Warning));

        // Adiciona Ollama como provedor de chat
        // Ollama roda por padrão em http://localhost:11434
        builder.AddOllamaChatCompletion(
            modelId: _ollamaModel,
            endpoint: new Uri("http://localhost:11434"));

        var kernel = builder.Build();

        // Importa as MCP tools como um plugin nativo do Semantic Kernel
        // Cada tool vira uma KernelFunction que o modelo pode invocar
        kernel.ImportPluginFromFunctions(
            "McpTools",
            mcpTools.Select(tool => tool.AsKernelFunction()).ToList());

        return kernel;
    }

    /// <summary>
    /// Gera o system prompt listando as tools disponíveis.
    /// </summary>
    private static string BuildSystemPrompt(IList<McpClientTool> tools)
    {
        var lines = tools.Select(t => $"  - {t.Name}: {t.Description}");
        var toolBlock = string.Join(Environment.NewLine, lines);
        return $"""
            Você é um assistente com acesso a ferramentas MCP registradas no plugin "McpTools".
            
            Ferramentas (use o nome exato ao chamar a função; não use underscore no nome):
            {toolBlock}

            Regras obrigatórias:
            - Use apenas o mecanismo de ferramentas do sistema (function calling). Não escreva JSON de ferramenta no texto da resposta.
            - Se o usuário disser que o nome dele é X, ou "meu nome é X", ou "sou o X", chame helloworld com nome=X (só o primeiro nome).
            - Para soma de dois inteiros, chame somar com a e b.
            - Depois de receber o resultado de uma ferramenta, responda em português do Brasil incorporando esse resultado.
            - Se nenhuma ferramenta se aplicar, responda em texto normal.
            """;
    }

    /// <summary>
    /// Alguns modelos Ollama devolvem um único JSON no corpo: { "name": "...", "parameters": { ... } }.
    /// </summary>
    private static bool TryParsePlainTextToolJson(string? content, out string toolName, out KernelArguments args)
    {
        toolName = string.Empty;
        args = new KernelArguments();

        if (string.IsNullOrWhiteSpace(content))
            return false;

        var t = content.Trim();
        if (!t.StartsWith('{'))
            return false;

        try
        {
            using var doc = JsonDocument.Parse(t);
            var root = doc.RootElement;
            if (!root.TryGetProperty("name", out var nameEl))
                return false;

            toolName = nameEl.GetString() ?? string.Empty;
            if (string.IsNullOrEmpty(toolName))
                return false;

            JsonElement props = default;
            if (root.TryGetProperty("parameters", out var p))
                props = p;
            else if (root.TryGetProperty("arguments", out var a))
                props = a;
            else
                return true;

            if (props.ValueKind != JsonValueKind.Object)
                return true;

            foreach (var prop in props.EnumerateObject())
                args[prop.Name] = JsonElementToArgValue(prop.Value);

            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    private static object JsonElementToArgValue(JsonElement el) =>
        el.ValueKind switch
        {
            JsonValueKind.String => el.GetString() ?? string.Empty,
            JsonValueKind.Number => el.TryGetInt32(out var i) ? i : el.GetDouble(),
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.Null => string.Empty,
            _ => el.ToString()
        };

    private async Task AppendManualToolResultsAsync(string functionName, KernelArguments args)
    {
        var fr = await _kernel.InvokeAsync("McpTools", functionName, args);
        var value = fr.GetValue<object?>();
        var payload = value ?? fr.ToString();
        var toolResults = new ChatMessageContentItemCollection
        {
            new FunctionResultContent(Guid.NewGuid().ToString(), "McpTools", functionName, payload)
        };
        _history.Add(new ChatMessageContent(AuthorRole.Tool, toolResults));
    }

    /// <summary>
    /// Corresponde nomes que o modelo inventa (ex.: hello_world) ao nome real no plugin (helloworld).
    /// </summary>
    private bool TryResolveMcpFunction(string modelName, out string functionName)
    {
        functionName = string.Empty;
        if (!_kernel.Plugins.TryGetPlugin("McpTools", out var plugin))
            return false;

        if (plugin.TryGetFunction(modelName, out _))
        {
            functionName = modelName;
            return true;
        }

        var norm = NormalizeToolName(modelName);
        foreach (var f in plugin)
        {
            if (NormalizeToolName(f.Name) == norm)
            {
                functionName = f.Name;
                return true;
            }
        }

        return false;
    }

    private static string NormalizeToolName(string name) =>
        Regex.Replace(name, @"[\s_\-]+", string.Empty, RegexOptions.IgnoreCase).ToLowerInvariant();

    /// <summary>PT-BR: extrai primeiro nome em frases do tipo "meu nome é Jorel".</summary>
    private static bool TryInferNomeApresentacao(string userText, out string nome)
    {
        nome = string.Empty;
        if (string.IsNullOrWhiteSpace(userText))
            return false;

        var m = Regex.Match(userText,
            @"(?i)(?:meu\s+nome\s*[ée]\s*|(?:eu\s+)?sou\s+(?:o\s+|a\s+)?|chamo-me\s+)([A-Za-zÀ-ÿ]+)",
            RegexOptions.CultureInvariant);
        if (!m.Success)
            return false;

        nome = m.Groups[1].Value;
        return nome.Length >= 2;
    }
}