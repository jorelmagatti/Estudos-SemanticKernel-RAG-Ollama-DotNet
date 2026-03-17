using AgenteUtilidadesRagReActGrafo;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Polly;
using Polly.Retry;

namespace AgenteGrafo;

class Program
{
    // ─── Configurações ────────────────────────────────────────────────────────
    const string LLM_MODEL = "llama3.2";               // ou llama3.1, gemma3
    const string OLLAMA_URL = "http://localhost:11434";
    // ─────────────────────────────────────────────────────────────────────────

    static async Task Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Console.WriteLine("╔══════════════════════════════════════════════════╗");
        Console.WriteLine("║    🔍 Agente de Pesquisa — Grafo de Estados      ║");
        Console.WriteLine("║    LangGraph → Semantic Kernel + Ollama          ║");
        Console.WriteLine("╚══════════════════════════════════════════════════╝\n");

        // ── Configura Kernel + Ollama ──────────────────────────────────────────
        var httpClient = new HttpClient(new SocketsHttpHandler
        {
            KeepAlivePingPolicy = HttpKeepAlivePingPolicy.WithActiveRequests,
            KeepAlivePingDelay = TimeSpan.FromSeconds(15),
            KeepAlivePingTimeout = TimeSpan.FromSeconds(15),
        })
        { Timeout = TimeSpan.FromMinutes(10) };

        var builder = Kernel.CreateBuilder();
        builder.Services.AddSingleton(httpClient);

#pragma warning disable SKEXP0070
        builder.AddOllamaChatCompletion(LLM_MODEL, new Uri(OLLAMA_URL));
#pragma warning restore SKEXP0070

        var kernel = builder.Build();
        var chatService = kernel.Services.GetRequiredService<IChatCompletionService>();

        // Ferramenta de busca — substitui TavilySearchResults(max_results=4)
        var httpBusca = new HttpClient { Timeout = TimeSpan.FromSeconds(15) };
        var ferramenta = new BuscaWebTool("sua chave aqui", maxResultados: 4);

        // ── System Prompt ──────────────────────────────────────────────────────
        // Equivalente ao prompt do Python com data atual injetada
        var dataAtual = DateTime.Now.ToString("dd/MM/yyyy");

        var systemPrompt = $"""
            Você é um assistente de pesquisa inteligente e altamente atualizado.
            Sua principal prioridade é encontrar as informações mais RECENTES sempre que possível.
            A data atual é {dataAtual}.
            Ao buscar sobre o tempo ou eventos que se referem a "hoje" ou "agora",
            inclua a data atual '{dataAtual}' na sua consulta para a ferramenta de busca.
            Por exemplo, se a pergunta é "tempo em cidade x hoje",
            a consulta para a ferramenta deve ser "tempo em cidade x {dataAtual}".
            Você tem permissão para fazer múltiplas chamadas (seja em conjunto ou em sequência).
            Procure informações apenas quando tiver certeza do que você quer.
            Se precisar pesquisar alguma informação antes de fazer uma pergunta de acompanhamento,
            você tem permissão para fazer isso!
            """;

        // Instancia o grafo
        // Equivalente a: abot = Agent(model_instance, [tool_instance], system=prompt)
        var agente = new GrafoAgente(chatService, ferramenta, systemPrompt, maxIteracoes: 10);

        // Polly retry
        var retryPolicy = new ResiliencePipelineBuilder()
            .AddRetry(new RetryStrategyOptions
            {
                MaxRetryAttempts = 3,
                Delay = TimeSpan.FromSeconds(2),
                BackoffType = DelayBackoffType.Exponential,
                ShouldHandle = new PredicateBuilder()
                    .Handle<TaskCanceledException>()
                    .Handle<TimeoutException>()
                    .Handle<HttpRequestException>(),
                OnRetry = retryArgs =>
                {
                    Console.WriteLine($"  ⚠️  Timeout — tentativa {retryArgs.AttemptNumber + 1}/3...");
                    return ValueTask.CompletedTask;
                }
            })
            .Build();

        // Função auxiliar de execução com log formatado
        // Equivalente ao bloco:
        //   for s in abot.graph.stream({"messages": messages}): print(s)
        //   print(final_result_state['llm']['messages'][-1].content)
        async Task ExecutarConsulta(string label, string pergunta)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine($"\n{'─',48}");
            Console.WriteLine($"  {label}: {pergunta}");
            Console.WriteLine($"{'─',48}");
            Console.ResetColor();

            Console.WriteLine("Agente: Pensando e buscando...");

            string resultado = string.Empty;
            try
            {
                await retryPolicy.ExecuteAsync(async ct =>
                {
                    resultado = await agente.InvocarAsync(pergunta);
                });
            }
            catch (Exception ex)
            {
                resultado = $"Erro: {ex.Message}";
            }

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\n\n--- Resultado Final ---");
            Console.WriteLine(resultado);
            Console.ResetColor();
            Console.WriteLine(new string('=', 80));
        }

        // ══════════════════════════════════════════════════════════════════════
        // INTERAÇÕES PRÉ-DEFINIDAS DO NOTEBOOK
        // ══════════════════════════════════════════════════════════════════════

        Console.WriteLine("Iniciando interações com o Agente:\n");

        // Interação 1 — tempo hoje
        // Equivalente a: messages = [HumanMessage("Como está o tempo em São Paulo hoje?")]
        await ExecutarConsulta("Interação 1", "Como está o tempo em São Paulo hoje?");

        // Interação 2 — tempo amanhã
        await ExecutarConsulta("Interação 2", "Como está o tempo em São Paulo amanhã?");

        // Interação 3 — tempo ontem
        await ExecutarConsulta("Interação 3", "Como foi o tempo em São Paulo ontem?");

        // Interação 4 — duas cidades
        await ExecutarConsulta("Interação 4", "Como está o tempo em São Paulo e no Rio de Janeiro hoje?");

        // Interação 5 — pergunta histórica complexa (múltiplas buscas)
        // Equivalente à query_passado do notebook
        await ExecutarConsulta("Interação 5",
            "Qual país sediou a Copa do Mundo de futebol em 1998? " +
            "Quem foi o campeão e qual o placar da final? " +
            "Qual a capital desse país e qual sua moeda atual? " +
            "Responda cada pergunta separadamente.");

        // ══════════════════════════════════════════════════════════════════════
        // MODO INTERATIVO
        // Equivalente à função iniciar_conversacao_com_agente() do notebook
        // ══════════════════════════════════════════════════════════════════════

        Console.WriteLine("\n╔══════════════════════════════════════════════════╗");
        Console.WriteLine("║   💬 Agente de Pesquisa Interativo               ║");
        Console.WriteLine("║  Digite sua pergunta ou 'sair' para encerrar.   ║");
        Console.WriteLine("╚══════════════════════════════════════════════════╝\n");

        while (true)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.Write("\nVocê: ");
            Console.ResetColor();

            var entrada = Console.ReadLine()?.Trim();
            if (string.IsNullOrEmpty(entrada)) continue;
            if (entrada.Equals("sair", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine("Agente: Encerrando a conversa. Até logo!");
                break;
            }

            Console.WriteLine("\nAgente: Pensando e buscando...");

            string resposta = string.Empty;
            try
            {
                await retryPolicy.ExecuteAsync(async ct =>
                {
                    resposta = await agente.InvocarAsync(entrada);
                });
            }
            catch (Exception ex)
            {
                resposta = $"Ocorreu um erro: {ex.Message}";
            }

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\nAgente:\n{resposta}");
            Console.ResetColor();
        }

        Console.WriteLine("\n--- Conversa Encerrada ---");
    }
}