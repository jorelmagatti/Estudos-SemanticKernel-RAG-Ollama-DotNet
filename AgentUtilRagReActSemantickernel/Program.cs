using AgentUtilRagReActSemantickernel;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

class Program
{
    // ─── Configurações ────────────────────────────────────────────────────────
    const string LLM_MODEL = "llama3.2";                    // ou llama3.1, gemma3
    const string OLLAMA_URL = "http://localhost:11434";
    const int MAX_ITER = 5;                            // max_iterations do Python
    // ─────────────────────────────────────────────────────────────────────────

    static async Task Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Console.WriteLine("╔══════════════════════════════════════════════════╗");
        Console.WriteLine("║       🤖 Agente ReAct — Inventário              ║");
        Console.WriteLine("║   Pensamento → Ação → Observação → Resposta     ║");
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
        var agente = new ReActAgent(chatService, MAX_ITER);

        // Função auxiliar: executa o agente com retry e exibe o resultado formatado
        // Equivalente ao bloco do Python:
        //   resposta_N = run_react_agent(pergunta_N)
        //   print(f"**RESPOSTA FINAL DO AGENTE N:** {resposta_N}")
        async Task ExecutarInteracao(string label, string pergunta)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine($"\n**{label}: {pergunta}**");
            Console.ResetColor();

            string respostaFinal = string.Empty;

            try
            {
                respostaFinal = await agente.ExecutarAsync(pergunta);
            }
            catch (Exception ex)
            {
                respostaFinal = $"Erro: {ex.Message}";
            }

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\n**RESPOSTA FINAL DO AGENTE:** {respostaFinal}");
            Console.ResetColor();
            Console.WriteLine("\n" + new string('=', 80) + "\n");
        }

        // ══════════════════════════════════════════════════════════════════════
        // INTERAÇÕES PRÉ-DEFINIDAS
        // Equivalente ao bloco final do notebook Python com as 5 perguntas
        // ══════════════════════════════════════════════════════════════════════

        Console.WriteLine("--- Começando as Interações com o Agente ReAct ---\n");

        // Interação 1: Consultar Estoque
        await ExecutarInteracao("Interação 1", "Quantos teclados temos em estoque?");

        // Interação 2: Consultar Preço
        await ExecutarInteracao("Interação 2", "Qual o preço de um headset?");

        // Interação 3: Item não encontrado
        await ExecutarInteracao("Interação 3", "Temos cadeiras em estoque?");

        // Interação 4: Produto mais caro (ferramenta sem argumento)
        await ExecutarInteracao("Interação 4", "Qual é o produto mais caro?");

        // Interação 5: Calcular valor total de lista (ferramenta com múltiplos itens)
        await ExecutarInteracao("Interação 5", "Qual o valor de um teclado, uma impressora e uma webcam?");

        Console.WriteLine("--- Fim das Interações ---\n");

        // ══════════════════════════════════════════════════════════════════════
        // MODO INTERATIVO
        // Equivalente à função Python: iniciar_conversacao_com_agente()
        // ══════════════════════════════════════════════════════════════════════

        Console.WriteLine("╔══════════════════════════════════════════════════╗");
        Console.WriteLine("║   💬 Modo Interativo — Agente de Inventário      ║");
        Console.WriteLine("║  Digite sua pergunta ou 'sair' para encerrar.   ║");
        Console.WriteLine("╚══════════════════════════════════════════════════╝\n");

        Console.WriteLine("Produtos disponíveis: monitor, teclado, mouse gamer, webcam, headset, impressora\n");

        while (true)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.Write("Você: ");
            Console.ResetColor();

            var entrada = Console.ReadLine()?.Trim();

            if (string.IsNullOrEmpty(entrada)) continue;
            if (entrada.Equals("sair", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine("Encerrando a conversa. Até logo!");
                break;
            }

            Console.WriteLine("\nAgente: Processando...");

            string resposta = string.Empty;

            try
            {
                 resposta = await agente.ExecutarAsync(entrada);
            }
            catch (Exception ex)
            {
                resposta = $"Ocorreu um erro ao processar sua pergunta: {ex.Message}";
            }

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\nAgente: {resposta}\n");
            Console.ResetColor();
        }
    }
}