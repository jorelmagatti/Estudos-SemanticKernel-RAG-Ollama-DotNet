using McpOllamaClient;

// ─── Configuração ──────────────────────────────────────────────────────────
// URL do MeuMcpServer (roda localmente via HTTP)
const string mcpServerUrl = "http://localhost:60672";

// Modelo Ollama com suporte a function-calling (tool_call)
// Outros bons candidatos: llama3.1, mistral-nemo, qwen2.5
const string ollamaModel = "llama3.1";

// ─── Banner ────────────────────────────────────────────────────────────────
Console.ForegroundColor = ConsoleColor.Cyan;
Console.WriteLine("╔═══════════════════════════════════════════════════╗");
Console.WriteLine("║         MCP Cliente  •  Semantic Kernel + Ollama  ║");
Console.WriteLine("╚═══════════════════════════════════════════════════╝");
Console.ResetColor();
Console.WriteLine();
Console.WriteLine($"  Servidor MCP : {mcpServerUrl}");
Console.WriteLine($"  Modelo Ollama: {ollamaModel}");
Console.WriteLine();

// ─── Inicializa e executa o agente ─────────────────────────────────────────
var agent = new McpAgent(mcpServerUrl, ollamaModel);

try
{
    await agent.InitializeAsync();
    await agent.RunChatLoopAsync();
}
catch (HttpRequestException ex)
{
    Console.ForegroundColor = ConsoleColor.Red;
    Console.WriteLine($"\n❌ Não foi possível conectar ao servidor MCP.");
    Console.WriteLine($"   Erro: {ex.Message}");
    Console.ResetColor();
    Console.WriteLine();
    Console.WriteLine("   ▶ Certifique-se de que o MeuMcpServer está rodando:");
    Console.WriteLine($"     cd MeuMcpServer && dotnet run");
    Console.WriteLine($"     (deve ouvir em {mcpServerUrl})");
    Environment.Exit(1);
}
catch (Exception ex)
{
    Console.ForegroundColor = ConsoleColor.Red;
    Console.WriteLine($"\n❌ Erro inesperado: {ex.Message}");
    Console.ResetColor();
    Environment.Exit(1);
}
