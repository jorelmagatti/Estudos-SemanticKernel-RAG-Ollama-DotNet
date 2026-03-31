using AgenteEmailHitlMemorySematicKernel;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;

Console.OutputEncoding = System.Text.Encoding.UTF8;
ConsoleRenderer.PrintBanner();

// ── Configuração ──────────────────────────────────────────────────────────────
var ollamaUrl = "http://localhost:11434";
var ollamaModel = "llama3.2";
var dbPath = "memory.db";

Console.ForegroundColor = ConsoleColor.DarkGray;
Console.WriteLine($"  Ollama : {ollamaUrl}  |  Modelo: {ollamaModel}");
Console.WriteLine($"  Memória: {Path.GetFullPath(dbPath)}");
Console.ResetColor();

// ── Kernel ────────────────────────────────────────────────────────────────────
var builder = Kernel.CreateBuilder();
builder.Services.AddHttpClient("ollama", c =>
{
    c.BaseAddress = new Uri(ollamaUrl);
    c.Timeout = TimeSpan.FromMinutes(10);
});
builder.AddOllamaChatCompletion(ollamaModel, new Uri(ollamaUrl));
var kernel = builder.Build();

// ── Perfil e regras — equivalente ao dict profile + prompt_instructions ────────
var profile = new UserProfile
{
    Name = "Sarah",
    FullName = "Sarah Chen",
    UserProfileBackground = "Engenheira de software sênior liderando uma equipe de 5 desenvolvedores"
};

var triageRules = new TriageRules
{
    Ignore = "Newsletters, spam, comunicados gerais da empresa",
    Notify = "Membro da equipe doente, notificações do sistema de build, atualizações de status de projeto",
    Respond = "Perguntas diretas de membros da equipe, solicitações de reunião, relatórios de bugs críticos"
};

var agentInstructions = """
    Você é um assistente executivo altamente eficiente.
    Execute as tarefas solicitadas de forma direta, sem diálogos desnecessários.
 
    Tarefas:
    - SEMPRE use search_memory antes de responder para verificar contexto anterior
    - Responder ao e-mail de entrada
    - Salvar tarefa de acompanhamento na memória com manage_memory
    - Sugerir próxima ação quando relevante
    """;

// ── Memória SQLite — equivalente ao InMemoryStore do LangGraph ────────────────
using var memStore = new MemoryStore(dbPath);
var orchestrator = new EmailAgentOrchestrator(
    kernel, profile, triageRules, agentInstructions, memStore);

// ── Estado da sessão ──────────────────────────────────────────────────────────
// Equivalente ao config = {"configurable": {"langgraph_user_id": "lance"}}
var currentUserId = Environment.GetEnvironmentVariable("USER_ID") ?? "lance";

Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"\n  ✅ Pronto! Usuário: '{currentUserId}'");
Console.ResetColor();

// ── Pré-popula memória — equivalente à célula 8 do notebook ───────────────────
var seedContent = "Acompanhamento necessário: Alice Smith perguntou sobre os endpoints de API ausentes " +
                  "na documentação do serviço de autenticação (/auth/refresh e /auth/validate). " +
                  "Sarah precisa revisar e esclarecer se foram intencionalmente omitidos.";

var seedResult = orchestrator.SeedMemory(currentUserId, seedContent);
Console.ForegroundColor = ConsoleColor.DarkMagenta;
Console.WriteLine($"  🧠 Memória inicial carregada: {seedResult}");
Console.ResetColor();

// ── E-mails de teste — equivalentes às células 7-8 do notebook ───────────────
var testEmails = new[]
{
    // E-mail 1: Acompanhamento de Alice — deve RESPONDER + HITL (célula 8)
    new EmailInput
    {
        From    = "Alice Smith <alice.smith@company.com>",
        To      = "Sarah Chen <sarah.chen@company.com>",
        Subject = "Acompanhamento",
        Body    = "Olá Sarah, como está minha solicitação?"
    },
 
    // E-mail 2: Spam — deve IGNORAR
    new EmailInput
    {
        From    = "marketing@amazingdeals.com",
        To      = "Sarah Chen <sarah.chen@company.com>",
        Subject = "🔥 OFERTA EXCLUSIVA para desenvolvedores!",
        Body    = "Não perca! 80% de desconto por 24 horas! Clique aqui: https://amazingdeals.com"
    },
 
    // E-mail 3: Build quebrado — deve NOTIFICAR
    new EmailInput
    {
        From    = "ci@company.com",
        To      = "Sarah Chen <sarah.chen@company.com>",
        Subject = "Build #4521 falhou no branch main",
        Body    = "3 testes de integração falharam após o merge do PR #142."
    }
};

// ── Ajuda ─────────────────────────────────────────────────────────────────────
Console.ForegroundColor = ConsoleColor.DarkGray;
Console.WriteLine("""
  Comandos:
    1, 2, 3        — processar e-mail de teste
    /memorias      — ver todas as memórias do usuário atual
    /usuario <id>  — trocar usuário (contexto de memória separado)
    /email         — digitar e-mail manualmente
    /sair          — encerrar
""");
Console.ResetColor();

// ── Loop interativo ───────────────────────────────────────────────────────────
while (true)
{
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.Write($"\n[{currentUserId}] 📧 Escolha [1/2/3] ou comando: ");
    Console.ResetColor();

    var input = Console.ReadLine()?.Trim();
    if (string.IsNullOrWhiteSpace(input)) continue;
    if (input.Equals("/sair", StringComparison.OrdinalIgnoreCase)) break;

    if (input.Equals("/memorias", StringComparison.OrdinalIgnoreCase))
    {
        ConsoleRenderer.PrintMemories(orchestrator.GetAllMemories(currentUserId), currentUserId);
        continue;
    }

    if (input.StartsWith("/usuario ", StringComparison.OrdinalIgnoreCase))
    {
        currentUserId = input[9..].Trim();
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"  ↪ Usuário trocado para '{currentUserId}'");
        Console.ResetColor();
        continue;
    }

    EmailInput? email = input switch
    {
        "1" => testEmails[0],
        "2" => testEmails[1],
        "3" => testEmails[2],
        _ => null
    };

    if (input.Equals("/email", StringComparison.OrdinalIgnoreCase))
    {
        Console.Write("  De: "); var from = Console.ReadLine() ?? "";
        Console.Write("  Assunto: "); var subject = Console.ReadLine() ?? "";
        Console.Write("  Corpo:\n");
        var lines = new System.Text.StringBuilder();
        string? line;
        while (!string.IsNullOrEmpty(line = Console.ReadLine()))
            lines.AppendLine(line);
        email = new EmailInput
        {
            From = from,
            To = $"Sarah Chen <sarah.chen@company.com>",
            Subject = subject,
            Body = lines.ToString()
        };
    }

    if (email == null) { Console.WriteLine("  Opção inválida."); continue; }

    Console.ForegroundColor = ConsoleColor.DarkGray;
    Console.WriteLine($"\n  📨 De: {email.From}\n  📨 Assunto: {email.Subject}");
    Console.ResetColor();

    try
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();

        // ── 1. Pipeline principal (triage + response) ─────────────────────
        var stream = orchestrator.ProcessEmailAsync(email, currentUserId);
        await ConsoleRenderer.RenderAsync(stream);

        // ── 2. HITL — apenas para e-mails que receberam resposta ─────────
        // Equivalente à célula 8: human_in_the_loop_schedule(...)
        // Só executa se o e-mail foi classificado como "respond"
        // (verificamos pelo fato de FinalReply ter sido preenchido)
        // Para simplificar, pergunta sempre após o pipeline
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.Write("\n  Deseja executar o fluxo Human-in-the-Loop? (sim/não): ");
        Console.ResetColor();
        var runHitl = (Console.ReadLine() ?? "").Trim().ToLower();

        if (runHitl is "sim" or "s")
            await orchestrator.RunHitlAsync(email, currentUserId);

        sw.Stop();
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine($"\n  ⏱  Tempo: {sw.Elapsed.TotalSeconds:F1}s");
        Console.ResetColor();
    }
    catch (Exception ex)
    {
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"\n  ❌ Erro: {ex.Message}");
        Console.ResetColor();
    }
}

Console.ForegroundColor = ConsoleColor.DarkGray;
Console.WriteLine("\n  Encerrando... Até mais! 👋");
Console.ResetColor();