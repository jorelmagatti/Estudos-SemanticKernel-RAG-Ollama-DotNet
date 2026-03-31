using Microsoft.Data.Sqlite;
using Microsoft.SemanticKernel;
using System.ComponentModel;
using System.Text;

namespace AgenteEmailHitlMemorySematicKernel;

// ════════════════════════════════════════════════════════════════════════════
//  MemoryStore — equivalente ao InMemoryStore do LangGraph +
//  manage_memory_tool e search_memory_tool do LangMem
//
//  Python:                              C#:
//  store = InMemoryStore(index=...)     MemoryStore (SQLite)
//  manage_memory_tool = create_...     ManageMemoryPlugin.ManageMemory()
//  search_memory_tool  = create_...    ManageMemoryPlugin.SearchMemory()
//  namespace=(user_id, "collection")   UserId + namespace no SQLite
// ════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Armazena memórias em SQLite com busca por keyword.
/// Substitui o InMemoryStore do LangGraph, adicionando persistência entre sessões.
/// </summary>
public class MemoryStore : IDisposable
{
    private readonly SqliteConnection _conn;

    public MemoryStore(string dbPath = "memory.db")
    {
        _conn = new SqliteConnection($"Data Source={dbPath}");
        _conn.Open();
        InitSchema();
    }

    private void InitSchema()
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            CREATE TABLE IF NOT EXISTS memories (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                namespace  TEXT NOT NULL DEFAULT 'collection',
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_memories_user
                ON memories(user_id, namespace);
            """;
        cmd.ExecuteNonQuery();
    }

    /// <summary>
    /// Cria uma nova memória.
    /// Equivalente ao manage_memory_tool.invoke({"action": "create", "content": ...})
    /// </summary>
    public string Create(string userId, string content, string ns = "collection")
    {
        var id = Guid.NewGuid().ToString();
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            INSERT INTO memories (id, user_id, namespace, content)
            VALUES ($id, $uid, $ns, $content)
            """;
        cmd.Parameters.AddWithValue("$id", id);
        cmd.Parameters.AddWithValue("$uid", userId);
        cmd.Parameters.AddWithValue("$ns", ns);
        cmd.Parameters.AddWithValue("$content", content);
        cmd.ExecuteNonQuery();
        return $"Memória criada com ID: {id}";
    }

    /// <summary>
    /// Busca memórias por similaridade de keywords.
    /// Equivalente ao search_memory_tool.invoke({"query": ...})
    /// Como não usamos embeddings vetoriais aqui, fazemos busca por tokens
    /// (abordagem pragmática para Ollama local sem servidor de embeddings).
    /// </summary>
    public List<MemoryEntry> Search(string userId, string query, int topK = 5)
    {
        var tokens = query.ToLower()
            .Split(' ', StringSplitOptions.RemoveEmptyEntries)
            .Where(t => t.Length > 3)
            .ToArray();

        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            SELECT id, user_id, namespace, content, created_at
            FROM memories
            WHERE user_id = $uid
            ORDER BY created_at DESC
            LIMIT 50
            """;
        cmd.Parameters.AddWithValue("$uid", userId);

        var all = new List<MemoryEntry>();
        using var r = cmd.ExecuteReader();
        while (r.Read())
            all.Add(new MemoryEntry
            {
                Id = r.GetString(0),
                UserId = r.GetString(1),
                Namespace = r.GetString(2),
                Content = r.GetString(3),
                CreatedAt = DateTime.Parse(r.GetString(4))
            });

        // Ranqueia por quantidade de tokens encontrados no conteúdo
        return all
            .Select(m => (entry: m, score: tokens.Count(t =>
                m.Content.ToLower().Contains(t))))
            .Where(x => x.score > 0)
            .OrderByDescending(x => x.score)
            .Take(topK)
            .Select(x => x.entry)
            .ToList();
    }

    /// <summary>Retorna todas as memórias de um usuário.</summary>
    public List<MemoryEntry> GetAll(string userId)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            SELECT id, user_id, namespace, content, created_at
            FROM memories WHERE user_id = $uid
            ORDER BY created_at DESC
            """;
        cmd.Parameters.AddWithValue("$uid", userId);
        var list = new List<MemoryEntry>();
        using var r = cmd.ExecuteReader();
        while (r.Read())
            list.Add(new MemoryEntry
            {
                Id = r.GetString(0),
                UserId = r.GetString(1),
                Namespace = r.GetString(2),
                Content = r.GetString(3),
                CreatedAt = DateTime.Parse(r.GetString(4))
            });
        return list;
    }

    public void Dispose() => _conn.Dispose();
}

/// <summary>
/// Plugin SK com as ferramentas de memória.
/// Equivalente ao manage_memory_tool e search_memory_tool do LangMem.
/// </summary>
public class MemoryPlugin
{
    private readonly MemoryStore _store;
    private string _currentUserId = "default";

    public MemoryPlugin(MemoryStore store) => _store = store;

    public void SetUserId(string userId) => _currentUserId = userId;

    /// <summary>
    /// Gerencia memórias — cria, atualiza ou deleta entradas.
    /// Equivalente ao manage_memory_tool do LangMem.
    /// </summary>
    [KernelFunction("manage_memory")]
    [Description("Armazena ou atualiza informações importantes na memória para referência futura. Use para salvar fatos sobre pessoas, compromissos, preferências ou tarefas de acompanhamento.")]
    public string ManageMemory(
        [Description("Ação a realizar: 'create' para criar nova memória")] string action,
        [Description("Conteúdo da memória a ser armazenado")] string content)
    {
        Console.ForegroundColor = ConsoleColor.DarkMagenta;
        Console.WriteLine($"\n  🧠 [manage_memory] action={action}");
        Console.WriteLine($"     Conteúdo: {content[..Math.Min(80, content.Length)]}...");
        Console.ResetColor();

        return action.ToLower() switch
        {
            "create" => _store.Create(_currentUserId, content),
            _ => $"Ação '{action}' não suportada. Use 'create'."
        };
    }

    /// <summary>
    /// Busca memórias relevantes.
    /// Equivalente ao search_memory_tool do LangMem.
    /// </summary>
    [KernelFunction("search_memory")]
    [Description("Busca informações relevantes na memória sobre pessoas, reuniões anteriores, preferências ou contexto de interações passadas.")]
    public string SearchMemory(
        [Description("Texto de busca para encontrar memórias relevantes")] string query)
    {
        Console.ForegroundColor = ConsoleColor.DarkMagenta;
        Console.WriteLine($"\n  🔍 [search_memory] query: {query}");
        Console.ResetColor();

        var results = _store.Search(_currentUserId, query);

        if (results.Count == 0)
            return "Nenhuma memória relevante encontrada.";

        var formatted = results
            .Select((m, i) => $"[{i + 1}] {m.Content} (salvo em {m.CreatedAt:dd/MM/yyyy HH:mm})")
            .ToList();

        var output = string.Join("\n", formatted);

        Console.ForegroundColor = ConsoleColor.DarkMagenta;
        Console.WriteLine($"     {results.Count} memória(s) encontrada(s)");
        Console.ResetColor();

        return output;
    }
}
