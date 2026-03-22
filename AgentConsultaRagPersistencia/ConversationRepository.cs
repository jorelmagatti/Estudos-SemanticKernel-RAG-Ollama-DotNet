using Microsoft.Data.Sqlite;

namespace AgentConsultaRagPersistencia;

/// <summary>
/// Repositório SQLite para histórico de conversas.
///
/// Equivalente ao SqliteSaver do LangGraph:
///   conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
///   memory = SqliteSaver(conn)
///
/// Cada conversa é identificada por um thread_id — conversas diferentes
/// têm históricos completamente isolados, exatamente como no notebook.
/// </summary>
public class ConversationRepository : IDisposable
{
    private readonly SqliteConnection _conn;

    public ConversationRepository(string dbPath = "conversations.db")
    {
        _conn = new SqliteConnection($"Data Source={dbPath}");
        _conn.Open();
        InitializeSchema();
    }

    // ── Schema ────────────────────────────────────────────────────────────────

    private void InitializeSchema()
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id  TEXT    NOT NULL,
                role       TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                tool_name  TEXT,
                created_at TEXT    NOT NULL DEFAULT (datetime('now'))
            );
 
            CREATE INDEX IF NOT EXISTS idx_messages_thread
                ON messages(thread_id, id);
            """;
        cmd.ExecuteNonQuery();
    }

    // ── Escrita ───────────────────────────────────────────────────────────────

    /// <summary>Persiste uma mensagem no histórico do thread.</summary>
    public void SaveMessage(string threadId, MessageRole role, string content, string? toolName = null)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            INSERT INTO messages (thread_id, role, content, tool_name)
            VALUES ($tid, $role, $content, $tool)
            """;
        cmd.Parameters.AddWithValue("$tid", threadId);
        cmd.Parameters.AddWithValue("$role", role.ToString());
        cmd.Parameters.AddWithValue("$content", content);
        cmd.Parameters.AddWithValue("$tool", toolName ?? (object)DBNull.Value);
        cmd.ExecuteNonQuery();
    }

    // ── Leitura ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Recupera todo o histórico de um thread em ordem cronológica.
    /// Equivalente ao LangGraph recarregando o checkpoint do thread_id.
    /// </summary>
    public List<ChatMessage> GetHistory(string threadId)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            SELECT id, thread_id, role, content, tool_name, created_at
            FROM messages
            WHERE thread_id = $tid
            ORDER BY id ASC
            """;
        cmd.Parameters.AddWithValue("$tid", threadId);

        var messages = new List<ChatMessage>();
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            messages.Add(new ChatMessage
            {
                Id = reader.GetInt64(0),
                ThreadId = reader.GetString(1),
                Role = Enum.Parse<MessageRole>(reader.GetString(2)),
                Content = reader.GetString(3),
                ToolName = reader.IsDBNull(4) ? null : reader.GetString(4),
                CreatedAt = DateTime.Parse(reader.GetString(5))
            });
        }
        return messages;
    }

    /// <summary>Lista todos os thread_ids existentes no banco.</summary>
    public List<string> ListThreads()
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            SELECT DISTINCT thread_id FROM messages ORDER BY thread_id
            """;
        var threads = new List<string>();
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
            threads.Add(reader.GetString(0));
        return threads;
    }

    /// <summary>Apaga todo o histórico de um thread.</summary>
    public void ClearThread(string threadId)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "DELETE FROM messages WHERE thread_id = $tid";
        cmd.Parameters.AddWithValue("$tid", threadId);
        cmd.ExecuteNonQuery();
    }

    public void Dispose() => _conn.Dispose();
}
