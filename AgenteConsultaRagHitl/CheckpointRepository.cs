using Microsoft.Data.Sqlite;

namespace AgenteConsultaRagHitl;


/// <summary>
/// Repositório SQLite para histórico e checkpoints do agente.
///
/// Equivalente ao SqliteSaver do LangGraph:
///   conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
///   memory = SqliteSaver(conn)
///
/// Persiste tanto o histórico de mensagens quanto o estado de interrupção
/// HITL, permitindo que a sessão seja retomada corretamente mesmo se o
/// processo for reiniciado.
/// </summary>
public class CheckpointRepository : IDisposable
{
    private readonly SqliteConnection _conn;

    public CheckpointRepository(string dbPath = "hitl_checkpoints.db")
    {
        _conn = new SqliteConnection($"Data Source={dbPath}");
        _conn.Open();
        InitSchema();
    }

    // ── Schema ────────────────────────────────────────────────────────────────

    private void InitSchema()
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id   TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                tool_name   TEXT,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );
 
            -- Checkpoints: guarda o estado de interrupção HITL por thread
            -- Equivalente ao checkpoint do LangGraph que permite retomar após interrupt
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id         TEXT PRIMARY KEY,
                interrupted_node  TEXT,
                pending_tool_name TEXT,
                pending_tool_query TEXT,
                pending_tool_id   TEXT,
                updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
            );
 
            CREATE INDEX IF NOT EXISTS idx_messages_thread
                ON messages(thread_id, id);
            """;
        cmd.ExecuteNonQuery();
    }

    // ── Mensagens ─────────────────────────────────────────────────────────────

    public void SaveMessage(string threadId, MessageRole role,
                            string content, string? toolName = null)
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

    public List<ChatMessage> GetMessages(string threadId)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            SELECT id, thread_id, role, content, tool_name, created_at
            FROM messages WHERE thread_id = $tid ORDER BY id
            """;
        cmd.Parameters.AddWithValue("$tid", threadId);

        var list = new List<ChatMessage>();
        using var r = cmd.ExecuteReader();
        while (r.Read())
            list.Add(new ChatMessage
            {
                Id = r.GetInt64(0),
                ThreadId = r.GetString(1),
                Role = Enum.Parse<MessageRole>(r.GetString(2)),
                Content = r.GetString(3),
                ToolName = r.IsDBNull(4) ? null : r.GetString(4),
                CreatedAt = DateTime.Parse(r.GetString(5))
            });
        return list;
    }

    // ── Checkpoints HITL ──────────────────────────────────────────────────────

    /// <summary>
    /// Salva o checkpoint de interrupção — equivalente ao LangGraph salvando
    /// o estado do grafo no momento do interrupt_before.
    /// </summary>
    public void SaveInterruptCheckpoint(
        string threadId,
        string interruptedNode,
        string toolName,
        string toolQuery,
        string toolId)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            INSERT INTO checkpoints
                (thread_id, interrupted_node, pending_tool_name, pending_tool_query, pending_tool_id)
            VALUES ($tid, $node, $tname, $tquery, $tid2)
            ON CONFLICT(thread_id) DO UPDATE SET
                interrupted_node   = excluded.interrupted_node,
                pending_tool_name  = excluded.pending_tool_name,
                pending_tool_query = excluded.pending_tool_query,
                pending_tool_id    = excluded.pending_tool_id,
                updated_at         = datetime('now')
            """;
        cmd.Parameters.AddWithValue("$tid", threadId);
        cmd.Parameters.AddWithValue("$node", interruptedNode);
        cmd.Parameters.AddWithValue("$tname", toolName);
        cmd.Parameters.AddWithValue("$tquery", toolQuery);
        cmd.Parameters.AddWithValue("$tid2", toolId);
        cmd.ExecuteNonQuery();
    }

    /// <summary>
    /// Recupera checkpoint de interrupção — equivalente ao graph.get_state().
    /// Retorna null se não há interrupção pendente.
    /// </summary>
    public (string Node, string ToolName, string ToolQuery, string ToolId)?
        GetInterruptCheckpoint(string threadId)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            SELECT interrupted_node, pending_tool_name, pending_tool_query, pending_tool_id
            FROM checkpoints WHERE thread_id = $tid
            """;
        cmd.Parameters.AddWithValue("$tid", threadId);
        using var r = cmd.ExecuteReader();
        if (!r.Read()) return null;
        return (r.GetString(0), r.GetString(1), r.GetString(2), r.GetString(3));
    }

    /// <summary>Limpa o checkpoint após a retomada ou cancelamento.</summary>
    public void ClearCheckpoint(string threadId)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "DELETE FROM checkpoints WHERE thread_id = $tid";
        cmd.Parameters.AddWithValue("$tid", threadId);
        cmd.ExecuteNonQuery();
    }

    public void ClearThread(string threadId)
    {
        ClearCheckpoint(threadId);
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "DELETE FROM messages WHERE thread_id = $tid";
        cmd.Parameters.AddWithValue("$tid", threadId);
        cmd.ExecuteNonQuery();
    }

    public List<string> ListThreads()
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT DISTINCT thread_id FROM messages ORDER BY thread_id";
        var list = new List<string>();
        using var r = cmd.ExecuteReader();
        while (r.Read()) list.Add(r.GetString(0));
        return list;
    }

    public void Dispose() => _conn.Dispose();
}