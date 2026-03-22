namespace AgentConsultaRagPersistencia;

public class OllamaConfig
{
    public string BaseUrl { get; set; } = "http://localhost:11434";
    public string ChatModel { get; set; } = "llama3.2";
    public string EmbedModel { get; set; } = "nomic-embed-text";
}
