namespace AgenteConsultaRagBuscaAgentica;

public class OllamaConfig
{
    // URL base do servidor Ollama local
    public string BaseUrl { get; set; } = "http://localhost:11434";

    // Modelo de geração de texto (ex: llama3, mistral, phi3, gemma2)
    public string ChatModel { get; set; } = "llama3.2";

    // Modelo de embeddings (ex: nomic-embed-text, mxbai-embed-large)
    public string EmbeddingModel { get; set; } = "nomic-embed-text";
}
