using Microsoft.SemanticKernel.ChatCompletion;

namespace RhAgentRagSemanticKernel;

public class Reranker(IChatCompletionService chatService)
{
    // Template do prompt de avaliação — equivalente ao PromptTemplate do Python
    private const string PROMPT_RERANK = """
        You are an expert in corporate HR policies.
        
        User question:
        {pergunta}
        
        Document excerpt:
        {texto}
        
        Evaluate the relevance of this excerpt for answering the question.
        Respond with ONLY a number from 0 to 10. No explanation, no text. Just the number.
        """;

    /// <summary>
    /// Pontua e reordena documentos por relevância à pergunta.
    /// </summary>
    /// <param name="pergunta">Pergunta do usuário</param>
    /// <param name="documentos">Chunks recuperados pela busca vetorial</param>
    /// <param name="topK">Quantos melhores retornar após reranking</param>
    public async Task<List<(double Score, DocumentRecord Doc)>> RerankAsync(
        string pergunta,
        List<DocumentRecord> documentos,
        int topK = 4)
    {
        Console.WriteLine($"\n  🔄 Reranking {documentos.Count} chunks...");

        var comScore = new List<(double Score, DocumentRecord Doc)>();

        for (int i = 0; i < documentos.Count; i++)
        {
            var doc = documentos[i];
            var prompt = PROMPT_RERANK
                .Replace("{pergunta}", pergunta)
                .Replace("{texto}", doc.Content[..Math.Min(600, doc.Content.Length)]);

            var history = new ChatHistory();
            history.AddUserMessage(prompt);

            double score = 0;

            try
            {
                var resposta = await chatService.GetChatMessageContentAsync(history);
                var texto = resposta.Content?.Trim() ?? "0";

                // Extrai o número da resposta (o LLM às vezes adiciona texto extra)
                var numero = new string(texto.Where(c => char.IsDigit(c) || c == '.').ToArray());
                score = double.TryParse(numero,
                    System.Globalization.NumberStyles.Any,
                    System.Globalization.CultureInfo.InvariantCulture,
                    out var parsed) ? parsed : 0;
            }
            catch
            {
                score = 0;
            }

            comScore.Add((score, doc));
            Console.Write($"\r  ✅ Avaliados: {i + 1}/{documentos.Count} (último score: {score:F1})");
        }

        Console.WriteLine();

        // Ordena do mais relevante para o menos relevante e retorna top-K
        return comScore
            .OrderByDescending(x => x.Score)
            .Take(topK)
            .ToList();
    }
}

