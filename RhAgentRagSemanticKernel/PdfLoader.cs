namespace RhAgentRagSemanticKernel;

using UglyToad.PdfPig;

public record PageDocument(
    string Content,
    string Documento,   // equivalente a doc.metadata["documento"]
    int Page
);

/// <summary>
/// Carrega PDFs e retorna lista de PageDocument com metadados.
/// Equivalente ao PyPDFLoader do LangChain.
/// </summary>
public static class PdfLoader
{
    /// <summary>
    /// Equivalente ao Python:
    ///   for caminho in caminhos:
    ///       loader = PyPDFLoader(caminho)
    ///       docs = loader.load()
    ///       for doc in docs: doc.metadata["documento"] = caminho
    /// </summary>
    public static List<PageDocument> Carregar(IEnumerable<string> caminhos)
    {
        var documentos = new List<PageDocument>();

        foreach (var caminho in caminhos)
        {
            if (!File.Exists(caminho))
            {
                Console.WriteLine($"  ⚠️  Arquivo não encontrado: {caminho} (ignorado)");
                continue;
            }

            Console.WriteLine($"  📄 Carregando: {caminho}");

            using var pdf = PdfDocument.Open(caminho);

            foreach (var pagina in pdf.GetPages())
            {
                var texto = pagina.Text?.Trim() ?? string.Empty;
                if (string.IsNullOrWhiteSpace(texto)) continue;

                documentos.Add(new PageDocument(
                    Content: texto,
                    Documento: caminho,
                    Page: pagina.Number - 1  // 0-based
                ));
            }
        }

        return documentos;
    }
}