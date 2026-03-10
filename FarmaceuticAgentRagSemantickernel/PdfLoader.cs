using UglyToad.PdfPig;

namespace FarmaceuticAgentRagSemantickernel;

public record PageDocument(
    string Content,
    string Source,
    int Page,
    string Medicamento
);

public static class PdfLoader
{
    /// <summary>
    /// Carrega múltiplos PDFs e adiciona metadado "medicamento" baseado no nome do arquivo.
    /// Equivalente ao loop com PyPDFLoader + doc.metadata["medicamento"] do Python.
    /// </summary>
    public static List<PageDocument> CarregarBulas(IEnumerable<string> caminhos)
    {
        var documentos = new List<PageDocument>();

        foreach (var caminho in caminhos)
        {
            // Extrai nome do medicamento do nome do arquivo (sem extensão)
            // Equivalente a: caminho.split("/")[-1].replace(".pdf", "")
            var medicamento = Path.GetFileNameWithoutExtension(caminho);

            Console.WriteLine($"  📄 Carregando: {caminho}");

            using var pdf = PdfDocument.Open(caminho);

            foreach (var paginaPdf in pdf.GetPages())
            {
                var texto = paginaPdf.Text?.Trim() ?? string.Empty;

                if (string.IsNullOrWhiteSpace(texto))
                    continue;

                documentos.Add(new PageDocument(
                    Content: texto,
                    Source: caminho,
                    Page: paginaPdf.Number - 1, // 0-based para paridade com Python
                    Medicamento: medicamento
                ));
            }
        }

        return documentos;
    }
}
