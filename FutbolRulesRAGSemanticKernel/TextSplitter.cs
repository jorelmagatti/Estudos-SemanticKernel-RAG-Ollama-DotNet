namespace FutbolRulesRAGSemanticKernel;

public static class TextSplitter
{
    public record Chunk(string Content, int PageNumber, string Source);

    public static List<Chunk> Split(
        IEnumerable<PdfLoader.PageDocument> pages,
        int chunkSize = 500,
        int chunkOverlap = 100)
    {
        var chunks = new List<Chunk>();

        foreach (var page in pages)
        {
            var text = page.Content;
            int start = 0;

            while (start < text.Length)
            {
                int end = Math.Min(start + chunkSize, text.Length);

                // Tenta quebrar no último espaço para não cortar palavras
                if (end < text.Length)
                {
                    int lastSpace = text.LastIndexOf(' ', end, end - start);
                    if (lastSpace > start)
                        end = lastSpace;
                }

                // Garante range válido
                if (end <= start)
                    end = Math.Min(start + chunkSize, text.Length);

                var content = text[start..end].Trim();

                if (!string.IsNullOrWhiteSpace(content))
                    chunks.Add(new Chunk(content, page.PageNumber, page.Source));

                // Avança com overlap — nunca retrocede
                int next = end - chunkOverlap;
                start = next > start ? next : end;
            }
        }

        return chunks;
    }
}
