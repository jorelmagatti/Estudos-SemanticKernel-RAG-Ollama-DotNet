namespace RhAgentRagSemanticKernel;

/// <summary>
/// Representa um chunk de texto com metadados completos.
/// </summary>
public record TextChunk(
    string Content,
    string Documento,
    int Page,
    string Categoria = "geral"
)
{
    public TextChunk WithCategoria(string categoria) => this with { Categoria = categoria };
}

/// <summary>
/// Divide documentos em chunks com overlap.
/// Equivalente ao RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150).
/// </summary>
public class TextSplitter(int chunkSize = 800, int chunkOverlap = 150)
{
    private static readonly string[] Separadores = ["\n\n", "\n", ". ", " ", ""];

    public List<TextChunk> SplitDocuments(IEnumerable<PageDocument> documentos)
    {
        var chunks = new List<TextChunk>();

        foreach (var doc in documentos)
        {
            foreach (var texto in SplitText(doc.Content))
            {
                if (string.IsNullOrWhiteSpace(texto)) continue;

                chunks.Add(new TextChunk(
                    Content: texto.Trim(),
                    Documento: doc.Documento,
                    Page: doc.Page
                ));
            }
        }

        return chunks;
    }

    private List<string> SplitText(string texto)
    {
        if (texto.Length <= chunkSize)
            return [texto];

        foreach (var sep in Separadores)
        {
            var partes = string.IsNullOrEmpty(sep)
                ? texto.Select(c => c.ToString()).ToArray()
                : texto.Split(sep, StringSplitOptions.None);

            if (partes.Length <= 1 && sep != "") continue;

            var resultado = AgregarPartes(partes, sep);
            if (resultado.Count > 0) return resultado;
        }

        return CortarPorCaractere(texto);
    }

    private List<string> AgregarPartes(string[] partes, string sep)
    {
        var chunks = new List<string>();
        var buffer = new List<string>();
        int tamanho = 0;

        foreach (var parte in partes)
        {
            int extra = (buffer.Count > 0 ? sep.Length : 0) + parte.Length;

            if (tamanho + extra > chunkSize && buffer.Count > 0)
            {
                var chunk = string.Join(sep, buffer);
                if (!string.IsNullOrWhiteSpace(chunk)) chunks.Add(chunk);

                while (buffer.Count > 0 && tamanho > chunkOverlap)
                {
                    tamanho -= buffer[0].Length + sep.Length;
                    buffer.RemoveAt(0);
                }
            }

            buffer.Add(parte);
            tamanho += extra;
        }

        if (buffer.Count > 0)
        {
            var ultimo = string.Join(sep, buffer);
            if (!string.IsNullOrWhiteSpace(ultimo)) chunks.Add(ultimo);
        }

        return chunks;
    }

    private List<string> CortarPorCaractere(string texto)
    {
        var chunks = new List<string>();
        int start = 0;

        while (start < texto.Length)
        {
            int end = Math.Min(start + chunkSize, texto.Length);
            chunks.Add(texto[start..end]);
            int next = end - chunkOverlap;
            start = next > start ? next : end;
        }

        return chunks;
    }
}
