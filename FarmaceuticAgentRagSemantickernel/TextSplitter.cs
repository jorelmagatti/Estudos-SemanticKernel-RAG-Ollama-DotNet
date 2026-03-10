namespace FarmaceuticAgentRagSemantickernel;

public record TextChunk(
    string Content,
    string Source,
    int Page,
    string Medicamento,
    string Categoria = "geral"
)
{
    public TextChunk WithCategoria(string categoria) => this with { Categoria = categoria };
}

/// <summary>
/// Divide documentos em chunks com overlap.
/// Equivalente ao RecursiveCharacterTextSplitter do LangChain.
/// 
/// Python:
///   text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
///   chunks = text_splitter.split_documents(documentos)
/// </summary>
public class TextSplitter
{
    private readonly int _chunkSize;
    private readonly int _chunkOverlap;

    // Separadores tentados em ordem (equivalente ao "recursive" do LangChain)
    private static readonly string[] Separadores = ["\n\n", "\n", ". ", " ", ""];

    public TextSplitter(int chunkSize = 600, int chunkOverlap = 150)
    {
        _chunkSize = chunkSize;
        _chunkOverlap = chunkOverlap;
    }

    /// <summary>
    /// Divide uma lista de PageDocuments em TextChunks.
    /// </summary>
    public List<TextChunk> SplitDocuments(IEnumerable<PageDocument> documentos)
    {
        var chunks = new List<TextChunk>();

        foreach (var doc in documentos)
        {
            var textos = SplitText(doc.Content);

            foreach (var texto in textos)
            {
                if (string.IsNullOrWhiteSpace(texto))
                    continue;

                chunks.Add(new TextChunk(
                    Content: texto.Trim(),
                    Source: doc.Source,
                    Page: doc.Page,
                    Medicamento: doc.Medicamento
                ));
            }
        }

        return chunks;
    }

    /// <summary>
    /// Divide um texto em chunks com overlap usando separadores hierárquicos.
    /// </summary>
    private List<string> SplitText(string texto)
    {
        var chunks = new List<string>();

        if (texto.Length <= _chunkSize)
        {
            chunks.Add(texto);
            return chunks;
        }

        // Tenta separadores em ordem até encontrar um que funcione bem
        foreach (var sep in Separadores)
        {
            var partes = string.IsNullOrEmpty(sep)
                ? texto.Select(c => c.ToString()).ToArray()
                : texto.Split(sep, StringSplitOptions.None);

            if (partes.Length <= 1 && sep != "")
                continue;

            // Agrupa partes respeitando chunkSize com overlap
            var resultado = AgregarPartes(partes, sep);
            if (resultado.Count > 0)
                return resultado;
        }

        // Fallback: corte direto por caractere
        return CortarPorCaractere(texto);
    }

    private List<string> AgregarPartes(string[] partes, string sep)
    {
        var chunks = new List<string>();
        var buffer = new List<string>();
        int tamanhoAtual = 0;

        foreach (var parte in partes)
        {
            int tamanhoComSep = (buffer.Count > 0 ? sep.Length : 0) + parte.Length;

            if (tamanhoAtual + tamanhoComSep > _chunkSize && buffer.Count > 0)
            {
                // Salva chunk atual
                var chunkTexto = string.Join(sep, buffer);
                if (!string.IsNullOrWhiteSpace(chunkTexto))
                    chunks.Add(chunkTexto);

                // Mantém overlap: remove do início até respeitar overlap
                while (buffer.Count > 0 && tamanhoAtual > _chunkOverlap)
                {
                    tamanhoAtual -= buffer[0].Length + sep.Length;
                    buffer.RemoveAt(0);
                }
            }

            buffer.Add(parte);
            tamanhoAtual += tamanhoComSep;
        }

        // Adiciona restante
        if (buffer.Count > 0)
        {
            var ultimo = string.Join(sep, buffer);
            if (!string.IsNullOrWhiteSpace(ultimo))
                chunks.Add(ultimo);
        }

        return chunks;
    }

    private List<string> CortarPorCaractere(string texto)
    {
        var chunks = new List<string>();
        int start = 0;

        while (start < texto.Length)
        {
            int end = Math.Min(start + _chunkSize, texto.Length);
            chunks.Add(texto[start..end]);

            int next = end - _chunkOverlap;
            start = next > start ? next : end;
        }

        return chunks;
    }
}