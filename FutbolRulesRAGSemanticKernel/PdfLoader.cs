using UglyToad.PdfPig;

namespace FutbolRulesRAGSemanticKernel;

public static class PdfLoader
{
    public record PageDocument(string Content, int PageNumber, string Source);

    public static List<PageDocument> Load(string pdfPath)
    {
        var pages = new List<PageDocument>();

        using var pdf = PdfDocument.Open(pdfPath);

        foreach (var page in pdf.GetPages())
        {
            var text = string.Join(" ", page.GetWords().Select(w => w.Text));
            if (!string.IsNullOrWhiteSpace(text))
                pages.Add(new PageDocument(text, page.Number, pdfPath));
        }

        return pages;
    }
}