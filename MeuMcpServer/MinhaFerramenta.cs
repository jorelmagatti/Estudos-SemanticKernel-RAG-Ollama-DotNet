using Microsoft.AspNetCore.Mvc;
using ModelContextProtocol.Server;
using System.ComponentModel;       

[McpServerToolType]
public class ExemploTools
{
    [McpServerTool]
    [ActionName("helloworld")]
    [Description("Retorna uma saudação pelo nome. Use quando o usuário disser o próprio nome ou pedir que o digite (ex.: \"meu nome é Ana\").")]
    public string HelloWorld(
        [Description("Primeiro nome extraído da mensagem do usuário (ex.: João).")] string nome = "mundo")
    {
        return $"Olá, {nome}! ✅ Servidor MCP rodando com .NET 10.";
    }

    [McpServerTool]
    [ActionName("somar")]
    [Description("Soma dois números inteiros.")]
    public int Somar(
        [Description("Primeiro número")] int a,
        [Description("Segundo número")] int b)
    {
        return a + b;
    }
}