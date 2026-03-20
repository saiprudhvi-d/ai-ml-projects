/**
 * Project 03: AI Code Suggestion — VS Code Extension
 * Connects to the FastAPI backend to provide inline completions.
 *
 * Install: Copy this folder to ~/.vscode/extensions/ai-code-suggest/
 * Run backend: uvicorn api:app --port 8002
 */

const vscode = require("vscode");
const https = require("http");

const API_URL = "http://localhost:8002";
const DEBOUNCE_MS = 300;

let debounceTimer = null;
let statusBar;

/**
 * Fetch code suggestions from the FastAPI backend.
 */
async function fetchSuggestions(prefix, maxTokens = 80) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({
      prefix,
      max_tokens: maxTokens,
      temperature: 0.2,
      num_suggestions: 3,
    });

    const options = {
      hostname: "localhost",
      port: 8002,
      path: "/suggest",
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(body),
      },
    };

    const req = https.request(options, (res) => {
      let data = "";
      res.on("data", (chunk) => (data += chunk));
      res.on("end", () => {
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          reject(e);
        }
      });
    });

    req.on("error", reject);
    req.setTimeout(500, () => {
      req.destroy();
      reject(new Error("Request timeout"));
    });
    req.write(body);
    req.end();
  });
}

/**
 * Inline completion provider — triggers on every keystroke (debounced).
 */
const completionProvider = {
  async provideInlineCompletionItems(document, position, context, token) {
    if (document.languageId !== "python") return;

    const lineText = document.lineAt(position).text.substring(0, position.character);
    if (lineText.trim().length < 3) return;

    // Get last ~20 lines of context
    const startLine = Math.max(0, position.line - 20);
    const prefix = document.getText(
      new vscode.Range(startLine, 0, position.line, position.character)
    );

    try {
      statusBar.text = "$(loading~spin) AI suggesting...";
      const data = await fetchSuggestions(prefix);
      statusBar.text = `$(sparkle) AI (${data.latency_ms}ms${data.cached ? " cached" : ""})`;

      return data.suggestions.slice(0, 3).map((text) => ({
        insertText: text,
        range: new vscode.Range(position, position),
      }));
    } catch (err) {
      statusBar.text = "$(warning) AI offline";
      return [];
    }
  },
};

/**
 * Command: Explain selected code.
 */
async function explainSelectedCode() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) return;

  const selection = editor.selection;
  const selectedText = editor.document.getText(selection);
  if (!selectedText) {
    vscode.window.showWarningMessage("Select code first.");
    return;
  }

  vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: "Explaining code..." },
    async () => {
      try {
        const body = JSON.stringify({ prefix: selectedText, max_tokens: 200, num_suggestions: 1 });
        const res = await fetch(`${API_URL}/explain`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body,
        });
        const data = await res.json();
        const explanation = data.suggestions?.[0] ?? "No explanation generated.";
        vscode.window.showInformationMessage(`💡 ${explanation}`, { modal: true });
      } catch {
        vscode.window.showErrorMessage("Could not connect to AI backend.");
      }
    }
  );
}

function activate(context) {
  statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBar.text = "$(sparkle) AI Code";
  statusBar.show();
  context.subscriptions.push(statusBar);

  // Register inline completion provider
  const provider = vscode.languages.registerInlineCompletionItemProvider(
    { language: "python" },
    completionProvider
  );
  context.subscriptions.push(provider);

  // Register explain command
  const explainCmd = vscode.commands.registerCommand(
    "aiCodeSuggest.explain",
    explainSelectedCode
  );
  context.subscriptions.push(explainCmd);

  vscode.window.showInformationMessage("✅ AI Code Suggest activated. Backend: http://localhost:8002");
}

function deactivate() {
  statusBar?.dispose();
}

module.exports = { activate, deactivate };
