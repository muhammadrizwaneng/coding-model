const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const promptInput = document.getElementById("prompt-input");
const sendBtn = document.getElementById("send-btn");
const clearChatBtn = document.getElementById("clear-chat");
const modelNameEl = document.getElementById("model-name");
const modelProviderEl = document.getElementById("model-provider");
const modelBaseEl = document.getElementById("model-base");
const statusTextEl = document.getElementById("status-text");

let conversationHistory = [];
let activeModel = "";

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatMessage(text) {
  const parts = text.split(/```([\s\S]*?)```/g);
  let html = "";

  for (let index = 0; index < parts.length; index += 1) {
    if (index % 2 === 1) {
      html += `<pre><code>${escapeHtml(parts[index].trim())}</code></pre>`;
    } else if (parts[index].trim()) {
      html += `<p>${escapeHtml(parts[index]).replaceAll("\n", "<br>")}</p>`;
    }
  }

  return html || "<p></p>";
}

function appendMessage(role, content, meta = "") {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "You" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = role === "assistant" ? formatMessage(content) : `<p>${escapeHtml(content).replaceAll("\n", "<br>")}</p>`;

  if (meta) {
    const metaEl = document.createElement("div");
    metaEl.className = "meta";
    metaEl.textContent = meta;
    bubble.appendChild(metaEl);
  }

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  return wrapper;
}

function setLoading(isLoading) {
  sendBtn.disabled = isLoading;
  clearChatBtn.disabled = isLoading;
  promptInput.disabled = isLoading;
  statusTextEl.textContent = isLoading ? "Generating response..." : "Ready";
}

async function loadModelInfo() {
  try {
    const response = await fetch("/api/model");
    const data = await response.json();
    activeModel = data.model;
    modelNameEl.textContent = data.model;
    modelProviderEl.textContent = `${data.provider} · ${data.backend || "ready"}`;
    if (data.base_model) {
      modelBaseEl.textContent = `Base: ${data.base_model}`;
    } else {
      modelBaseEl.textContent = "";
    }
    if (data.adapter_ready === "false") {
      statusTextEl.textContent = "Adapter not found. Add your Colab model files.";
    }
  } catch (error) {
    modelNameEl.textContent = "Unavailable";
    statusTextEl.textContent = "Could not load model info";
  }
}

async function sendMessage(message) {
  appendMessage("user", message);
  conversationHistory.push({ role: "user", content: message });

  const typingMessage = appendMessage("assistant", "Thinking...");
  typingMessage.querySelector(".bubble").classList.add("typing");
  setLoading(true);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        model: activeModel,
        history: conversationHistory.slice(0, -1),
      }),
    });

    const data = await response.json();
    typingMessage.remove();

    if (!response.ok) {
      throw new Error(data.detail || "Request failed");
    }

    appendMessage(
      "assistant",
      data.response,
      `Model: ${data.model} · ${data.duration_seconds}s`
    );
    conversationHistory.push({ role: "assistant", content: data.response });
  } catch (error) {
    typingMessage.remove();
    appendMessage("assistant", `Error: ${error.message}`);
    statusTextEl.textContent = "Error while generating response";
  } finally {
    setLoading(false);
    promptInput.focus();
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = promptInput.value.trim();
  if (!message) return;

  promptInput.value = "";
  await sendMessage(message);
});

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

clearChatBtn.addEventListener("click", () => {
  conversationHistory = [];
  messagesEl.innerHTML = "";
  appendMessage(
    "assistant",
    "Started a new chat. Ask me to generate code, debug an error, or explain a concept."
  );
  promptInput.focus();
});

loadModelInfo();
promptInput.focus();
