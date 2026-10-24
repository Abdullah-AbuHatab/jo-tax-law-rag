// Vanilla JS for the single page. Every server value is written with
// textContent (never innerHTML), so model output cannot inject markup.
"use strict";

const MIN_QUESTION_CHARS = 3;
const MAX_QUESTION_CHARS = 1000;

const ERROR_TEXT = {
  RATE_LIMITED: "أرسلت أسئلة كثيرة خلال وقت قصير. انتظر دقيقة ثم حاول مرة أخرى.",
  DATABASE_UNAVAILABLE: "قاعدة البيانات غير متاحة حالياً (قد تكون في وضع السكون). حاول بعد لحظات.",
  EMBEDDING_UNAVAILABLE: "خدمة البحث غير متاحة حالياً. حاول لاحقاً.",
  LLM_TIMEOUT: "تأخر النموذج في الرد. حاول مرة أخرى.",
  LLM_RATE_LIMITED: "النموذج المجاني وصل إلى حد الاستخدام. حاول لاحقاً.",
  LLM_UNAVAILABLE: "النموذج غير متاح حالياً. حاول لاحقاً.",
  LLM_CONFIG_ERROR: "الخدمة غير مهيأة بالكامل حالياً.",
  LLM_BAD_RESPONSE: "وصل رد غير صالح من النموذج. حاول مرة أخرى.",
  VALIDATION_ERROR: "تحقق من نص السؤال.",
  UNKNOWN_LAW: "القانون المختار غير معروف.",
};
const GENERIC_ERROR = "حدث خطأ غير متوقع. حاول مرة أخرى.";

const $ = (id) => document.getElementById(id);

function setStatus(text, isError = false) {
  const status = $("status");
  status.textContent = text;
  status.className = isError ? "status error" : "status";
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    const code = body && body.error ? body.error.code : "";
    throw new Error(ERROR_TEXT[code] || GENERIC_ERROR);
  }
  return body;
}

async function loadLaws() {
  try {
    const laws = await fetchJson("/laws");
    const select = $("law");
    for (const law of laws) {
      const option = document.createElement("option");
      option.value = law.law_id;
      option.textContent = law.name;
      select.appendChild(option);
    }
  } catch {
    // The filter is optional; asking still works over all laws.
  }
}

function renderCitations(citations) {
  const list = $("citations");
  list.replaceChildren();
  for (const citation of citations) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "secondary";
    button.textContent = `${citation.law_name} — المادة (${citation.article_label})`;
    button.addEventListener("click", () => showArticle(citation.law_id, citation.article));
    const snippet = document.createElement("span");
    snippet.className = "snippet";
    snippet.textContent = citation.snippet;
    item.append(button, snippet);
    list.appendChild(item);
  }
  $("citations-block").hidden = citations.length === 0;
}

function renderRetrieved(retrieved, model) {
  const list = $("retrieved");
  list.replaceChildren();
  for (const hit of retrieved) {
    const item = document.createElement("li");
    item.textContent = `${hit.law_id} — المادة ${hit.article} (جزء ${hit.chunk_index + 1}): ${hit.score.toFixed(3)}`;
    list.appendChild(item);
  }
  $("model").textContent = model ? `النموذج: ${model}` : "لم يُستدعَ النموذج اللغوي.";
}

function renderAnswer(data) {
  $("answer").textContent = data.answer;
  $("refused-badge").hidden = !data.refused;
  renderCitations(data.citations);
  renderRetrieved(data.retrieved, data.model);
  $("result").hidden = false;
}

async function showArticle(lawId, article) {
  const panel = $("article-panel");
  try {
    const data = await fetchJson(`/laws/${encodeURIComponent(lawId)}/articles/${encodeURIComponent(article)}`);
    $("article-title").textContent = `المادة (${data.article_label})`;
    $("article-law").textContent = data.law_name;
    $("article-text").textContent = data.text;
    const notes = $("article-notes");
    notes.replaceChildren();
    for (const note of data.notes) {
      const item = document.createElement("li");
      item.textContent = note;
      notes.appendChild(item);
    }
    $("article-source").href = data.source_url;
    panel.hidden = false;
    panel.focus();
  } catch (error) {
    setStatus(error.message, true);
  }
}

function validateQuestion(question) {
  if (question.length < MIN_QUESTION_CHARS) return "اكتب سؤالاً من ثلاثة أحرف على الأقل.";
  if (question.length > MAX_QUESTION_CHARS) return "السؤال طويل جداً (الحد 1000 حرف).";
  return "";
}

async function onSubmit(event) {
  event.preventDefault();
  const question = $("question").value.trim();
  const fieldError = $("question-error");
  const problem = validateQuestion(question);
  fieldError.textContent = problem;
  fieldError.hidden = !problem;
  if (problem) {
    $("question").focus();
    return;
  }

  const submit = $("submit");
  submit.disabled = true;
  setStatus("جارٍ البحث في نصوص القوانين…");
  try {
    const body = { question };
    if ($("law").value) body.law_id = $("law").value;
    const data = await fetchJson("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    renderAnswer(data);
    setStatus("");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    submit.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("ask-form").addEventListener("submit", onSubmit);
  $("close-article").addEventListener("click", () => {
    $("article-panel").hidden = true;
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") $("article-panel").hidden = true;
  });
  loadLaws();
});
