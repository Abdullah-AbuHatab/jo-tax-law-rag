// Vanilla JS for the single page. Every server value is written with
// textContent (never innerHTML), so model output cannot inject markup.
"use strict";

const MIN_QUESTION_CHARS = 3;
const MAX_QUESTION_CHARS = 1000;
// Short waits (per-minute limit, busy server) lock the buttons until they pass;
// daily limits only show the message, since the wait is hours.
const MAX_LOCK_SECONDS = 120;
// The server gives one question at most 35 s upstream; a cold start on the free
// host adds about a minute. Past this, the page stops waiting and says so.
const REQUEST_TIMEOUT_MS = 120000;
const SLOW_AFTER_SECONDS = 30;
const TICK_MS = 1000;
const SECONDS_PER_MINUTE = 60;
const SCORE_DIGITS = 3;

const ERROR_TEXT = {
  RATE_LIMITED: "أرسلت أسئلة كثيرة خلال وقت قصير.",
  DAILY_LIMIT_REACHED: "وصلت إلى الحد اليومي لعدد الأسئلة. يتجدد الحد عند منتصف الليل بتوقيت UTC.",
  DAILY_BUDGET_EXHAUSTED: "استهلكت الخدمة سعتها المجانية لهذا اليوم. تتجدد السعة عند منتصف الليل بتوقيت UTC.",
  BUSY: "الخدمة مشغولة بأسئلة أخرى الآن.",
  PAYLOAD_TOO_LARGE: "الطلب أكبر من المسموح.",
  DATABASE_UNAVAILABLE: "قاعدة البيانات غير متاحة حالياً (قد تكون في وضع السكون). حاول بعد لحظات.",
  EMBEDDING_UNAVAILABLE: "خدمة البحث غير متاحة حالياً. حاول لاحقاً.",
  LLM_TIMEOUT: "تأخر النموذج في الرد. حاول مرة أخرى.",
  LLM_RATE_LIMITED: "النموذج المجاني وصل إلى حد الاستخدام. حاول لاحقاً.",
  LLM_UNAVAILABLE: "النموذج غير متاح حالياً. حاول لاحقاً.",
  LLM_CONFIG_ERROR: "الخدمة غير مهيأة بالكامل حالياً.",
  LLM_BAD_RESPONSE: "وصل رد غير صالح من النموذج. حاول مرة أخرى.",
  VALIDATION_ERROR: "تحقق من نص السؤال.",
  UNKNOWN_LAW: "القانون المختار غير معروف.",
  CAPTCHA_REQUIRED: "أكمل خطوة التحقق أولاً، ثم أرسل السؤال.",
  CAPTCHA_FAILED: "تعذّر تأكيد خطوة التحقق. أعد المحاولة.",
};
const GENERIC_ERROR = "حدث خطأ غير متوقع. حاول مرة أخرى.";
const NETWORK_ERROR = "تعذّر الوصول إلى الخادم. تحقق من اتصالك بالإنترنت، ثم أعد المحاولة.";
const TIMEOUT_ERROR = "لم يصل رد خلال دقيقتين. أعد المحاولة بعد قليل.";
const ERROR_TITLE = "تعذّر الحصول على إجابة";
const LOADING_TITLE = "جارٍ البحث في نصوص القانونين وصياغة الإجابة";
const LOADING_TEXT = "تستغرق الإجابة عادةً بين 10 و25 ثانية.";
const SLOW_TEXT = "ما زلنا ننتظر الرد. تعمل الخدمة على استضافة مجانية، وقد يستغرق أول سؤال بعد فترة خمول دقيقة تقريباً.";
const ANSWER_READY_TEXT = "وصلت الإجابة.";
const REFUSAL_READY_TEXT = "لم تجد الأداة إجابة في نصوص القانونين.";
const REFUSAL_REASON_TEXT = {
  LOW_RETRIEVAL_SCORE: "لم يعثر البحث على مادة قريبة بما يكفي من سؤالك، فلم يُستدعَ النموذج اللغوي أصلاً.",
  MODEL_REFUSED: "وجد البحث مواد قريبة من سؤالك، لكن النموذج اللغوي قرر أنها لا تجيب عنه.",
  NO_VALID_CITATIONS: "اقترح النموذج إجابة لا تستند إلى أي مادة استُرجعت فعلاً لهذا السؤال، فاستُبعدت.",
  LLM_INVALID_OUTPUT: "وصل من النموذج رد غير صالح، فلم تُعرض إجابة.",
};
const GENERIC_REFUSAL_REASON = "لم تجد الأداة في المواد المسترجعة ما يجيب عن سؤالك.";
const DEFAULT_DISCLAIMER = "هذه الإجابة معلومات عامة مستخرجة آلياً من نصوص القوانين وليست استشارة قانونية.";
const SUBMIT_LABEL = "اسأل";
const ARTICLE_LOADING_TEXT = "جارٍ تحميل نص المادة…";
const ARTICLE_ERROR_TEXT = "تعذّر تحميل نص المادة.";
const LAWS_ERROR_TEXT = "تعذّر تحميل قائمة القوانين الآن. يمكنك السؤال مع ذلك.";
const TURNSTILE_SCRIPT = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";

const $ = (id) => document.getElementById(id);

// A new element with optional class and text; text always goes in as textContent.
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

class ApiError extends Error {
  constructor(message, code, retryAfter) {
    super(message);
    this.code = code;
    this.retryAfter = retryAfter;
  }
}

// ---- Theme Management --------------------------------------------------------

function initTheme() {
  const saved = localStorage.getItem("theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const active = saved || (prefersDark ? "dark" : "light");
  applyTheme(active);

  const toggle = $("theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme") || (prefersDark ? "dark" : "light");
      const next = current === "dark" ? "light" : "dark";
      localStorage.setItem("theme", next);
      applyTheme(next);
    });
  }
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const label = $("theme-label");
  if (label) {
    label.textContent = theme === "dark" ? "النهاري" : "الليلي";
  }
}

// Retry-After in whole seconds; 0 when absent or not a plain number.
function retryAfterSeconds(response) {
  const value = Number.parseInt(response.headers.get("Retry-After") || "", 10);
  return Number.isFinite(value) && value > 0 ? value : 0;
}

// Arabic counted nouns change form for 1, 2, 3-10 and 11 or more.
const UNIT_FORMS = {
  hour: ["ساعة واحدة", "ساعتين", "ساعات", "ساعة"],
  minute: ["دقيقة واحدة", "دقيقتين", "دقائق", "دقيقة"],
  second: ["ثانية واحدة", "ثانيتين", "ثوانٍ", "ثانية"],
};

function countedText(count, [one, two, few, many]) {
  if (count === 1) return one;
  if (count === 2) return two;
  return `${count} ${count <= 10 ? few : many}`;
}

function waitText(seconds) {
  if (seconds >= 3600) return `${countedText(Math.ceil(seconds / 3600), UNIT_FORMS.hour)} تقريباً`;
  if (seconds >= 60) return `${countedText(Math.ceil(seconds / 60), UNIT_FORMS.minute)} تقريباً`;
  return countedText(seconds, UNIT_FORMS.second);
}

function errorMessage(code, retryAfter) {
  const text = ERROR_TEXT[code] || GENERIC_ERROR;
  return retryAfter ? `${text} يمكنك المحاولة بعد: ${waitText(retryAfter)}.` : text;
}

async function fetchJson(url, options = {}, timeoutMs = 0) {
  const controller = timeoutMs ? new AbortController() : null;
  const timer = controller ? setTimeout(() => controller.abort(), timeoutMs) : 0;
  let response;
  try {
    response = await fetch(url, controller ? { ...options, signal: controller.signal } : options);
  } catch (error) {
    const timedOut = Boolean(error && error.name === "AbortError");
    throw new ApiError(timedOut ? TIMEOUT_ERROR : NETWORK_ERROR, timedOut ? "TIMEOUT" : "NETWORK", 0);
  } finally {
    clearTimeout(timer);
  }
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    const code = body && body.error ? body.error.code : "";
    const retryAfter = retryAfterSeconds(response);
    throw new ApiError(errorMessage(code, retryAfter), code, retryAfter);
  }
  return body;
}

// Links built from server data must be plain https URLs.
function safeHttpsUrl(value) {
  try {
    const url = new URL(String(value));
    return url.protocol === "https:" ? url.href : "";
  } catch {
    return "";
  }
}

function clockText(totalSeconds) {
  const minutes = Math.floor(totalSeconds / SECONDS_PER_MINUTE);
  const seconds = String(totalSeconds % SECONDS_PER_MINUTE).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

// "4bis" is printed in the laws as "4 مكرر".
function articleLabel(key) {
  const match = /^(\d+)bis$/.exec(String(key));
  return match ? `${match[1]} مكرر` : String(key);
}

// ---- laws -------------------------------------------------------------------

const lawsById = new Map();

function lawShortName(lawId) {
  const law = lawsById.get(lawId);
  return law ? law.short_name || law.name : lawId;
}

function renderCoverage(laws) {
  const list = $("coverage");
  list.replaceChildren();
  for (const law of laws) {
    const item = el("li");
    item.append(
      el("span", "coverage-name", law.name),
      el("span", "coverage-meta", `عدد المواد: ${law.article_count}`),
      el("span", "coverage-meta", `أُخذ النص من المصدر في ${law.retrieved}`),
    );
    const href = safeHttpsUrl(law.source_url);
    if (href) {
      const link = el("a", "", "النص الرسمي (ملف خارجي)");
      link.href = href;
      link.rel = "noopener noreferrer";
      item.append(link);
    }
    list.appendChild(item);
  }
  list.hidden = laws.length === 0;
  $("coverage-status").hidden = laws.length > 0;
}

async function loadLaws() {
  try {
    const laws = await fetchJson("/laws");
    if (!Array.isArray(laws)) throw new Error("unexpected /laws body");
    const select = $("law");
    for (const law of laws) {
      lawsById.set(law.law_id, law);
      const option = el("option", "", law.short_name || law.name);
      option.value = law.law_id;
      select.appendChild(option);
    }
    renderCoverage(laws);
  } catch {
    // The filter is optional; asking still works over all laws.
    $("coverage-status").textContent = LAWS_ERROR_TEXT;
  }
}

function initLawPills() {
  const pills = document.querySelectorAll(".law-pill");
  const select = $("law");
  pills.forEach((pill) => {
    pill.addEventListener("click", () => {
      const val = pill.dataset.value || "";
      if (select) {
        select.value = val;
      }
      pills.forEach((p) => p.setAttribute("aria-selected", p === pill ? "true" : "false"));
    });
  });
  if (select) {
    select.addEventListener("change", () => {
      pills.forEach((p) => p.setAttribute("aria-selected", p.dataset.value === select.value ? "true" : "false"));
    });
  }
}

function initClearButton() {
  const clearBtn = $("clear-question");
  const textarea = $("question");
  if (!clearBtn || !textarea) return;

  clearBtn.addEventListener("click", () => {
    textarea.value = "";
    updateCount();
    showFieldError("");
    clearBtn.hidden = true;
    textarea.focus();
  });
}

// ---- status and progress ---------------------------------------------------

let progressTimer = 0;
let progressStartedAt = 0;
let slowNoticeShown = false;

// One live region for loading, errors and the "answer is ready" notice.
function setStatus(state, title = "", text = "") {
  $("status-area").dataset.state = state;
  const status = $("status");
  status.replaceChildren();
  if (title) status.appendChild(el("strong", "status-title", title));
  if (text) status.appendChild(el("span", "status-text", text));
}

function setError(message) {
  setStatus("error", ERROR_TITLE, message);
}

function elapsedSeconds() {
  return Math.floor((Date.now() - progressStartedAt) / TICK_MS);
}

function updatePipelineSteps(elapsed) {
  const steps = document.querySelectorAll(".pipeline-step");
  if (!steps.length) return;

  // 1: 0-3s, 2: 3-6s, 3: 6-22s, 4: >22s (or on arrival)
  let activeIndex = 1;
  if (elapsed >= 22) activeIndex = 4;
  else if (elapsed >= 6) activeIndex = 3;
  else if (elapsed >= 3) activeIndex = 2;

  steps.forEach((step) => {
    const num = Number(step.dataset.step);
    if (num < activeIndex) step.dataset.state = "done";
    else if (num === activeIndex) step.dataset.state = "active";
    else step.dataset.state = "pending";
  });
}

function resetPipelineSteps(allDone = false) {
  document.querySelectorAll(".pipeline-step").forEach((step) => {
    step.dataset.state = allDone ? "done" : "pending";
  });
}

function tickProgress() {
  const elapsed = elapsedSeconds();
  $("progress-elapsed").textContent = clockText(elapsed);
  updatePipelineSteps(elapsed);
  if (!slowNoticeShown && elapsed >= SLOW_AFTER_SECONDS) {
    slowNoticeShown = true;
    setStatus("loading", LOADING_TITLE, SLOW_TEXT);
  }
}

function startProgress() {
  progressStartedAt = Date.now();
  slowNoticeShown = false;
  setStatus("loading", LOADING_TITLE, LOADING_TEXT);
  $("progress-elapsed").textContent = clockText(0);
  resetPipelineSteps(false);
  updatePipelineSteps(0);
  $("progress").hidden = false;
  clearInterval(progressTimer);
  progressTimer = setInterval(tickProgress, TICK_MS);
}

function stopProgress() {
  clearInterval(progressTimer);
  progressTimer = 0;
  resetPipelineSteps(true);
  $("progress").hidden = true;
}

// ---- the answer --------------------------------------------------------------

function renderCitations(citations) {
  const list = $("citations");
  list.replaceChildren();
  citations.forEach((citation, index) => {
    const item = el("li", "citation");
    const ref = el("p", "citation-ref");
    const lawName = el("span", "citation-law", citation.law_name);
    lawName.id = `citation-law-${index}`;
    ref.append(el("span", "citation-article", `المادة (${citation.article_label})`), lawName);

    const quote = el("blockquote", "law-quote", citation.snippet);

    const open = el("button", "text-button", `اقرأ نص المادة (${citation.article_label}) كاملاً`);
    open.type = "button";
    open.setAttribute("aria-describedby", lawName.id);
    open.setAttribute("aria-haspopup", "dialog");
    open.addEventListener("click", (event) =>
      openArticle(citation.law_id, citation.article, citation.article_label, citation.law_name, event.currentTarget),
    );
    item.append(ref, quote, open);
    list.appendChild(item);
  });
  $("citations-block").hidden = citations.length === 0;
}

function renderSearchDetails(data) {
  const model = $("model");
  model.replaceChildren();
  if (data.model) {
    const name = el("code", "", data.model);
    name.dir = "ltr";
    model.append("النموذج اللغوي: ", name);
  } else {
    model.textContent = "لم يُستدعَ النموذج اللغوي، فقد رُفض السؤال قبل ذلك.";
  }

  const rows = $("retrieved");
  rows.replaceChildren();
  for (const hit of data.retrieved || []) {
    const row = el("tr");
    const score = Number(hit.score);
    const scoreText = Number.isFinite(score) ? score.toFixed(SCORE_DIGITS) : "";

    const scoreCell = el("td", "num");
    scoreCell.append(scoreText);
    if (Number.isFinite(score)) {
      const meter = document.createElement("meter");
      meter.className = "score-meter";
      meter.min = 0;
      meter.max = 1;
      meter.low = 0.6;
      meter.high = 0.75;
      meter.optimum = 0.9;
      meter.value = score;
      meter.setAttribute("aria-hidden", "true");
      scoreCell.append(meter);
    }

    row.append(
      el("td", "", lawShortName(hit.law_id).replace(/^قانون\s+/, "")),
      el("td", "num", articleLabel(hit.article)),
      el("td", "num", Number(hit.chunk_index) + 1),
      scoreCell,
    );
    rows.appendChild(row);
  }

  const dropped = data.dropped_citations || [];
  const droppedNote = $("dropped");
  droppedNote.textContent = dropped.length
    ? `استشهادات اقترحها النموذج واستُبعدت لأنها لا تشير إلى مادة مسترجعة: ${dropped.length}`
    : "";
  droppedNote.hidden = dropped.length === 0;
}

function renderFormattedAnswer(answerText, citations) {
  const container = $("answer");
  container.replaceChildren();

  // Match mentions of articles like: المادة (11) or المادة 11 or المادة (4 مكرر)
  const regex = /(الماد[ةه]\s*(?:\(\s*\d+\s*(?:مكرر)?\s*\)|\d+)(?:\s*مكرر)?)/g;
  const parts = answerText.split(regex);

  // Map article numbers to citations
  const citationMap = new Map();
  citations.forEach((c) => {
    citationMap.set(c.article, c);
    citationMap.set(c.article_label, c);
    citationMap.set(articleLabel(c.article), c);
  });

  parts.forEach((part) => {
    if (!part) return;
    if (regex.test(part)) {
      // Find matching citation
      const match = /\d+/.exec(part);
      const isBis = part.includes("مكرر");
      const key = match ? (isBis ? `${match[0]}bis` : match[0]) : null;
      const citation = key ? (citationMap.get(key) || citations[0]) : null;

      if (citation) {
        const btn = el("button", "inline-citation", part);
        btn.type = "button";
        btn.setAttribute("aria-haspopup", "dialog");
        btn.addEventListener("click", (e) => {
          openArticle(citation.law_id, citation.article, citation.article_label, citation.law_name, e.currentTarget, citation.snippet);
        });
        container.appendChild(btn);
        return;
      }
    }
    container.appendChild(document.createTextNode(part));
  });
}

function renderAnswer(data, question, elapsed = 0) {
  const refused = Boolean(data.refused);
  $("result-question").textContent = question;

  if (refused) {
    $("answer").textContent = data.answer;
  } else {
    renderFormattedAnswer(data.answer, data.citations || []);
  }

  $("refusal").hidden = !refused;
  $("refusal-reason").textContent = refused
    ? REFUSAL_REASON_TEXT[data.refusal_reason] || GENERIC_REFUSAL_REASON
    : "";
  $("result").classList.toggle("is-refused", refused);

  // Update Toolbar Badges
  const badgeStatus = $("badge-status");
  if (badgeStatus) {
    if (refused) {
      badgeStatus.className = "badge-refusal";
      badgeStatus.textContent = "رفض صريح (خارج نصوص القانونين)";
    } else {
      badgeStatus.className = "badge-verified";
      const citCount = (data.citations || []).length;
      badgeStatus.textContent = citCount > 0 ? `موثّق بـ ${citCount} مواد قانونية` : "إجابة مستندة لنص القانون";
    }
  }
  const badgeTime = $("badge-time");
  if (badgeTime && elapsed > 0) {
    badgeTime.textContent = `${elapsed}s`;
  }

  renderCitations(refused ? [] : data.citations || []);
  renderSearchDetails(data);
  $("result-disclaimer").textContent = data.disclaimer || DEFAULT_DISCLAIMER;
  $("result").hidden = false;

  setupAnswerActions(data, question);
}

function setupAnswerActions(data, question) {
  const copyBtn = $("copy-answer");
  const copyText = $("copy-text");
  if (copyBtn) {
    copyBtn.onclick = async () => {
      let fullText = `السؤال: ${question}\n\nالإجابة:\n${data.answer}\n`;
      if (data.citations && data.citations.length) {
        fullText += "\nالمواد المستند إليها:\n";
        data.citations.forEach((c) => {
          fullText += `• المادة (${c.article_label}) - ${c.law_name}\n`;
        });
      }
      fullText += `\n${data.disclaimer || DEFAULT_DISCLAIMER}\nالمصدر: قوانين الضرائب الأردنية (https://jo-tax-law-rag.onrender.com/)`;
      try {
        await navigator.clipboard.writeText(fullText);
        if (copyText) copyText.textContent = "تم النسخ!";
        setTimeout(() => {
          if (copyText) copyText.textContent = "نسخ الإجابة";
        }, 2000);
      } catch {
        // fallback
      }
    };
  }

  const shareBtn = $("share-link");
  const shareText = $("share-text");
  if (shareBtn) {
    shareBtn.onclick = async () => {
      const url = new URL(window.location.href);
      url.searchParams.set("q", question);
      const lawVal = $("law") ? $("law").value : "";
      if (lawVal) url.searchParams.set("law", lawVal);
      else url.searchParams.delete("law");

      try {
        await navigator.clipboard.writeText(url.href);
        if (shareText) shareText.textContent = "تم نسخ الرابط!";
        setTimeout(() => {
          if (shareText) shareText.textContent = "مشاركة الرابط";
        }, 2000);
      } catch {
        // fallback
      }
    };
  }
}

// ---- the article sheet -------------------------------------------------------

const articleCache = new Map();
let articleTicket = 0; // a newer request (or closing the sheet) makes older responses stale
let articleTrigger = null;

function focusableIn(container) {
  const selector = "a[href]:not([hidden]), button:not([disabled]), [tabindex]:not([tabindex='-1'])";
  return [...container.querySelectorAll(selector)].filter((node) => node.getClientRects().length > 0);
}

function setArticleStatus(text, isError = false) {
  const status = $("article-status");
  status.textContent = text;
  status.classList.toggle("is-error", isError);
}

let currentArticleMeta = null;

function fillArticle(data, highlightSnippet = "") {
  currentArticleMeta = {
    lawId: data.law_id,
    number: data.article,
    articleNum: Number.parseInt(data.article, 10) || 1,
    lawName: data.law_name,
  };

  $("article-title").textContent = `المادة (${data.article_label})`;
  $("article-law").textContent = data.law_name;

  const articleTextNode = $("article-text");
  articleTextNode.replaceChildren();

  // Split text into paragraphs/clauses and format
  const lines = data.text.split("\n");
  lines.forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed) return;

    const p = el("p", "law-clause");
    // Check if snippet text matches this paragraph
    if (highlightSnippet && highlightSnippet.length > 15 && trimmed.includes(highlightSnippet.slice(0, 30))) {
      const mark = el("mark", "cited-span", trimmed);
      p.appendChild(mark);
    } else {
      p.textContent = trimmed;
    }
    articleTextNode.appendChild(p);
  });

  const notes = $("article-notes");
  notes.replaceChildren(...(data.notes || []).map((note) => el("li", "", note)));
  $("article-notes-block").hidden = notes.children.length === 0;

  const href = safeHttpsUrl(data.source_url);
  const source = $("article-source");
  source.hidden = !href;
  if (href) source.href = href;
  setArticleStatus("");

  // Update navigation buttons
  updateArticleNav();
}

function updateArticleNav() {
  if (!currentArticleMeta) return;
  const num = currentArticleMeta.articleNum;
  const prevBtn = $("article-prev");
  const nextBtn = $("article-next");
  if (prevBtn) prevBtn.disabled = num <= 1;
  if (nextBtn) {
    const maxArticles = currentArticleMeta.lawId.includes("income") ? 82 : 78;
    nextBtn.disabled = num >= maxArticles;
  }
}

async function openArticle(lawId, article, label, lawName, trigger, highlightSnippet = "") {
  const dialog = $("article-dialog");
  const ticket = ++articleTicket;
  articleTrigger = trigger || null;
  $("article-title").textContent = `المادة (${label})`;
  $("article-law").textContent = lawName;
  $("article-text").textContent = "";
  $("article-notes-block").hidden = true;
  $("article-source").hidden = true;
  setArticleStatus(ARTICLE_LOADING_TEXT);
  if (!dialog.open) dialog.showModal();
  dialog.scrollTop = 0;
  $("close-article").focus();

  const key = `${lawId}/${article}`;
  try {
    let data = articleCache.get(key);
    if (!data) {
      data = await fetchJson(`/laws/${encodeURIComponent(lawId)}/articles/${encodeURIComponent(article)}`);
      articleCache.set(key, data);
    }
    if (ticket === articleTicket) fillArticle(data, highlightSnippet);
  } catch (error) {
    if (ticket !== articleTicket) return;
    setArticleStatus(error instanceof ApiError ? error.message : ARTICLE_ERROR_TEXT, true);
  }
}

function setupArticleDialog() {
  const dialog = $("article-dialog");
  $("close-article").addEventListener("click", () => dialog.close());

  // Esc closes a modal <dialog> natively
  dialog.addEventListener("close", () => {
    articleTicket += 1;
    if (articleTrigger && articleTrigger.isConnected) articleTrigger.focus();
    articleTrigger = null;
  });

  // Light-dismiss with boundary check per modern-web-guidance
  dialog.addEventListener("click", (event) => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    const isDialogContent = (
      rect.top <= event.clientY &&
      event.clientY <= rect.top + rect.height &&
      rect.left <= event.clientX &&
      event.clientX <= rect.left + rect.width
    );
    if (!isDialogContent) dialog.close();
  });

  // Tab navigation trap
  dialog.addEventListener("keydown", (event) => {
    if (event.key !== "Tab") return;
    const items = focusableIn(dialog);
    if (items.length === 0) return;
    const first = items[0];
    const last = items[items.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  });

  // Prev / Next button listeners
  const prevBtn = $("article-prev");
  if (prevBtn) {
    prevBtn.addEventListener("click", () => {
      if (!currentArticleMeta) return;
      const targetNum = currentArticleMeta.articleNum - 1;
      if (targetNum >= 1) {
        openArticle(currentArticleMeta.lawId, String(targetNum), String(targetNum), currentArticleMeta.lawName, prevBtn);
      }
    });
  }

  const nextBtn = $("article-next");
  if (nextBtn) {
    nextBtn.addEventListener("click", () => {
      if (!currentArticleMeta) return;
      const targetNum = currentArticleMeta.articleNum + 1;
      openArticle(currentArticleMeta.lawId, String(targetNum), String(targetNum), currentArticleMeta.lawName, nextBtn);
    });
  }

  // Jump to article form
  const jumpForm = $("article-jump-form");
  if (jumpForm) {
    jumpForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const val = $("jump-number") ? $("jump-number").value.trim() : "";
      if (val && currentArticleMeta) {
        openArticle(currentArticleMeta.lawId, val, val, currentArticleMeta.lawName, $("jump-number"));
        $("jump-number").value = "";
      }
    });
  }
}

// ---- Turnstile ------------------------------------------------------------------

// Optional Cloudflare Turnstile: on only when the server returns a site key from
// /config (the page CSP then allows challenges.cloudflare.com). A token is single
// use, so the widget is reset after every question.
const captcha = { enabled: false, widgetId: null, token: "" };

function loadTurnstile(siteKey) {
  const box = $("captcha");
  box.hidden = false;
  const script = document.createElement("script");
  script.src = TURNSTILE_SCRIPT;
  script.addEventListener("load", () => {
    captcha.widgetId = window.turnstile.render(box, {
      sitekey: siteKey,
      callback: (token) => {
        captcha.token = token;
      },
      "expired-callback": () => {
        captcha.token = "";
      },
      "error-callback": () => {
        captcha.token = "";
      },
    });
  });
  document.head.appendChild(script);
}

async function setupCaptcha() {
  try {
    const config = await fetchJson("/config");
    if (config && config.turnstile_site_key) {
      captcha.enabled = true;
      loadTurnstile(config.turnstile_site_key);
    }
  } catch {
    // No config: asking works when Turnstile is off; when it is on, the server answers CAPTCHA_REQUIRED.
  }
}

function resetCaptcha() {
  captcha.token = "";
  if (captcha.enabled && window.turnstile && captcha.widgetId !== null) window.turnstile.reset(captcha.widgetId);
}

// ---- Session Recent Questions History ------------------------------------

const RECENT_KEY = "jo_tax_recent_asks";
const MAX_RECENT = 5;

function getRecentAsks() {
  try {
    const raw = sessionStorage.getItem(RECENT_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveRecentAsk(question, lawVal, data, elapsed) {
  try {
    let list = getRecentAsks();
    list = list.filter((item) => item.question !== question);
    list.unshift({ question, lawVal, data, elapsed });
    if (list.length > MAX_RECENT) list = list.slice(0, MAX_RECENT);
    sessionStorage.setItem(RECENT_KEY, JSON.stringify(list));
    renderRecentAsks();
  } catch {
    // quota
  }
}

function renderRecentAsks() {
  const container = $("recent-container");
  const chipsNode = $("recent-chips");
  if (!container || !chipsNode) return;

  const list = getRecentAsks();
  if (!list.length) {
    container.hidden = true;
    return;
  }

  chipsNode.replaceChildren();
  list.forEach((item) => {
    const chip = el("button", "recent-chip", item.question);
    chip.type = "button";
    chip.addEventListener("click", () => {
      $("question").value = item.question;
      updateCount();
      if ($("law")) $("law").value = item.lawVal || "";
      const select = $("law");
      document.querySelectorAll(".law-pill").forEach((p) => {
        p.setAttribute("aria-selected", p.dataset.value === (item.lawVal || "") ? "true" : "false");
      });
      renderAnswer(item.data, item.question, item.elapsed);
      focusAnswer();
    });
    chipsNode.appendChild(chip);
  });
  container.hidden = false;
}

// ---- asking ----------------------------------------------------------------------

function validateQuestion(question) {
  if (question.length < MIN_QUESTION_CHARS) return "اكتب سؤالاً من ثلاثة أحرف على الأقل.";
  if (question.length > MAX_QUESTION_CHARS) return "السؤال طويل جداً (الحد 1000 حرف).";
  return "";
}

function showFieldError(problem) {
  const fieldError = $("question-error");
  fieldError.textContent = problem;
  fieldError.hidden = !problem;
  if (problem) $("question").setAttribute("aria-invalid", "true");
  else $("question").removeAttribute("aria-invalid");
}

// One request at a time: the ask button and the examples share this state.
let inFlight = false;
let lockedUntil = 0;
let lockTimer = 0;

function lockSecondsLeft() {
  return Math.max(0, Math.ceil((lockedUntil - Date.now()) / TICK_MS));
}

function isBlocked() {
  return inFlight || lockSecondsLeft() > 0;
}

// aria-disabled rather than disabled, so a focused button keeps focus during a request.
function refreshButtons() {
  const blocked = isBlocked();
  const secondsLeft = lockSecondsLeft();
  const submit = $("submit");
  for (const button of [submit, ...document.querySelectorAll("button.example")]) {
    button.setAttribute("aria-disabled", blocked ? "true" : "false");
  }
  submit.textContent = !inFlight && secondsLeft > 0 ? `اسأل بعد ${clockText(secondsLeft)}` : SUBMIT_LABEL;
  $("ask-form").setAttribute("aria-busy", inFlight ? "true" : "false");
}

function tickLock() {
  if (lockSecondsLeft() === 0) {
    clearInterval(lockTimer);
    lockTimer = 0;
  }
  refreshButtons();
}

function lockFor(seconds) {
  if (!seconds || seconds > MAX_LOCK_SECONDS) return;
  lockedUntil = Date.now() + seconds * TICK_MS;
  clearInterval(lockTimer);
  lockTimer = setInterval(tickLock, TICK_MS);
  refreshButtons();
}

// Move focus to the answer only if the reader is still where they asked from.
function focusAnswer() {
  const active = document.activeElement;
  if (!active || active === document.body || $("ask-form").contains(active)) $("answer-title").focus();
}

async function onSubmit(event) {
  event.preventDefault();
  if (isBlocked()) return;
  const question = $("question").value.trim();
  const problem = validateQuestion(question);
  showFieldError(problem);
  if (problem) {
    setStatus("idle"); // the inline field error says what to fix; drop any older message
    $("question").focus();
    return;
  }

  if (captcha.enabled && !captcha.token) {
    setError(ERROR_TEXT.CAPTCHA_REQUIRED);
    return;
  }

  inFlight = true;
  refreshButtons();
  $("result").hidden = true;
  startProgress();
  const startedAt = Date.now();
  try {
    const body = { question };
    if ($("law").value) body.law_id = $("law").value;
    const headers = { "Content-Type": "application/json" };
    if (captcha.enabled) headers["CF-Turnstile-Response"] = captcha.token;
    const data = await fetchJson(
      "/ask",
      {
        method: "POST",
        headers,
        body: JSON.stringify(body),
      },
      REQUEST_TIMEOUT_MS,
    );
    const elapsed = Math.round((Date.now() - startedAt) / 1000);
    stopProgress();
    renderAnswer(data, question, elapsed);
    setStatus("done", "", data.refused ? REFUSAL_READY_TEXT : ANSWER_READY_TEXT);
    saveRecentAsk(question, $("law").value, data, elapsed);
    focusAnswer();
  } catch (error) {
    stopProgress();
    setError(error instanceof ApiError ? error.message : GENERIC_ERROR);
    if (error instanceof ApiError) lockFor(error.retryAfter);
  } finally {
    inFlight = false;
    resetCaptcha();
    refreshButtons();
  }
}

function onExample(event) {
  if (isBlocked()) return;
  const question = event.currentTarget.dataset.question || "";
  $("question").value = question;
  updateCount();
  $("law").value = ""; // the examples span both laws
  $("ask-form").requestSubmit();
}

function updateCount() {
  const len = $("question").value.length;
  $("question-count").textContent = `${len} / ${MAX_QUESTION_CHARS}`;
  const clearBtn = $("clear-question");
  if (clearBtn) clearBtn.hidden = len === 0;
}

function onQuestionInput() {
  updateCount();
  if (!$("question-error").hidden && !validateQuestion($("question").value.trim())) showFieldError("");
}

function onQuestionKeydown(event) {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    $("ask-form").requestSubmit();
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initLawPills();
  initClearButton();
  $("ask-form").addEventListener("submit", onSubmit);
  $("question").addEventListener("input", onQuestionInput);
  $("question").addEventListener("keydown", onQuestionKeydown);
  for (const button of document.querySelectorAll("button.example, button.guardrail-chip")) {
    button.addEventListener("click", onExample);
  }
  setupArticleDialog();
  updateCount();
  refreshButtons();
  loadLaws();
  setupCaptcha();
  renderRecentAsks();

  // URL query pre-fill: ?q=...&law=...
  const params = new URLSearchParams(window.location.search);
  const qParam = params.get("q");
  if (qParam) {
    $("question").value = qParam;
    updateCount();
    const lawParam = params.get("law");
    if (lawParam && $("law")) {
      $("law").value = lawParam;
      document.querySelectorAll(".law-pill").forEach((p) => {
        p.setAttribute("aria-selected", p.dataset.value === lawParam ? "true" : "false");
      });
    }
  }
});
