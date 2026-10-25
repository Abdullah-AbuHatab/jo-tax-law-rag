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

// ---- HTTP -----------------------------------------------------------------

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

function tickProgress() {
  const elapsed = elapsedSeconds();
  $("progress-elapsed").textContent = clockText(elapsed);
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
  $("progress").hidden = false;
  clearInterval(progressTimer);
  progressTimer = setInterval(tickProgress, TICK_MS);
}

function stopProgress() {
  clearInterval(progressTimer);
  progressTimer = 0;
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
    row.append(
      el("td", "", lawShortName(hit.law_id).replace(/^قانون\s+/, "")), // the column is already "القانون"
      el("td", "num", articleLabel(hit.article)),
      el("td", "num", Number(hit.chunk_index) + 1),
      el("td", "num", Number.isFinite(score) ? score.toFixed(SCORE_DIGITS) : ""),
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

function renderAnswer(data, question) {
  const refused = Boolean(data.refused);
  $("result-question").textContent = question;
  $("answer").textContent = data.answer;
  $("refusal").hidden = !refused;
  $("refusal-reason").textContent = refused
    ? REFUSAL_REASON_TEXT[data.refusal_reason] || GENERIC_REFUSAL_REASON
    : "";
  $("result").classList.toggle("is-refused", refused);
  renderCitations(refused ? [] : data.citations || []);
  renderSearchDetails(data);
  $("result-disclaimer").textContent = data.disclaimer || DEFAULT_DISCLAIMER;
  $("result").hidden = false;
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

function fillArticle(data) {
  $("article-title").textContent = `المادة (${data.article_label})`;
  $("article-law").textContent = data.law_name;
  $("article-text").textContent = data.text;
  const notes = $("article-notes");
  notes.replaceChildren(...(data.notes || []).map((note) => el("li", "", note)));
  $("article-notes-block").hidden = notes.children.length === 0;
  const href = safeHttpsUrl(data.source_url);
  const source = $("article-source");
  source.hidden = !href;
  if (href) source.href = href;
  setArticleStatus("");
}

async function openArticle(lawId, article, label, lawName, trigger) {
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
    if (ticket === articleTicket) fillArticle(data);
  } catch (error) {
    if (ticket !== articleTicket) return;
    setArticleStatus(error instanceof ApiError ? error.message : ARTICLE_ERROR_TEXT, true);
  }
}

function setupArticleDialog() {
  const dialog = $("article-dialog");
  $("close-article").addEventListener("click", () => dialog.close());
  // Esc closes a modal <dialog> natively (the "cancel" event); afterwards focus
  // goes back to the button that opened it.
  dialog.addEventListener("close", () => {
    articleTicket += 1;
    if (articleTrigger && articleTrigger.isConnected) articleTrigger.focus();
    articleTrigger = null;
  });
  // A click on the backdrop lands on the <dialog> element itself.
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
  // showModal() already makes the page behind inert; this keeps Tab cycling
  // inside the sheet instead of leaving for the browser's own toolbar.
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
    stopProgress();
    renderAnswer(data, question);
    setStatus("done", "", data.refused ? REFUSAL_READY_TEXT : ANSWER_READY_TEXT);
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
  $("question-count").textContent = `${$("question").value.length} / ${MAX_QUESTION_CHARS}`;
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
  $("ask-form").addEventListener("submit", onSubmit);
  $("question").addEventListener("input", onQuestionInput);
  $("question").addEventListener("keydown", onQuestionKeydown);
  for (const button of document.querySelectorAll("button.example")) {
    button.addEventListener("click", onExample);
  }
  setupArticleDialog();
  updateCount();
  refreshButtons();
  loadLaws();
  setupCaptcha();
});
