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

const appConfig = { minSimilarity: null };

// ---- laws -------------------------------------------------------------------

const lawsById = new Map();

function lawShortName(lawId) {
  const law = lawsById.get(lawId);
  return law ? law.short_name || law.name : lawId;
}

// A value that must read left to right inside Arabic text (dates, scores, codes).
function ltr(text) {
  const node = el("bdi", "", text);
  node.dir = "ltr";
  return node;
}

// One <div><dt/><dd/></div> pair for a <dl>.
function fact(label, value) {
  const row = el("div");
  const dd = el("dd");
  dd.append(value);
  row.append(el("dt", "", `${label}:`), dd);
  return row;
}

function renderCoverage(laws) {
  const list = $("coverage");
  list.replaceChildren();
  for (const law of laws) {
    const item = el("li", "coverage-item");
    const shortName = law.short_name || law.name;
    item.append(el("span", "coverage-name", shortName));
    // The rest of the official name ("رقم (34) لسنة 2014 وتعديلاته"), as the server gives it.
    const reference = law.name.startsWith(shortName) ? law.name.slice(shortName.length).trim() : "";
    if (reference) {
      item.append(el("span", "coverage-ref", reference));
    }
    if (law.version_ar) {
      item.append(el("span", "coverage-meta", law.version_ar));
    }
    const facts = el("dl", "coverage-facts");
    facts.append(fact("عدد المواد", String(law.article_count)), fact("أُخذ النص في", ltr(law.retrieved)));
    item.append(facts);
    if (law.coverage_note_ar) {
      item.append(el("span", "coverage-note", law.coverage_note_ar));
    }

    const actions = el("div", "coverage-actions");
    const browseBtn = el("button", "text-button", "تصفّح مواد القانون");
    browseBtn.type = "button";
    browseBtn.setAttribute("aria-haspopup", "dialog");
    browseBtn.addEventListener("click", () => openArticle(law.law_id, "1", "1", law.name, browseBtn));
    actions.append(browseBtn);

    const href = safeHttpsUrl(law.source_url);
    if (href) {
      const link = el("a", "", "النص الرسمي (ملف خارجي)");
      link.href = href;
      link.rel = "noopener noreferrer";
      actions.append(link);
    }
    item.append(actions);
    list.appendChild(item);
  }
  list.hidden = laws.length === 0;
  $("coverage-status").hidden = laws.length > 0;
}

const SCOPE_ALL_LABEL = "القانونان كلاهما";

function selectedLawId() {
  const checked = document.querySelector('input[name="law"]:checked');
  return checked ? checked.value : "";
}

function scopeLabel(lawId) {
  if (!lawId) return SCOPE_ALL_LABEL;
  const law = lawsById.get(lawId);
  return law ? law.name : lawId;
}

function renderScope(laws) {
  const container = $("scope-options");
  if (!container) return;
  // Keep the first radio (#scope-all) and append options from laws
  for (const law of laws) {
    const label = el("label", "scope-option");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "law";
    input.value = law.law_id;
    const shortText = (law.short_name || law.name).replace(/^قانون\s+/, "");
    label.append(input, el("span", "", shortText));
    container.appendChild(label);
  }
}

async function loadLaws() {
  try {
    const laws = await fetchJson("/laws");
    if (!Array.isArray(laws)) throw new Error("unexpected /laws body");
    for (const law of laws) {
      lawsById.set(law.law_id, law);
    }
    renderScope(laws);
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

// The whole word is the link, with its attached prefixes ("للمادة", "وبالمادة"), so a link
// never splits a word the way "لل" + "مادة 9" used to. No lookbehind: Safari before 16.4 cannot parse
// one, and the whole script would fail; renderAnswerText checks the letter before instead.
const ARTICLE_MENTION =
  /(?:[وف]?(?:بال|كال|لل|ال|[بلك])|[وف])?ماد[ةه]\s*(?:\(\s*)?([0-9\u0660-\u0669\u06F0-\u06F9]{1,4})(?![0-9\u0660-\u0669\u06F0-\u06F9])\s*(مكرر)?\s*\)?(?:\s*(مكرر))?/gu;

const ARABIC_LETTER = /[\u0621-\u064A]/u;

function toAsciiDigits(text) {
  return text
    .replace(/[\u0660-\u0669]/g, (ch) => String.fromCharCode(ch.charCodeAt(0) - 0x0660 + 0x30))
    .replace(/[\u06F0-\u06F9]/g, (ch) => String.fromCharCode(ch.charCodeAt(0) - 0x06F0 + 0x30));
}

function articleLinkTargets(citations) {
  const counts = new Map();
  for (const c of citations) {
    counts.set(c.article, (counts.get(c.article) || 0) + 1);
  }
  const targets = new Map();
  for (const c of citations) {
    if (counts.get(c.article) === 1) {
      targets.set(c.article, c);
    }
  }
  return targets;
}

function renderAnswerText(container, text, citations) {
  container.replaceChildren();
  const targets = articleLinkTargets(citations);
  const matches = [...text.matchAll(ARTICLE_MENTION)];
  if (matches.length === 0) {
    container.textContent = text;
    return;
  }
  let lastIndex = 0;
  for (const match of matches) {
    const matchIndex = match.index;
    if (matchIndex > lastIndex) {
      container.appendChild(document.createTextNode(text.slice(lastIndex, matchIndex)));
    }
    const digits = toAsciiDigits(match[1]);
    const hasBis = Boolean(match[2] || match[3]);
    const articleKey = hasBis ? `${digits}bis` : digits;
    // A match that starts inside a word ("اعتماده 5") names no article.
    const insideWord = matchIndex > 0 && ARABIC_LETTER.test(text[matchIndex - 1]);
    const citation = insideWord ? undefined : targets.get(articleKey);

    if (citation) {
      const btn = el("button", "article-link", match[0]);
      btn.type = "button";
      btn.setAttribute("aria-haspopup", "dialog");
      btn.addEventListener("click", () => {
        openArticle(citation.law_id, citation.article, citation.article_label, citation.law_name, btn);
      });
      container.appendChild(btn);
    } else {
      container.appendChild(document.createTextNode(match[0]));
    }
    lastIndex = matchIndex + match[0].length;
  }
  if (lastIndex < text.length) {
    container.appendChild(document.createTextNode(text.slice(lastIndex)));
  }
}

let lastAnswer = null;
let copyResetTimer = null;

function buildCopyText(request, data, origin) {
  const lines = [];
  lines.push(`السؤال: ${request.question}`);
  lines.push(`نطاق البحث: ${scopeLabel(request.lawId)}`);
  lines.push("");
  lines.push("الإجابة:");
  lines.push(data.answer);
  if (data.citations && data.citations.length > 0) {
    lines.push("");
    lines.push("المواد التي استندت إليها الإجابة:");
    for (const c of data.citations) {
      lines.push(`- المادة (${c.article_label}) من ${c.law_name}`);
    }
  }
  lines.push("");
  lines.push(data.disclaimer || DEFAULT_DISCLAIMER);
  lines.push(`المصدر: ${origin}/`);
  return lines.join("\n");
}

async function copyAnswer() {
  if (!lastAnswer) return;
  const btn = $("copy-answer");
  try {
    await navigator.clipboard.writeText(buildCopyText(lastAnswer.request, lastAnswer.data, location.origin));
    btn.textContent = "نُسخت";
    setStatus("done", "", "نُسخت الإجابة مع أرقام المواد.");
    clearTimeout(copyResetTimer);
    copyResetTimer = setTimeout(() => {
      btn.textContent = "نسخ الإجابة مع أرقام المواد";
    }, 2000);
  } catch {
    btn.textContent = "تعذّر النسخ";
    setStatus("done", "", "تعذّر النسخ تلقائياً. حدّد نص الإجابة وانسخه يدوياً.");
    clearTimeout(copyResetTimer);
    copyResetTimer = setTimeout(() => {
      btn.textContent = "نسخ الإجابة مع أرقام المواد";
    }, 2000);
  }
}

let copyJsonResetTimer = null;

async function copyJson() {
  if (!lastAnswer || !lastAnswer.data) return;
  const btn = $("copy-json");
  if (!btn) return;
  try {
    await navigator.clipboard.writeText(JSON.stringify(lastAnswer.data, null, 2));
    btn.textContent = "نُسخت استجابة JSON";
    clearTimeout(copyJsonResetTimer);
    copyJsonResetTimer = setTimeout(() => {
      btn.textContent = "نسخ استجابة JSON الأصلية";
    }, 2000);
  } catch {
    btn.textContent = "تعذّر النسخ";
    clearTimeout(copyJsonResetTimer);
    copyJsonResetTimer = setTimeout(() => {
      btn.textContent = "نسخ استجابة JSON الأصلية";
    }, 2000);
  }
}

function renderBadge(data) {
  const badge = $("result-badge");
  if (!badge) return;
  if (data.refused) {
    badge.hidden = true;
    return;
  }
  const dropped = data.dropped_citations || [];
  const badgeText = $("badge-text");
  if (badgeText) {
    if (dropped.length === 0) {
      badgeText.textContent = "كل مادة مذكورة هنا استُرجعت فعلاً لهذا السؤال";
    } else {
      badgeText.textContent = "استُبعدت استشهادات لم تُسترجع؛ المعروض هنا ما بقي";
    }
  }
  badge.hidden = false;
}

// A share of the bar, 0-100, for a similarity score shown on a 0-1 scale.
function scorePercent(score) {
  return `${Math.round(Math.min(1, Math.max(0, score)) * 100)}%`;
}

// The score as text, plus a bar with the gate's threshold marked on it. Widths are set
// through CSSOM (style.setProperty), which the page CSP allows; style="" would not be.
function scoreMeter(score) {
  const meter = el("div", "score");
  const track = el("span", "score-track");
  track.setAttribute("aria-hidden", "true");
  if (Number.isFinite(score)) {
    const fill = el("span", "score-fill");
    fill.style.setProperty("inline-size", scorePercent(score));
    track.append(fill);
  }
  if (appConfig.minSimilarity !== null) {
    const threshold = el("span", "score-threshold");
    threshold.style.setProperty("inset-inline-start", scorePercent(appConfig.minSimilarity));
    track.append(threshold);
  }
  const value = el("span", "score-value");
  value.append("درجة التشابه: ", ltr(Number.isFinite(score) ? score.toFixed(SCORE_DIGITS) : "—"));
  meter.append(track, value);
  return meter;
}

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

    const open = el("button", "text-button btn-read-article", `اقرأ نص المادة (${citation.article_label}) كاملاً`);
    open.type = "button";
    open.setAttribute("aria-describedby", lawName.id);
    open.setAttribute("aria-haspopup", "dialog");
    open.addEventListener("click", (event) =>
      openArticle(citation.law_id, citation.article, citation.article_label, citation.law_name, event.currentTarget),
    );
    item.append(ref, scoreMeter(Number(citation.score)), quote, open);
    list.appendChild(item);
  });
  $("citations-block").hidden = citations.length === 0;
}

function renderSearchDetails(data, elapsedMs = null) {
  const retrieved = data.retrieved || [];
  const scores = retrieved.map((hit) => Number(hit.score)).filter((s) => Number.isFinite(s));
  const topScore = scores.length > 0 ? Math.max(...scores) : null;
  const isGateRefused = data.refusal_reason === "LOW_RETRIEVAL_SCORE";
  const gatePassed = !isGateRefused && retrieved.length > 0;
  const acceptedCitations = (data.citations || []).length;
  const droppedCitations = (data.dropped_citations || []).length;

  const chipsContainer = $("telemetry-chips");
  if (chipsContainer) {
    chipsContainer.replaceChildren();

    const chipScore = el("span", "telemetry-chip");
    const chipScoreBdi = el("bdi", "", topScore !== null ? topScore.toFixed(SCORE_DIGITS) : "—");
    chipScoreBdi.dir = "ltr";
    chipScore.appendChild(chipScoreBdi);
    chipsContainer.appendChild(chipScore);

    const chipGate = el("span", `telemetry-chip ${gatePassed ? "is-pass" : (isGateRefused ? "is-refusal" : "")}`);
    chipGate.textContent = isGateRefused ? "دون العتبة" : (gatePassed ? "اجتاز العتبة" : "—");
    chipsContainer.appendChild(chipGate);

    const chipCitations = el("span", "telemetry-chip");
    if (data.refused) {
      chipCitations.textContent = "مرفوض";
      chipCitations.classList.add("is-refusal");
    } else {
      chipCitations.textContent = `${acceptedCitations} مقبولة`;
      if (acceptedCitations > 0) chipCitations.classList.add("is-pass");
    }
    chipsContainer.appendChild(chipCitations);
  }

  const metrics = $("telemetry-metrics");
  if (metrics) {
    metrics.replaceChildren();

    function addMetric(label, valueNode, isClass = "") {
      const card = el("div", "metric-card");
      card.append(el("span", "metric-label", label));
      const val = el("span", `metric-value ${isClass}`.trim());
      if (typeof valueNode === "string") val.textContent = valueNode;
      else val.appendChild(valueNode);
      card.append(val);
      metrics.append(card);
    }

    const topScoreBdi = el("bdi", "", topScore !== null ? topScore.toFixed(SCORE_DIGITS) : "—");
    topScoreBdi.dir = "ltr";
    addMetric("أعلى درجة تشابه", topScoreBdi);

    const gateText = isGateRefused ? "دون العتبة، فلم يُستدعَ النموذج" : (retrieved.length > 0 ? "اجتاز العتبة" : "—");
    addMetric("عتبة الاسترجاع", gateText, gatePassed ? "is-pass" : (isGateRefused ? "is-refusal" : ""));

    if (appConfig.minSimilarity !== null) {
      const threshBdi = el("bdi", "", appConfig.minSimilarity.toFixed(2));
      threshBdi.dir = "ltr";
      addMetric("قيمة العتبة", threshBdi);
    }

    if (data.model) {
      const modelCode = el("code", "", data.model);
      modelCode.dir = "ltr";
      addMetric("النموذج اللغوي", modelCode);
    } else {
      addMetric("النموذج اللغوي", "لم يُستدعَ");
    }

    addMetric("الاستشهادات", `مقبولة ${acceptedCitations}، مستبعدة ${droppedCitations}`);

    if (typeof elapsedMs === "number" && Number.isFinite(elapsedMs)) {
      const timeSpan = el("span");
      const timeBdi = el("bdi", "", `${(elapsedMs / 1000).toFixed(1)} ث`);
      timeBdi.dir = "ltr";
      timeSpan.append(timeBdi, " ", el("span", "telemetry-sub", "(يشمل الشبكة، وقد يشمل إيقاظ الخادم)"));
      addMetric("زمن الانتظار في متصفحك", timeSpan);
    }

    if (data.error_code) {
      const errCode = el("code", "", data.error_code);
      errCode.dir = "ltr";
      addMetric("رمز الخطأ", errCode);
    }
  }

  const rows = $("retrieved");
  rows.replaceChildren();
  for (const hit of retrieved) {
    const row = el("tr");
    const score = Number(hit.score);
    const tdScore = el("td", "num");
    if (Number.isFinite(score)) {
      const scoreBdi = el("bdi", "", score.toFixed(SCORE_DIGITS));
      scoreBdi.dir = "ltr";
      tdScore.appendChild(scoreBdi);
    }
    row.append(
      el("td", "", lawShortName(hit.law_id).replace(/^قانون\s+/, "")),
      el("td", "num", articleLabel(hit.article)),
      el("td", "num", Number(hit.chunk_index) + 1),
      tdScore,
    );
    rows.appendChild(row);
  }

  const dropped = data.dropped_citations || [];
  const droppedNote = $("dropped");
  if (dropped.length > 0) {
    droppedNote.textContent = `استشهادات اقترحها النموذج واستُبعدت لأنها لا تشير إلى مادة مسترجعة: ${dropped.length}`;
    droppedNote.hidden = false;
  } else {
    droppedNote.textContent = "";
    droppedNote.hidden = true;
  }
}

function renderAnswer(data, request, elapsedMs = null) {
  lastAnswer = { data, request };
  const refused = Boolean(data.refused);
  $("result-question").textContent = request.question;
  const scopeNode = $("result-scope");
  if (scopeNode) scopeNode.textContent = scopeLabel(request.lawId);
  renderBadge(data);
  renderAnswerText($("answer"), data.answer, refused ? [] : data.citations || []);
  const copyBtn = $("copy-answer");
  if (copyBtn) copyBtn.hidden = refused;
  $("refusal").hidden = !refused;
  $("refusal-reason").textContent = refused
    ? REFUSAL_REASON_TEXT[data.refusal_reason] || GENERIC_REFUSAL_REASON
    : "";
  $("result").classList.toggle("is-refused", refused);
  renderCitations(refused ? [] : data.citations || []);
  renderSearchDetails(data, elapsedMs);
  $("result-disclaimer").textContent = data.disclaimer || DEFAULT_DISCLAIMER;
  $("result").hidden = false;
}

// ---- the article sheet -------------------------------------------------------

const articleCache = new Map();
const articleIndexCache = new Map();
let articleTicket = 0; // a newer request (or closing the sheet) makes older responses stale
let articleTrigger = null;
let currentArticle = null;

function focusableIn(container) {
  const selector =
    "a[href]:not([hidden]), button:not([disabled]), select:not([disabled]):not([hidden]), [tabindex]:not([tabindex='-1'])";
  return [...container.querySelectorAll(selector)].filter((node) => node.getClientRects().length > 0);
}

function setArticleStatus(text, isError = false) {
  const status = $("article-status");
  status.textContent = text;
  status.classList.toggle("is-error", isError);
}

function loadArticleIndex(lawId) {
  let promise = articleIndexCache.get(lawId);
  if (!promise) {
    promise = fetchJson(`/laws/${encodeURIComponent(lawId)}/articles`).catch((err) => {
      articleIndexCache.delete(lawId);
      throw err;
    });
    articleIndexCache.set(lawId, promise);
  }
  return promise;
}

function fillArticleNav(data) {
  const prevBtn = $("article-prev");
  const nextBtn = $("article-next");
  const jump = $("article-jump");

  if (data.prev_article) {
    prevBtn.textContent = `السابقة: المادة (${data.prev_article_label})`;
    prevBtn.dataset.article = data.prev_article;
    prevBtn.dataset.label = data.prev_article_label;
    prevBtn.removeAttribute("aria-disabled");
  } else {
    prevBtn.textContent = "لا مادة قبلها";
    delete prevBtn.dataset.article;
    delete prevBtn.dataset.label;
    prevBtn.setAttribute("aria-disabled", "true");
  }

  if (data.next_article) {
    nextBtn.textContent = `التالية: المادة (${data.next_article_label})`;
    nextBtn.dataset.article = data.next_article;
    nextBtn.dataset.label = data.next_article_label;
    nextBtn.removeAttribute("aria-disabled");
  } else {
    nextBtn.textContent = "لا مادة بعدها";
    delete nextBtn.dataset.article;
    delete nextBtn.dataset.label;
    nextBtn.setAttribute("aria-disabled", "true");
  }

  if (!jump.hidden) {
    jump.value = data.article;
  }
}

function fillArticle(data) {
  currentArticle = data;
  $("article-title").textContent = `المادة (${data.article_label})`;
  $("article-law").textContent = data.law_name;
  const law = lawsById.get(data.law_id);
  $("article-version").textContent = law && law.version_ar ? law.version_ar : "";
  $("article-text").textContent = data.text;
  const notes = $("article-notes");
  notes.replaceChildren(...(data.notes || []).map((note) => el("li", "", note)));
  $("article-notes-block").hidden = notes.children.length === 0;
  const href = safeHttpsUrl(data.source_url);
  const source = $("article-source");
  source.hidden = !href;
  if (href) source.href = href;
  fillArticleNav(data);
  setArticleStatus("");
}

async function openArticle(lawId, article, label, lawName, trigger) {
  const dialog = $("article-dialog");
  const ticket = ++articleTicket;
  if (!dialog.open) {
    articleTrigger = trigger || null;
  }
  $("article-title").textContent = `المادة (${label})`;
  $("article-law").textContent = lawName;
  const law = lawsById.get(lawId);
  $("article-version").textContent = law && law.version_ar ? law.version_ar : "";
  $("article-text").textContent = "";
  $("article-notes-block").hidden = true;
  $("article-source").hidden = true;
  setArticleStatus(ARTICLE_LOADING_TEXT);
  if (!dialog.open) dialog.showModal();
  dialog.scrollTop = 0;
  $("close-article").focus();

  const jump = $("article-jump");
  loadArticleIndex(lawId)
    .then((articles) => {
      if (ticket !== articleTicket) return;
      if (jump.dataset.lawId !== lawId) {
        jump.dataset.lawId = lawId;
        jump.replaceChildren(
          ...articles.map((item) => {
            const opt = document.createElement("option");
            opt.value = item.article;
            opt.textContent = `المادة (${item.article_label})`;
            return opt;
          })
        );
      }
      jump.hidden = false;
      if (currentArticle && currentArticle.law_id === lawId) {
        jump.value = currentArticle.article;
      }
    })
    .catch(() => {
      if (ticket !== articleTicket) return;
      jump.hidden = true;
    });

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
  $("article-prev").addEventListener("click", () => {
    const btn = $("article-prev");
    if (btn.getAttribute("aria-disabled") === "true" || !btn.dataset.article || !currentArticle) return;
    openArticle(currentArticle.law_id, btn.dataset.article, btn.dataset.label, currentArticle.law_name);
  });
  $("article-next").addEventListener("click", () => {
    const btn = $("article-next");
    if (btn.getAttribute("aria-disabled") === "true" || !btn.dataset.article || !currentArticle) return;
    openArticle(currentArticle.law_id, btn.dataset.article, btn.dataset.label, currentArticle.law_name);
  });
  $("article-jump").addEventListener("change", (event) => {
    if (!currentArticle) return;
    const article = event.target.value;
    const label = event.target.selectedOptions[0]?.textContent.replace(/^المادة \(|\)$/g, "") || article;
    openArticle(currentArticle.law_id, article, label, currentArticle.law_name);
  });
  // Esc closes a modal <dialog> natively (the "cancel" event); afterwards focus
  // goes back to the button that opened it.
  dialog.addEventListener("close", () => {
    articleTicket += 1;
    if (articleTrigger && articleTrigger.isConnected) articleTrigger.focus();
    articleTrigger = null;
  });
  // A click on the backdrop lands on the <dialog> element outside its bounding rect.
  dialog.addEventListener("click", (event) => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    const isInDialog =
      rect.top <= event.clientY &&
      event.clientY <= rect.top + rect.height &&
      rect.left <= event.clientX &&
      event.clientX <= rect.left + rect.width;
    if (!isInDialog) dialog.close();
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
    if (config) {
      if (typeof config.min_similarity === "number") {
        appConfig.minSimilarity = config.min_similarity;
      }
      if (config.turnstile_site_key) {
        captcha.enabled = true;
        loadTurnstile(config.turnstile_site_key);
      }
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
  for (const button of [submit, ...document.querySelectorAll("button[data-question]")]) {
    button.setAttribute("aria-disabled", blocked ? "true" : "false");
  }
  const scopeFieldset = $("scope");
  if (scopeFieldset) scopeFieldset.disabled = inFlight;
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
  if (
    !active ||
    active === document.body ||
    $("ask-form").contains(active) ||
    (active.matches && active.matches("button[data-question]"))
  ) {
    $("answer-title").focus();
  }
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

  const request = { question, lawId: selectedLawId() };

  inFlight = true;
  refreshButtons();
  $("result").hidden = true;
  startProgress();
  try {
    const body = { question: request.question };
    if (request.lawId) body.law_id = request.lawId;
    const headers = { "Content-Type": "application/json" };
    if (captcha.enabled) headers["CF-Turnstile-Response"] = captcha.token;
    const startFetch = performance.now();
    const data = await fetchJson(
      "/ask",
      {
        method: "POST",
        headers,
        body: JSON.stringify(body),
      },
      REQUEST_TIMEOUT_MS,
    );
    const elapsedMs = performance.now() - startFetch;
    stopProgress();
    renderAnswer(data, request, elapsedMs);
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
  const scopeAll = $("scope-all");
  if (scopeAll) scopeAll.checked = true; // the examples span both laws
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

function initAskForm() {
  $("ask-form").addEventListener("submit", onSubmit);
  $("question").addEventListener("input", onQuestionInput);
  $("question").addEventListener("keydown", onQuestionKeydown);
  for (const button of document.querySelectorAll("button[data-question]")) {
    button.addEventListener("click", onExample);
  }
  const copyBtn = $("copy-answer");
  if (copyBtn) copyBtn.addEventListener("click", copyAnswer);
  const copyJsonBtn = $("copy-json");
  if (copyJsonBtn) copyJsonBtn.addEventListener("click", copyJson);
  updateCount();
  refreshButtons();
  $("submit").disabled = false;
}

function runOptional(setup) {
  try {
    setup();
  } catch {
    // optional setup failure does not block asking
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initAskForm();
  loadLaws();
  setupCaptcha();
  runOptional(setupArticleDialog);
});
