// Ark frontend — no build step, no framework. Vanilla DOM + WebSocket.

const feedEl = document.getElementById("feed");
const emptyStateEl = document.getElementById("emptyState");
const filterEmptyEl = document.getElementById("filterEmptyState");
const feedFilterEl = document.getElementById("feedFilter");
const rosterListEl = document.getElementById("rosterList");
const timelineListEl = document.getElementById("timelineList");
const tickerStatusEl = document.getElementById("tickerStatus");
const tickerDateEl = document.getElementById("tickerDate");
const tickerDotEl = document.getElementById("tickerDot");
const composerForm = document.getElementById("composer");
const promptInput = document.getElementById("prompt");
const launchBtn = document.getElementById("launchBtn");
const composerHint = document.getElementById("composerHint");
const simListEl = document.getElementById("simList");
const exitBtn = document.getElementById("exitBtn");
const portalOverlay = document.getElementById("portalOverlay");
const portalTitle = document.getElementById("portalTitle");
const profileOverlay = document.getElementById("profileOverlay");
const profileContent = document.getElementById("profileContent");
const closeProfileBtn = document.getElementById("closeProfile");

let currentSocket = null;
let currentSimId = null;
let currentTimeline = [];
let currentEventCursor = -1;
let seenPostIds = new Set();
let agentsById = new Map();
let dividedEventIds = new Set();
let postMetaById = new Map();
let feedFilterMode = "all"; // "all" | "following"

// ---- PWA: register service worker (safe no-op if unsupported) ---------

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/static/sw.js").catch(() => {});
  });
}

// ---- deterministic avatar color from a handle string -----------------

function hashString(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = (h << 5) - h + str.charCodeAt(i);
    h |= 0;
  }
  return Math.abs(h);
}

function avatarStyle(seed) {
  const hue = hashString(seed) % 360;
  return `background: hsl(${hue} 65% 90%); color: hsl(${hue} 45% 32%);`;
}

// DiceBear avatars — "notionists" style matches the Notion-inspired
// direction of the redesign. Layered OVER the existing color+initials
// circle rather than replacing it: the fallback renders instantly (no
// network round-trip needed to show something), and if the DiceBear image
// fails to load (offline, CDN down, ad-blocker), onerror removes the
// broken <img> and the color+initials circle underneath is what's left —
// never a broken-image icon.
function avatarHtml(seed, displayName, extraAttrs) {
  const style = avatarStyle(seed);
  const url = `https://api.dicebear.com/9.x/notionists/svg?seed=${encodeURIComponent(seed)}&backgroundColor=f4f4f2,eef0ff,fff4e0,e9f5ec`;
  return `<div class="avatar" style="${style}" ${extraAttrs || ""}>
    <span class="avatar-fallback">${initials(displayName)}</span>
    <img class="avatar-img" src="${url}" alt="" loading="lazy" onerror="this.remove()" />
  </div>`;
}

function initials(name) {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0].toUpperCase())
    .join("");
}

// ---- small decorative, deterministic engagement counts ----------------

function fakeCount(seed, max) {
  return hashString(seed) % max;
}

function formatCount(n) {
  if (n === 0) return "";
  if (n < 1000) return String(n);
  return (n / 1000).toFixed(1) + "K";
}

// ---- follow functionality (per-simulation, persisted in localStorage) --
// This is a real deployed web app (not a Claude Artifact), so localStorage
// is the right tool here — follows should survive a page reload.

function followKey(simId) {
  return `ark_follows_${simId}`;
}

function loadFollows(simId) {
  try {
    const raw = localStorage.getItem(followKey(simId));
    return new Set(raw ? JSON.parse(raw) : []);
  } catch (e) {
    return new Set();
  }
}

function saveFollows(simId, set) {
  try {
    localStorage.setItem(followKey(simId), JSON.stringify([...set]));
  } catch (e) {
    // storage unavailable/full — following just won't persist this session
  }
}

let currentFollows = new Set();

function isFollowing(agentId) {
  return currentFollows.has(agentId);
}

function toggleFollow(agentId) {
  if (currentFollows.has(agentId)) {
    currentFollows.delete(agentId);
  } else {
    currentFollows.add(agentId);
  }
  if (currentSimId) saveFollows(currentSimId, currentFollows);
  applyFeedFilter();
  return currentFollows.has(agentId);
}

function applyFeedFilter() {
  const posts = feedEl.querySelectorAll(".post[data-agent-id]");
  posts.forEach((el) => {
    const show = feedFilterMode === "all" || currentFollows.has(el.dataset.agentId);
    el.classList.toggle("filtered-out", !show);
  });
  const anyVisible = [...posts].some((el) => !el.classList.contains("filtered-out"));
  filterEmptyEl.style.display = (feedFilterMode === "following" && posts.length && !anyVisible) ? "block" : "none";
}

// ---- rendering ----------------------------------------------------------

function clearFeed() {
  feedEl.querySelectorAll(".post, .time-divider").forEach((el) => el.remove());
  seenPostIds = new Set();
}

function setStatus(text, live) {
  tickerStatusEl.textContent = text;
  tickerDotEl.classList.toggle("live", !!live);
}

function renderRoster(roster) {
  if (!roster || roster.length === 0) return;
  rosterListEl.innerHTML = "";
  roster.forEach((a) => {
    agentsById.set(a.id, a);
    const li = document.createElement("li");
    li.className = "roster-row";
    const badge = a.narrative_role === "commentator" ? '<span class="commentator-badge">commentary</span>' : "";
    li.innerHTML = `
      ${avatarHtml(a.avatar_seed || a.handle, a.name, `data-agent-id="${a.id}"`)}
      <div class="roster-info">
        <span class="roster-name" data-agent-id="${a.id}">${escapeHtml(a.name)}${badge}</span>
        <span class="roster-role">${escapeHtml(a.role || "")}</span>
      </div>`;
    rosterListEl.appendChild(li);
  });
}

function renderTimeline(timeline) {
  currentTimeline = timeline || [];
  timelineListEl.innerHTML = "";
  if (!timeline || timeline.length === 0) return;
  timeline.forEach((ev) => {
    const li = document.createElement("li");
    li.className = "tl-item";
    li.dataset.order = ev.order;
    li.dataset.eventId = ev.id;
    li.tabIndex = 0;
    const gapChip = ev.gap_label ? `<span class="tl-gap">+${escapeHtml(ev.gap_label)}</span>` : "";
    li.innerHTML = `
      <span class="tl-date">${escapeHtml(ev.sim_date || "")}${gapChip}</span>
      <span class="tl-title">${escapeHtml(ev.title)}</span>
      <span class="tl-mode">${ev.mode}</span>`;
    li.addEventListener("click", () => jumpToEvent(ev.id));
    timelineListEl.appendChild(li);
  });
}

function jumpToEvent(eventId) {
  const target = feedEl.querySelector(`.post[data-event-id="${eventId}"]`);
  const tlItem = timelineListEl.querySelector(`.tl-item[data-event-id="${eventId}"]`);
  if (target) {
    target.scrollIntoView({ behavior: "smooth", block: "center" });
    target.classList.add("jump-highlight");
    setTimeout(() => target.classList.remove("jump-highlight"), 1200);
  } else if (tlItem) {
    tlItem.classList.add("pending-flash");
    setTimeout(() => tlItem.classList.remove("pending-flash"), 700);
  }
}

function markTimelineProgress(eventId) {
  const idx = currentTimeline.findIndex((e) => e.id === eventId);
  if (idx === -1) return;
  currentEventCursor = idx;
  const items = timelineListEl.querySelectorAll(".tl-item");
  items.forEach((li, i) => {
    li.classList.toggle("current", i === idx);
    li.classList.toggle("done", i < idx);
  });
  const ev = currentTimeline[idx];
  if (ev && ev.sim_date) tickerDateEl.textContent = ev.sim_date;
}

function getEventById(eventId) {
  return currentTimeline.find((e) => e.id === eventId);
}

const MEDIA_ICON = { Photo: "\ud83d\uddbc", Video: "\ud83c\udfa5" };

function appendPost(post) {
  if (seenPostIds.has(post.id)) return;
  seenPostIds.add(post.id);
  emptyStateEl.style.display = "none";

  markTimelineProgress(post.event_id);

  // Time-skip divider — the compressed-but-still-temporal marker.
  const event = getEventById(post.event_id);
  if (event && event.gap_label && !dividedEventIds.has(event.id)) {
    dividedEventIds.add(event.id);
    const divider = document.createElement("div");
    divider.className = "time-divider";
    divider.dataset.eventId = event.id;
    divider.innerHTML = `<span>${escapeHtml(event.gap_label)}</span>`;
    feedEl.appendChild(divider);
  }

  const div = document.createElement("div");
  div.className = "post";
  div.dataset.eventId = post.event_id;
  div.dataset.agentId = post.agent_id;
  div.dataset.postId = post.id;
  let mediaBlock = "";
  if (post.media_url) {
    mediaBlock = `<img class="media-image" src="${post.media_url}" alt="${escapeHtml(post.media_hint || 'media')}" loading="lazy" />`;
  } else if (post.media_hint) {
    mediaBlock = `<div class="media-hint">${MEDIA_ICON[post.media_hint] || "\ud83d\udcce"} ${escapeHtml(post.media_hint)} attached</div>`;
  }
  const agent = agentsById.get(post.agent_id);
  const commentatorBadge = agent && agent.narrative_role === "commentator" ? '<span class="commentator-badge">commentary</span>' : "";
  const replyCount = fakeCount(post.id + "r", 40);
  const repostCount = fakeCount(post.id + "rt", 90);
  const likeCount = fakeCount(post.id + "l", 400);

  let replyBlock = "";
  if (post.reply_to_post_id) {
    const parent = postMetaById.get(post.reply_to_post_id);
    if (parent) {
      replyBlock = `<div class="reply-context" data-jump-post-id="${post.reply_to_post_id}">↩ Replying to ${escapeHtml(parent.agent_handle)}</div>`;
    }
  }

  div.innerHTML = `
    ${avatarHtml(post.agent_handle, post.agent_name, `data-agent-id="${post.agent_id}"`)}
    <div class="post-body">
      ${replyBlock}
      <div class="post-head">
        <span class="post-name" data-agent-id="${post.agent_id}">${escapeHtml(post.agent_name)}</span>
        <span class="role-tag">${escapeHtml(post.agent_role || "")}</span>${commentatorBadge}
        <span class="post-handle">${escapeHtml(post.agent_handle)}</span>
        <span class="post-dot">·</span>
        <span class="post-date">${escapeHtml(post.sim_date || "")}</span>
      </div>
      <div class="post-content">${escapeHtml(post.content)}</div>
      ${mediaBlock}
      <div class="post-actions">
        <span>${replyIcon()} ${formatCount(replyCount)}</span>
        <span>${repostIcon()} ${formatCount(repostCount)}</span>
        <span>${likeIcon()} ${formatCount(likeCount)}</span>
        <span>${shareIcon()}</span>
      </div>
    </div>`;
  if (feedFilterMode === "following" && !currentFollows.has(post.agent_id)) {
    div.classList.add("filtered-out");
  }
  feedEl.appendChild(div);
  postMetaById.set(post.id, { agent_name: post.agent_name, agent_handle: post.agent_handle });
  div.scrollIntoView({ behavior: "smooth", block: "end" });
}

function escapeHtml(str) {
  const d = document.createElement("div");
  d.textContent = str == null ? "" : String(str);
  return d.innerHTML;
}

function replyIcon() {
  return `<svg viewBox="0 0 24 24"><path d="M21 12c0 4.4-4 8-9 8-1.1 0-2.2-.2-3.1-.5L3 21l1.6-4.8C3.6 14.9 3 13.5 3 12c0-4.4 4-8 9-8s9 3.6 9 8z"/></svg>`;
}
function repostIcon() {
  return `<svg viewBox="0 0 24 24"><path d="M17 1l4 4-4 4M3 11V9a4 4 0 0 1 4-4h14M7 23l-4-4 4-4M21 13v2a4 4 0 0 1-4 4H3"/></svg>`;
}
function likeIcon() {
  return `<svg viewBox="0 0 24 24"><path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.6l-1-1a5.5 5.5 0 0 0-7.8 7.8l1 1L12 21l7.8-7.6 1-1a5.5 5.5 0 0 0 0-7.8z"/></svg>`;
}
function shareIcon() {
  return `<svg viewBox="0 0 24 24"><path d="M4 12v7a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-7M16 6l-4-4-4 4M12 2v14"/></svg>`;
}

// ---- portal transition (immersive enter) -------------------------------

let portalHideTimer = null;

function showPortal(title) {
  clearTimeout(portalHideTimer);
  portalTitle.textContent = title ? `Entering: ${title}` : "Entering the moment…";
  portalOverlay.classList.remove("fading");
  portalOverlay.classList.add("visible");
  portalOverlay.setAttribute("aria-hidden", "false");
  portalHideTimer = setTimeout(hidePortal, 12000);
}

function hidePortal() {
  clearTimeout(portalHideTimer);
  portalOverlay.classList.add("fading");
  setTimeout(() => {
    portalOverlay.classList.remove("visible", "fading");
    portalOverlay.setAttribute("aria-hidden", "true");
  }, 400);
}

// ---- feed filter pills ---------------------------------------------------

feedFilterEl.querySelectorAll(".filter-pill").forEach((btn) => {
  btn.addEventListener("click", () => {
    feedFilterMode = btn.dataset.mode;
    feedFilterEl.querySelectorAll(".filter-pill").forEach((b) => b.classList.toggle("active", b === btn));
    applyFeedFilter();
  });
});

// ---- websocket -----------------------------------------------------------

function connect(simId, titleHint) {
  if (currentSocket) {
    currentSocket.close();
  }
  currentSimId = simId;
  agentsById = new Map();
  dividedEventIds = new Set();
  postMetaById = new Map();
  currentFollows = loadFollows(simId);
  feedFilterMode = "all";
  feedFilterEl.querySelectorAll(".filter-pill").forEach((b) => b.classList.toggle("active", b.dataset.mode === "all"));
  filterEmptyEl.style.display = "none";
  clearFeed();
  emptyStateEl.style.display = "block";
  emptyStateEl.querySelector("p").textContent = "Connecting…";
  rosterListEl.innerHTML = '<li class="panel-empty">No agents yet.</li>';
  timelineListEl.innerHTML = '<li class="panel-empty">No events yet.</li>';
  setStatus("connecting…", false);
  showPortal(titleHint);
  closeDrawers();

  const proto = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${proto}://${location.host}/ws/${simId}`);
  currentSocket = socket;

  socket.onopen = () => setStatus("planning…", true);

  socket.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "roster") {
      renderRoster(msg.roster);
      setStatus("building timeline…", true);
      if (msg.title) portalTitle.textContent = `Entering: ${msg.title}`;
    } else if (msg.type === "timeline") {
      renderTimeline(msg.timeline);
      setStatus("live", true);
      emptyStateEl.style.display = "none";
      hidePortal();
    } else if (msg.type === "post") {
      appendPost(msg.post);
    } else if (msg.type === "done") {
      setStatus("simulation complete", false);
      hidePortal();
    } else if (msg.type === "error") {
      if (msg.auth_error) {
        window.location.href = "/login";
        return;
      }
      setStatus("error: " + msg.message, false);
      hidePortal();
    }
  };

  socket.onclose = () => {
    if (currentSocket === socket) setStatus((tickerStatusEl.textContent || "disconnected"), false);
  };
}

function exitToToday() {
  if (currentSocket) {
    currentSocket.close();
    currentSocket = null;
  }
  currentSimId = null;
  hidePortal();
  clearFeed();
  agentsById = new Map();
  dividedEventIds = new Set();
  postMetaById = new Map();
  currentFollows = new Set();
  feedFilterMode = "all";
  feedFilterEl.querySelectorAll(".filter-pill").forEach((b) => b.classList.toggle("active", b.dataset.mode === "all"));
  filterEmptyEl.style.display = "none";
  rosterListEl.innerHTML = '<li class="panel-empty">No agents yet.</li>';
  timelineListEl.innerHTML = '<li class="panel-empty">No events yet.</li>';
  currentTimeline = [];
  emptyStateEl.style.display = "block";
  emptyStateEl.querySelector("p").textContent = "No timeline yet.";
  setStatus("no simulation loaded", false);
  tickerDateEl.textContent = "—";
  setActiveSimInList(null);
}

exitBtn.addEventListener("click", exitToToday);

function setActiveSimInList(simId) {
  [...simListEl.children].forEach((li) => li.classList.toggle("active", li.dataset.id === simId));
}

async function refreshSimList() {
  const res = await fetch("/api/simulations");
  const sims = await res.json();
  simListEl.innerHTML = "";
  sims.forEach((s) => {
    const li = document.createElement("li");
    li.className = "sim-item";
    li.dataset.id = s.id;
    li.tabIndex = 0;
    li.innerHTML = `<span class="sim-title">${escapeHtml(s.title)}</span><span class="sim-meta">${s.agent_count} agents · ${s.status}</span>`;
    li.addEventListener("click", () => {
      setActiveSimInList(s.id);
      connect(s.id, s.title);
    });
    simListEl.appendChild(li);
  });
}

// ---- agent profile overlay ---------------------------------------------

async function openProfile(agentId) {
  if (!currentSimId || !agentId) return;
  profileContent.innerHTML = '<p class="panel-empty">Loading profile…</p>';
  profileOverlay.classList.add("visible");
  profileOverlay.setAttribute("aria-hidden", "false");
  try {
    const res = await fetch(`/api/simulations/${currentSimId}/agents/${agentId}`);
    if (!res.ok) throw new Error("Could not load profile");
    const data = await res.json();
    renderProfile(data.agent, data.posts);
  } catch (err) {
    profileContent.innerHTML = `<p class="panel-empty">${escapeHtml(err.message || "Something went wrong.")}</p>`;
  }
}

function renderProfile(agent, posts) {
  const badge = agent.narrative_role === "commentator" ? '<span class="commentator-badge">commentary</span>' : "";
  const groundedNote = agent.narrative_role === "commentator"
    ? (agent.grounded
        ? '<dt>Source</dt><dd>Grounded in a real historical figure/outlet found via search.</dd>'
        : '<dt>Source</dt><dd>No confident real-world match found — a plausible, period-appropriate voice.</dd>')
    : "";
  const backstoryRow = agent.backstory ? `<dt>Backstory</dt><dd>${escapeHtml(agent.backstory)}</dd>` : "";
  const relationships = agent.relationships || [];
  const relationshipsRow = relationships.length
    ? `<dt>Relationships</dt><dd>${relationships.map((r) => {
        const tagChips = (r.tags || []).map((t) => `<span class="rel-tag">${escapeHtml(t)}</span>`).join(" ");
        const clickable = r.target_id ? ` data-agent-id="${r.target_id}" style="cursor:pointer;text-decoration:underline;"` : "";
        return `<div class="rel-row"><span${clickable}>${escapeHtml(r.target_name)}</span> ${tagChips}</div>`;
      }).join("")}</dd>`
    : "";
  const postsHtml = posts.length
    ? posts.map((p) => {
        const media = p.media_url
          ? `<img class="media-image" src="${p.media_url}" alt="${escapeHtml(p.media_hint || 'media')}" loading="lazy" />`
          : (p.media_caption ? `<div class="media-hint">${MEDIA_ICON[p.media_hint] || "\ud83d\udcce"} ${escapeHtml(p.media_caption)}</div>` : "");
        return `
        <div class="post" style="padding-left:0;padding-right:0;">
          ${avatarHtml(p.agent_handle, p.agent_name)}
          <div class="post-body">
            <div class="post-head">
              <span class="post-date">${escapeHtml(p.sim_date || "")}</span>
            </div>
            <div class="post-content">${escapeHtml(p.content)}</div>
            ${media}
          </div>
        </div>`;
      }).join("")
    : '<p class="panel-empty">No posts yet.</p>';

  const following = isFollowing(agent.id);
  profileContent.innerHTML = `
    <div class="profile-header">
      ${avatarHtml(agent.avatar_seed || agent.handle, agent.name)}
      <div>
        <div class="profile-name">${escapeHtml(agent.name)}${badge}</div>
        <div class="profile-handle">${escapeHtml(agent.handle)} · ${escapeHtml(agent.role || "")}</div>
      </div>
      <button type="button" class="follow-btn ${following ? 'following' : ''}" id="profileFollowBtn">
        ${following ? "Following" : "Follow"}
      </button>
    </div>
    <dl class="profile-bio">
      <dt>Personality</dt><dd>${escapeHtml(agent.personality || "")}</dd>
      <dt>Goals</dt><dd>${escapeHtml(agent.goals || "")}</dd>
      <dt>What they know</dt><dd>${escapeHtml(agent.era_context || "")}</dd>
      ${backstoryRow}
      ${relationshipsRow}
      ${groundedNote}
    </dl>
    <div class="profile-posts">
      <h4>Posts</h4>
      ${postsHtml}
    </div>`;

  const followBtn = document.getElementById("profileFollowBtn");
  followBtn.addEventListener("click", () => {
    const nowFollowing = toggleFollow(agent.id);
    followBtn.textContent = nowFollowing ? "Following" : "Follow";
    followBtn.classList.toggle("following", nowFollowing);
  });
}

function closeProfileOverlay() {
  profileOverlay.classList.remove("visible");
  profileOverlay.setAttribute("aria-hidden", "true");
}

closeProfileBtn.addEventListener("click", closeProfileOverlay);
profileOverlay.addEventListener("click", (e) => {
  if (e.target === profileOverlay) closeProfileOverlay();
});

// Event delegation: reply-context lines jump to their parent post; any
// other element carrying data-agent-id opens that agent's profile. Checked
// in this order because the outer .post div itself carries data-agent-id,
// so a reply-context line nested inside it would otherwise incorrectly
// resolve to "open this post's own author" via closest().
document.addEventListener("click", (e) => {
  const jumpTarget = e.target.closest("[data-jump-post-id]");
  if (jumpTarget) {
    const parentPost = feedEl.querySelector(`.post[data-post-id="${jumpTarget.dataset.jumpPostId}"]`);
    if (parentPost) {
      parentPost.scrollIntoView({ behavior: "smooth", block: "center" });
      parentPost.classList.add("jump-highlight");
      setTimeout(() => parentPost.classList.remove("jump-highlight"), 1200);
    }
    return;
  }
  const agentTarget = e.target.closest("[data-agent-id]");
  if (agentTarget) openProfile(agentTarget.dataset.agentId);
});

// ---- mobile page router (bottom tab bar: Feed / Cast / Library / Account)
// Desktop shows all .page sections at once (the 3-column layout); below
// 980px only one is visible at a time, toggled by the bottom nav.
//
// The composer ("New") is DELIBERATELY NOT part of this system — it uses
// the same simple, already-proven overlay pattern as the profile and
// portal overlays (a .visible class toggle on a fixed-position panel)
// instead of the page-router. This is a deliberate reliability choice:
// the page-router has a sharper edge (every page container AND every nav
// button share the routing data-page attribute, so a selector that isn't
// scoped carefully enough can accidentally hide nav buttons themselves),
// and the composer is the single most important control in the app — it
// gets the simpler, harder-to-break mechanism on purpose.

const PAGE_TITLES = { feed: "Feed", cast: "Cast", library: "Library", account: "Account" };
const mobileTopbarTitle = document.getElementById("mobileTopbarTitle");
const bottomNavEl = document.getElementById("bottomNav");
const composerSection = document.getElementById("composerSection");

function isMobileLayout() {
  return window.innerWidth <= 980;
}

function showPage(name) {
  if (!isMobileLayout()) return;
  document.querySelectorAll(".page").forEach((el) => {
    el.classList.toggle("page-active", el.dataset.page === name);
  });
  bottomNavEl.querySelectorAll(".nav-btn[data-page]").forEach((b) => {
    b.classList.toggle("active", b.dataset.page === name);
  });
  if (mobileTopbarTitle) mobileTopbarTitle.textContent = PAGE_TITLES[name] || "Ark";
  window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
}

bottomNavEl.querySelectorAll(".nav-btn[data-page]").forEach((btn) => {
  btn.addEventListener("click", () => showPage(btn.dataset.page));
});

function openComposer() {
  composerSection.classList.add("composer-open");
  document.getElementById("navNewBtn").classList.add("active");
}

function closeComposerOverlay() {
  composerSection.classList.remove("composer-open");
  document.getElementById("navNewBtn").classList.remove("active");
}

document.getElementById("navNewBtn").addEventListener("click", openComposer);
document.getElementById("closeComposer").addEventListener("click", closeComposerOverlay);

const emptyStateCta = document.getElementById("emptyStateCta");
if (emptyStateCta) emptyStateCta.addEventListener("click", openComposer);

function closeDrawers() {
  showPage("feed");
  closeComposerOverlay();
}

// First-time users have nothing to look at on the Feed page — open the
// composer directly instead of leaving them on an empty feed with no
// obvious next step. Returning users with past simulations land on Feed.
showPage("feed");
if (isMobileLayout() && simListEl.children.length === 0) openComposer();

// ---- composer submit -------------------------------------------------

composerForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const prompt = promptInput.value.trim();
  if (prompt.length < 3) {
    composerHint.textContent = "Give Ark a bit more to work with.";
    return;
  }
  launchBtn.disabled = true;
  launchBtn.textContent = "Building the world…";
  composerHint.textContent = "";
  try {
    const res = await fetch("/api/simulations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    if (res.status === 401) {
      window.location.href = "/login";
      return;
    }
    if (!res.ok) throw new Error("Could not start simulation");
    const data = await res.json();
    await refreshSimList();
    setActiveSimInList(data.id);
    connect(data.id, data.title);
  } catch (err) {
    composerHint.textContent = err.message || "Something went wrong.";
  } finally {
    launchBtn.disabled = false;
    launchBtn.textContent = "Launch simulation";
  }
});

// ---- wire up existing sim list on page load ---------------------------

[...simListEl.children].forEach((li) => {
  li.tabIndex = 0;
  li.addEventListener("click", () => {
    setActiveSimInList(li.dataset.id);
    connect(li.dataset.id, li.querySelector(".sim-title")?.textContent);
  });
});