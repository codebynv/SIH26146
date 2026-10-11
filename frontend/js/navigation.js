import { store } from "./state.js";

/**
 * Deterministic investigator navigation (Step 9B Phase 1).
 *
 * Plain in-page anchor links to the existing dashboard section containers. No
 * router, no URL rewriting, no external dependency. The only state kept here is
 * presentation state (which link is marked current); dashboard data is never read
 * or written by this module.
 */

export const NAV_TARGETS = Object.freeze([
  Object.freeze({ id: "overview-content", label: "Overview" }),
  Object.freeze({ id: "why-lead-content", label: "Why This Lead" }),
  Object.freeze({ id: "evidence-chain-content", label: "Evidence Chain" }),
  Object.freeze({ id: "cross-layer-content", label: "Cross-Layer Correlation" }),
  Object.freeze({ id: "paths-content", label: "Investigation Paths" }),
  Object.freeze({ id: "behavior-ml-content", label: "Behavioral / ML" }),
  Object.freeze({ id: "provenance-content", label: "Provenance / Verification" })
]);

const NAV_LINK_SELECTOR = ".investigator-nav__link";
const ACTIVE_CLASS = "is-active";

let observer = null;
let activeSectionId = NAV_TARGETS[0].id;
let listenersAttached = false;
let trackedRatios = new Map();
let unsubscribe = null;

function navLinks() {
  return Array.from(document.querySelectorAll(NAV_LINK_SELECTOR));
}

function isKnownSection(id) {
  return NAV_TARGETS.some(t => t.id === id);
}

export function getActiveSectionId() {
  return activeSectionId;
}

/**
 * Marks exactly one navigation link as current. Presentation only: touches
 * aria-current and a class on the anchor, never dashboard data.
 */
export function setActiveSection(id) {
  if (!isKnownSection(id)) return activeSectionId;
  activeSectionId = id;
  navLinks().forEach(link => {
    const isActive = link.dataset.section === id;
    link.classList.toggle(ACTIVE_CLASS, isActive);
    if (isActive) link.setAttribute("aria-current", "true");
    else link.removeAttribute("aria-current");
  });
  return activeSectionId;
}

/**
 * Moves the viewport to an existing dashboard section using normal anchor
 * semantics. Falls back to a direct scroll when the hash is already current.
 */
export function navigateToSection(id) {
  if (!isKnownSection(id)) return false;
  const target = document.getElementById(id);
  setActiveSection(id);
  if (target && typeof target.scrollIntoView === "function") {
    target.scrollIntoView({ block: "start" });
  }
  if (window.location.hash !== "#" + id) {
    window.location.hash = "#" + id;
  } else if (target && typeof target.scrollIntoView === "function") {
    target.scrollIntoView({ block: "start" });
  }
  return true;
}

function pickMostVisible() {
  let bestId = null;
  let bestRatio = 0;
  // Iterate the frozen target order so equal ratios resolve deterministically.
  NAV_TARGETS.forEach(target => {
    const ratio = trackedRatios.get(target.id) || 0;
    if (ratio > bestRatio) {
      bestRatio = ratio;
      bestId = target.id;
    }
  });
  return bestRatio > 0 ? bestId : null;
}

function setupObserver() {
  if (typeof IntersectionObserver === "undefined") return;
  observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry && entry.target && entry.target.id) {
        trackedRatios.set(entry.target.id, entry.isIntersecting ? entry.intersectionRatio : 0);
      }
    });
    const visible = pickMostVisible();
    if (visible) setActiveSection(visible);
  }, { root: null, rootMargin: "0px 0px -60% 0px", threshold: [0, 0.25, 0.5, 0.75, 1] });

  NAV_TARGETS.forEach(target => {
    const element = document.getElementById(target.id);
    if (element) observer.observe(element);
  });
}

function teardownObserver() {
  if (observer) {
    observer.disconnect();
    observer = null;
  }
  trackedRatios = new Map();
}

function onNavClick(event) {
  const link = event.currentTarget;
  const id = link.dataset.section;
  if (!isKnownSection(id)) return;
  setActiveSection(id);
}

function onHashChange() {
  const id = window.location.hash.replace(/^#/, "");
  if (isKnownSection(id)) setActiveSection(id);
}

export function initNavigation() {
  navLinks().forEach(link => link.addEventListener("click", onNavClick));
  window.addEventListener("hashchange", onHashChange);
  listenersAttached = true;

  const hashId = window.location.hash.replace(/^#/, "");
  setActiveSection(isKnownSection(hashId) ? hashId : activeSectionId);

  setupObserver();

  // Re-assert the active link whenever the dashboard re-renders its sections so
  // a full reload cannot leave the nav out of sync with the current hash.
  unsubscribe = store.subscribe(() => {
    const current = window.location.hash.replace(/^#/, "");
    setActiveSection(isKnownSection(current) ? current : activeSectionId);
  });
}

export function cleanupNavigation() {
  if (listenersAttached) {
    navLinks().forEach(link => link.removeEventListener("click", onNavClick));
    window.removeEventListener("hashchange", onHashChange);
    listenersAttached = false;
  }
  teardownObserver();
  if (unsubscribe) {
    unsubscribe();
    unsubscribe = null;
  }
}