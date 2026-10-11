/**
 * Minimal DOM implementation for Step 9B Phase 3 behavioural tests.
 *
 * The frontend is vanilla JavaScript with no external libraries, so the tests
 * exercise it against a small, self-contained DOM rather than a browser or a
 * third-party emulation package. Only the surface the dashboard actually uses
 * is implemented:
 *
 *   document.createElement / createTextNode / getElementById
 *   document.querySelector / querySelectorAll
 *   Element.append / appendChild / removeChild / replaceChildren
 *   Element.setAttribute / getAttribute / hasAttribute / removeAttribute
 *   Element.classList / dataset / id / textContent / hidden / style
 *   Element.addEventListener / removeEventListener / dispatchEvent (bubbling)
 *   Element.querySelector / querySelectorAll (descendant combinator)
 *   Event.type / key / target / currentTarget / bubbles / preventDefault
 *   KeyboardEvent and MouseEvent constructors
 *
 * Selectors supported: comma-separated lists of compound selectors built from
 * `tag`, `#id`, `.class`, `[attr]`, `[attr="value"]`, optionally combined with
 * the descendant combinator (a space). That is the whole grammar the dashboard
 * renderers use.
 */

const TAG_RE = /^[a-zA-Z][a-zA-Z0-9-]*$/;
const SELECTOR_PART_RE = /([a-zA-Z][a-zA-Z0-9-]*)|#([\w-]+)|\.([\w-]+)|\[([\w-]+)(?:=(?:"([^"]*)"|'([^']*)'|([^\]]*)))?\]/g;

const CAMEL_RE = /-([a-z0-9])/g;

function camel(name) {
  return name.replace(CAMEL_RE, (_, ch) => ch.toUpperCase());
}

function dasherize(name) {
  return name.replace(/[A-Z]/g, ch => "-" + ch.toLowerCase());
}

export class DomEvent {
  constructor(type, init = {}) {
    this.type = type;
    this.bubbles = init.bubbles !== false;
    this.cancelable = init.cancelable !== false;
    this.defaultPrevented = false;
    this.target = null;
    this.currentTarget = null;
    if (init.key !== undefined) this.key = init.key;
    if (init.detail !== undefined) this.detail = init.detail;
    Object.assign(this, init.extra || {});
  }

  preventDefault() {
    if (this.cancelable) this.defaultPrevented = true;
  }

  stopPropagation() {
    this.__stopped = true;
  }
}

export class KeyboardEvent extends DomEvent {
  constructor(type, init = {}) {
    super(type, init);
    this.key = init.key !== undefined ? init.key : "";
  }
}

export class MouseEvent extends DomEvent {
  constructor(type, init = {}) {
    super(type, init);
    this.detail = init.detail !== undefined ? init.detail : 1;
  }
}

class ClassList {
  constructor(element) {
    this._element = element;
  }

  get _tokens() {
    const raw = this._element.getAttribute("class") || "";
    return raw.split(/\s+/).filter(Boolean);
  }

  _write(tokens) {
    if (tokens.length === 0) this._element.removeAttribute("class");
    else this._element.setAttribute("class", tokens.join(" "));
  }

  contains(token) {
    return this._tokens.indexOf(token) !== -1;
  }

  add(...tokens) {
    const current = this._tokens;
    tokens.forEach(token => {
      if (token && current.indexOf(token) === -1) current.push(token);
    });
    this._write(current);
  }

  remove(...tokens) {
    const current = this._tokens.filter(token => tokens.indexOf(token) === -1);
    this._write(current);
  }

  toggle(token, force) {
    const has = this.contains(token);
    const shouldHave = force === undefined ? !has : !!force;
    if (shouldHave) this.add(token);
    else this.remove(token);
    return shouldHave;
  }

  get length() {
    return this._tokens.length;
  }

  get value() {
    return this._tokens.join(" ");
  }
}

class Dataset {
  constructor(element) {
    this._element = element;
  }

  get(key) {
    const value = this._element.getAttribute("data-" + dasherize(key));
    return value === null ? undefined : value;
  }

  set(key, value) {
    this._element.setAttribute("data-" + dasherize(key), String(value));
  }

  has(key) {
    return this._element.hasAttribute("data-" + dasherize(key));
  }

  delete(key) {
    this._element.removeAttribute("data-" + dasherize(key));
  }
}

const VOID_TAGS = new Set([
  "area", "base", "br", "col", "embed", "hr", "img", "input",
  "link", "meta", "param", "source", "track", "wbr"
]);

const ENTITIES = {
  amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: "\u00a0",
  hellip: "\u2026", mdash: "\u2014", ndash: "\u2013",
  times: "\u00d7", minus: "\u2212", arrow: "\u2192"
};

function decodeEntities(text) {
  return text.replace(/&(#x?[0-9a-fA-F]+|[a-zA-Z]+);/g, (match, body) => {
    if (body[0] === "#") {
      const code =
        body[1] === "x" || body[1] === "X"
          ? parseInt(body.slice(2), 16)
          : parseInt(body.slice(1), 10);
      if (!Number.isFinite(code)) return match;
      return String.fromCodePoint(code);
    }
    const decoded = ENTITIES[body.toLowerCase()];
    return decoded === undefined ? match : decoded;
  });
}

/**
 * Parse an HTML fragment into an array of nodes.
 *
 * Deliberately small: elements, attributes, void and self-closing tags, text and
 * entities. A closing tag with no open element, or an unterminated tag, throws so
 * a broken template cannot quietly render as an empty element.
 */
function parseFragment(document, html) {
  const root = document.createElement("div");
  const stack = [root];
  const top = () => stack[stack.length - 1];
  let index = 0;

  const addText = text => {
    if (text === "") return;
    top().appendChild(document.createTextNode(decodeEntities(text)));
  };

  while (index < html.length) {
    const lt = html.indexOf("<", index);
    if (lt === -1) {
      addText(html.slice(index));
      break;
    }
    if (lt > index) addText(html.slice(index, lt));

    if (html.startsWith("<!--", lt)) {
      const end = html.indexOf("-->", lt);
      if (end === -1) throw new Error("unterminated comment in innerHTML");
      index = end + 3;
      continue;
    }
    if (html.startsWith("<!", lt)) {
      const end = html.indexOf(">", lt);
      if (end === -1) throw new Error("unterminated declaration in innerHTML");
      index = end + 1;
      continue;
    }

    const gt = html.indexOf(">", lt);
    if (gt === -1) throw new Error("unterminated tag in innerHTML");
    const raw = html.slice(lt + 1, gt).trim();
    index = gt + 1;

    if (raw.startsWith("/")) {
      const name = raw.slice(1).trim().toLowerCase();
      const open = top();
      if (open === root) throw new Error("closing tag </" + name + "> with nothing open");
      if (open.tagName.toLowerCase() !== name) {
        throw new Error("closing tag </" + name + "> does not match <" + open.tagName + ">");
      }
      stack.pop();
      continue;
    }

    const selfClosing = raw.endsWith("/");
    const body = selfClosing ? raw.slice(0, -1).trim() : raw;
    const spaceAt = body.search(/[\s/]/);
    const tagName = (spaceAt === -1 ? body : body.slice(0, spaceAt)).toLowerCase();
    if (!TAG_RE.test(tagName)) throw new Error("invalid tag in innerHTML: " + tagName);

    const element = document.createElement(tagName);
    const attrText = spaceAt === -1 ? "" : body.slice(spaceAt);
    forEachAttribute(attrText, (name, value) =>
      element.setAttribute(name, value === null ? "" : decodeEntities(value))
    );
    top().appendChild(element);
    if (!selfClosing && !VOID_TAGS.has(tagName)) stack.push(element);
  }

  if (stack.length !== 1) {
    const unclosed = stack.slice(1).map(node => node.tagName);
    throw new Error("unclosed element(s) in innerHTML: " + unclosed.join(", "));
  }
  return root.children.slice();
}

/** Walk an attribute string, yielding `(name, value)`; value is null if bare. */
function forEachAttribute(text, visit) {
  const re = /([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+)))?/g;
  let match;
  while ((match = re.exec(text)) !== null) {
    const value =
      match[2] !== undefined ? match[2]
      : match[3] !== undefined ? match[3]
      : match[4] !== undefined ? match[4]
      : null;
    visit(match[1], value);
  }
}

function parseCompound(text) {
  const compound = { tag: null, id: null, classes: [], attrs: [] };
  let consumed = 0;
  let match;
  SELECTOR_PART_RE.lastIndex = 0;
  while ((match = SELECTOR_PART_RE.exec(text)) !== null) {
    if (match.index !== consumed) {
      throw new Error("unsupported selector fragment in " + JSON.stringify(text));
    }
    consumed = SELECTOR_PART_RE.lastIndex;
    if (match[1]) compound.tag = match[1].toUpperCase();
    else if (match[2]) compound.id = match[2];
    else if (match[3]) compound.classes.push(match[3]);
    else if (match[4]) {
      compound.attrs.push({
        name: match[4],
        value: match[5] !== undefined ? match[5] : (match[6] !== undefined ? match[6] : match[7])
      });
    }
  }
  if (consumed !== text.length || consumed === 0) {
    throw new Error("unsupported selector " + JSON.stringify(text));
  }
  return compound;
}

function splitSelector(selector) {
  return selector
    .split(",")
    .map(part => part.trim())
    .filter(Boolean)
    .map(part =>
      part
        .split(/\s+/)
        .map(token => (token === "*" ? { tag: null, id: null, classes: [], attrs: [], universal: true } : parseCompound(token)))
    );
}

function matchesCompound(element, compound) {
  if (element.nodeType !== 1) return false;
  if (compound.tag && element.tagName !== compound.tag) return false;
  if (compound.universal) return true;
  if (compound.id && element.getAttribute("id") !== compound.id) return false;
  for (const cls of compound.classes) {
    if (!element.classList.contains(cls)) return false;
  }
  for (const attr of compound.attrs) {
    if (!element.hasAttribute(attr.name)) return false;
    if (attr.value !== undefined && element.getAttribute(attr.name) !== attr.value) return false;
  }
  return true;
}

function matchesChain(element, chain) {
  let index = chain.length - 1;
  if (!matchesCompound(element, chain[index])) return false;
  index -= 1;
  let node = element.parentNode;
  while (index >= 0) {
    if (!node) return false;
    if (matchesCompound(node, chain[index])) index -= 1;
    node = node.parentNode;
  }
  return true;
}

export class Element {
  constructor(tagName, ownerDocument) {
    if (!TAG_RE.test(tagName)) throw new Error("invalid tag name: " + tagName);
    this.nodeType = 1;
    this.tagName = tagName.toUpperCase();
    this.ownerDocument = ownerDocument;
    this.attributes = new Map();
    this.childNodes = [];
    this.parentNode = null;
    this.style = {};
    this.classList = new ClassList(this);
    this.dataset = new Dataset(this);
    this._listeners = new Map();
    this.hidden = false;
  }

  get children() {
    return this.childNodes.filter(node => node.nodeType === 1);
  }

  get firstElementChild() {
    return this.children[0] || null;
  }

  get id() {
    return this.getAttribute("id") || "";
  }

  set id(value) {
    this.setAttribute("id", value);
  }

  get className() {
    return this.getAttribute("class") || "";
  }

  set className(value) {
    this.setAttribute("class", value);
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }

  hasAttribute(name) {
    return this.attributes.has(name);
  }

  removeAttribute(name) {
    this.attributes.delete(name);
  }

  appendChild(node) {
    if (node.parentNode) node.parentNode.removeChild(node);
    node.parentNode = this;
    this.childNodes.push(node);
    return node;
  }

  append(...nodes) {
    nodes.forEach(node => this.appendChild(node));
  }

  removeChild(node) {
    const index = this.childNodes.indexOf(node);
    if (index === -1) return node;
    this.childNodes.splice(index, 1);
    node.parentNode = null;
    return node;
  }

  remove() {
    if (this.parentNode) this.parentNode.removeChild(this);
  }

  replaceChildren(...nodes) {
    this.childNodes.forEach(node => {
      node.parentNode = null;
    });
    this.childNodes = [];
    this.append(...nodes);
  }

  get textContent() {
    if (this.nodeType === 3) return this.data;
    return this.childNodes.map(node => node.textContent).join("");
  }

  set textContent(value) {
    this.childNodes = [];
    if (value !== "" && value !== null && value !== undefined) {
      this.appendChild(this.ownerDocument.createTextNode(String(value)));
    }
  }

  get innerHTML() {
    return this.childNodes.map(node => node.outerHTML).join("");
  }

  /**
   * Parse `html` into this element's children.
   *
   * This is a small HTML parser, not a browser. It handles the subset these
   * modules emit: elements, attributes (quoted or bare), self-closing and void
   * tags, text, and entity decoding. Anything it cannot parse raises rather than
   * being silently dropped, so a test cannot pass against markup the real code
   * would have failed to produce.
   */
  set innerHTML(html) {
    const parsed = parseFragment(this.ownerDocument, String(html));
    this.replaceChildren(...parsed);
  }

  addEventListener(type, handler) {
    if (typeof handler !== "function") return;
    if (!this._listeners.has(type)) this._listeners.set(type, []);
    const list = this._listeners.get(type);
    if (list.indexOf(handler) === -1) list.push(handler);
  }

  removeEventListener(type, handler) {
    const list = this._listeners.get(type);
    if (!list) return;
    const index = list.indexOf(handler);
    if (index !== -1) list.splice(index, 1);
  }

  dispatchEvent(event) {
    event.target = event.target || this;
    let node = this;
    while (node) {
      event.currentTarget = node;
      const list = node._listeners ? node._listeners.get(event.type) : null;
      if (list) list.slice().forEach(handler => handler.call(node, event));
      if (!event.bubbles || event.__stopped) break;
      node = node.parentNode;
    }
    return !event.defaultPrevented;
  }

  click() {
    return this.dispatchEvent(new MouseEvent("click", { detail: 1 }));
  }

  focus() {
    if (this.ownerDocument) this.ownerDocument.activeElement = this;
  }

  blur() {
    if (this.ownerDocument && this.ownerDocument.activeElement === this) {
      this.ownerDocument.activeElement = null;
    }
  }

  matches(selector) {
    return splitSelector(selector).some(chain => matchesChain(this, chain));
  }

  querySelectorAll(selector) {
    const chains = splitSelector(selector);
    const found = [];
    const walk = node => {
      node.children.forEach(child => {
        if (chains.some(chain => matchesChain(child, chain))) found.push(child);
        walk(child);
      });
    };
    walk(this);
    return found;
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }

  closest(selector) {
    let node = this;
    while (node && node.nodeType === 1) {
      if (node.matches(selector)) return node;
      node = node.parentNode;
    }
    return null;
  }

  get outerHTML() {
    const attrs = Array.from(this.attributes.entries())
      .map(([name, value]) => " " + name + '="' + value + '"')
      .join("");
    const inner = this.innerHTML;
    return "<" + this.tagName.toLowerCase() + attrs + ">" + inner + "</" + this.tagName.toLowerCase() + ">";
  }
}

export class TextNode {
  constructor(data, ownerDocument) {
    this.nodeType = 3;
    this.data = data;
    this.ownerDocument = ownerDocument;
    this.parentNode = null;
  }

  get textContent() {
    return this.data;
  }

  set textContent(value) {
    this.data = String(value);
  }

  get outerHTML() {
    return this.data;
  }
}

export class Document {
  constructor() {
    this.nodeType = 9;
    this.ownerDocument = this;
    this.documentElement = new Element("html", this);
    this.body = new Element("body", this);
    this.documentElement.appendChild(this.body);
    this.activeElement = null;
    this._listeners = new Map();
  }

  createElement(tagName) {
    return new Element(tagName, this);
  }

  createTextNode(data) {
    return new TextNode(data, this);
  }

  getElementById(id) {
    return this.documentElement.querySelector('[id="' + id + '"]');
  }

  querySelector(selector) {
    return this.documentElement.querySelector(selector);
  }

  querySelectorAll(selector) {
    return this.documentElement.querySelectorAll(selector);
  }

  addEventListener(type, handler) {
    if (!this._listeners.has(type)) this._listeners.set(type, []);
    const list = this._listeners.get(type);
    if (list.indexOf(handler) === -1) list.push(handler);
  }

  removeEventListener(type, handler) {
    const list = this._listeners.get(type);
    if (!list) return;
    const index = list.indexOf(handler);
    if (index !== -1) list.splice(index, 1);
  }

  dispatchEvent(event) {
    event.target = event.target || this;
    this._listeners.get(event.type)?.slice().forEach(handler => handler.call(this, event));
    return !event.defaultPrevented;
  }
}

export class Window {
  constructor(document) {
    this.document = document;
    this.location = { hash: "" };
    this.innerWidth = 1280;
    this.innerHeight = 800;
    this._listeners = new Map();
  }

  addEventListener(type, handler) {
    if (!this._listeners.has(type)) this._listeners.set(type, []);
    const list = this._listeners.get(type);
    if (list.indexOf(handler) === -1) list.push(handler);
  }

  removeEventListener(type, handler) {
    const list = this._listeners.get(type);
    if (!list) return;
    const index = list.indexOf(handler);
    if (index !== -1) list.splice(index, 1);
  }

  dispatchEvent(event) {
    this._listeners.get(event.type)?.slice().forEach(handler => handler.call(this, event));
    return !event.defaultPrevented;
  }

  print() {
    this.printCalls = (this.printCalls || 0) + 1;
  }
}

/** Build a document with the given body children and install DOM globals. */
export function installDom(bodyChildren = []) {
  const document = new Document();
  bodyChildren.forEach(child => document.body.appendChild(child));
  const window = new Window(document);

  globalThis.document = document;
  globalThis.window = window;
  globalThis.Event = DomEvent;
  globalThis.CustomEvent = DomEvent;
  globalThis.KeyboardEvent = KeyboardEvent;
  globalThis.MouseEvent = MouseEvent;
  globalThis.Element = Element;
  globalThis.Node = Object.assign(function Node() {}, { ELEMENT_NODE: 1, TEXT_NODE: 3 });

  return { document, window };
}