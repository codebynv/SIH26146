/**
 * Minimal DOM construction helpers (Step 9B Phase 3).
 *
 * Vanilla JavaScript with no external libraries. The Phase 3 sections build
 * their markup through `el()` rather than by assigning an HTML string, which
 * means every value reaches the document as a text node or an attribute and is
 * therefore escaped by construction: a rationale containing `<` or a quote
 * cannot break out into markup.
 *
 * `el()` is the only element factory these sections use, so the rendered DOM is
 * inspectable in a test with the same API a browser provides.
 */

/**
 * Create an element.
 *
 * `props` keys are either plain attribute names (`class`, `role`, `type`,
 * `hidden`) or a `data`/`aria` prefixed name (`data-relationship-id`,
 * `aria-expanded`). A `null` or `undefined` value omits the attribute; `false`
 * omits it and `true` writes the empty string, which is how boolean HTML
 * attributes such as `hidden` behave.
 *
 * `children` may contain nodes, strings, or `null`. Strings become text nodes.
 */
export function el(tag, props, children) {
  const node = document.createElement(tag);
  Object.keys(props || {}).forEach(name => {
    const value = props[name];
    if (value === null || value === undefined || value === false) return;
    node.setAttribute(name, value === true ? "" : String(value));
  });
  const list = Array.isArray(children) ? children : [children];
  list.forEach(child => {
    if (child === null || child === undefined || child === false) return;
    node.appendChild(
      typeof child === "string" || typeof child === "number"
        ? document.createTextNode(String(child))
        : child
    );
  });
  return node;
}

/** Create a text node. */
export function text(value) {
  return document.createTextNode(String(value));
}

/** Replace an element's children with `children`. */
export function replaceChildrenWith(node, children) {
  node.replaceChildren();
  const list = Array.isArray(children) ? children : [children];
  list.forEach(child => {
    if (child === null || child === undefined || child === false) return;
    node.appendChild(child);
  });
  return node;
}

/** Remove every child of an element. */
export function clear(node) {
  node.replaceChildren();
  return node;
}

/**
 * Create a `<span>` whose content is an id, rendered monospaced.
 *
 * The full id is always carried in the `title` attribute so a truncated
 * display value is never the only value available to the investigator.
 */
export function idSpan(value, className, truncate) {
  const shown =
    typeof truncate === "number" && String(value).length > truncate
      ? String(value).slice(0, truncate) + "…"
      : String(value);
  return el("span", { class: className || "mono", title: String(value) }, shown);
}