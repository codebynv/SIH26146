class Store {
  constructor(initialState = {}) {
    this._state = { ...initialState };
    this._listeners = new Set();
  }
  
  get state() {
    return this._state;
  }
  
  setState(partial) {
    this._state = { ...this._state, ...partial };
    this._notify();
  }
  
  subscribe(listener) {
    this._listeners.add(listener);
    return () => this._listeners.delete(listener);
  }
  
  _notify() {
    this._listeners.forEach(l => l(this._state));
  }
}

export const store = new Store({
  overview: null,
  signals: [],
  evidence: [],
  paths: [],
  behavior: null,
  provenance: null,
  loading: true,
  errors: {}
});

export function select(selector) {
  return selector(store.state);
}
