/* Progressive keyboard support. Never intercept typing or browser shortcuts. */
(function () {
  'use strict';
  function click(id) { const el = document.getElementById(id); if (el && !el.disabled) el.click(); }
  function input(id) {
    const el = document.getElementById(id); if (!el) return;
    const target = el.matches('input') ? el : el.querySelector('input,[role="combobox"]');
    if (target) { target.focus(); if (target.select) target.select(); }
  }
  document.addEventListener('keydown', function (event) {
    const modal = document.querySelector('#ex-open-trace[open]');
    if (modal) {
      if (event.key === 'Escape') { event.preventDefault(); click('ex-close-trace'); return; }
      if (event.key === 'Tab') {
        const fields = Array.from(modal.querySelectorAll('button,input,[tabindex="0"],[role="combobox"],a[href]'))
          .filter(el => !el.disabled && el.getClientRects().length);
        if (fields.length) {
          const first=fields[0],last=fields[fields.length-1];
          if (event.shiftKey && (document.activeElement===first || !modal.contains(document.activeElement))) { event.preventDefault();last.focus(); }
          else if (!event.shiftKey && (document.activeElement===last || !modal.contains(document.activeElement))) { event.preventDefault();first.focus(); }
        }
      }
      return;
    }
    if (event.ctrlKey || event.metaKey || event.altKey || event.defaultPrevented) return;
    const el=event.target;
    if (el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName) || el.getAttribute('role')==='combobox')) return;
    const key=event.key.toLowerCase();
    const actions={'/':()=>input('ex-metric'),'f':()=>input('ex-row-search'),'b':()=>click('ex-back'),
      'r':()=>click('ex-reset-time'),'o':()=>click('ex-open'),'?':()=>click('ex-help-button')};
    if (actions[key]) { event.preventDefault(); actions[key](); }
  });
})();
