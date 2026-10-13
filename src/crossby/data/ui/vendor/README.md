# Vendored frontend assets

Committed rather than fetched so `pip install crossby` yields a working offline
UI with no npm, no build step, and no CDN dependency at runtime.

| File | Package | Version | License |
|------|---------|---------|---------|
| `xterm.js`, `xterm.css` | [`@xterm/xterm`](https://www.npmjs.com/package/@xterm/xterm) | 5.5.0 | MIT |
| `addon-fit.js` | [`@xterm/addon-fit`](https://www.npmjs.com/package/@xterm/addon-fit) | 0.10.0 | MIT |

To upgrade, re-fetch at the new version and update the table:

```sh
V=5.5.0; F=0.10.0; D=src/crossby/data/ui/vendor
curl -sSfo "$D/xterm.js"     "https://cdn.jsdelivr.net/npm/@xterm/xterm@$V/lib/xterm.js"
curl -sSfo "$D/xterm.css"    "https://cdn.jsdelivr.net/npm/@xterm/xterm@$V/css/xterm.css"
curl -sSfo "$D/addon-fit.js" "https://cdn.jsdelivr.net/npm/@xterm/addon-fit@$F/lib/addon-fit.js"
```
