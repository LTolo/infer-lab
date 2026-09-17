# infer-lab Dashboard-Fix

Ersetzt das Einzel-View-Dashboard durch ein Control Centre mit Navigationsleiste
und eingebautem Last-Generator.

## Installation

ZIP über den bestehenden Projektordner entpacken — es werden genau zwei Dateien
überschrieben:

    clients/node/dashboard.js
    clients/node/public/index.html

PowerShell:

    cd $HOME\Downloads
    Expand-Archive -Path .\infer-lab-dashboard-fix.zip -DestinationPath .\infer-lab -Force

Neu starten:

    cd $HOME\Downloads\infer-lab
    python scripts\run_stack.py --port 8100 --dashboard-port 3100

Dann **nur noch** http://127.0.0.1:3100 öffnen.

## Was neu ist

| Tab | Inhalt |
|---|---|
| **Live** | Durchsatz, Latenz-Perzentile, KV-Auslastung als Live-Charts |
| **Load** | Last-Generator direkt in der UI — kein zweites Terminal mehr nötig |
| **Playground** | Prompt senden, Tokens und Timings pro Request sehen |
| **Engine Stats** | Scheduler-Zähler, KV-Belegung, Prefix-Cache-Hit-Rate |
| **Kernels** | Backend-Verfügbarkeit mit Begründung |
| **Raw Metrics** | Prometheus-Expositionstext im Original |

## Behobene Probleme

* **Leere Charts ohne Erklärung** — bei null Requests erscheint jetzt ein Hinweis
  statt einer wortlos flachen Linie.
* **Zweites Terminal für Last** — entfällt. Der Dashboard-Prozess erzeugt den
  Traffic selbst (Tab *Load*).
* **`{"detail":"Not Found"}`** — die API hat keine Root-Route. Das Dashboard
  proxied alle Endpunkte, es genügt ein einziger Port im Browser.
* **Absturz durch unbehandelte Promise-Rejection** — ein fehlgeschlagener
  Load-Request hätte den gesamten Dashboard-Prozess beenden können; jetzt
  abgefangen.

## Getestet

Gegen einen Mock der infer-lab-API verifiziert: alle 7 Routen liefern 200,
Last-Generator 1666 Requests in 5 s bei 0 Fehlern, Worker fahren beim Stoppen
sauber auf 0 herunter, Prozess bleibt stabil.
