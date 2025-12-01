## vault-schreibvorlage
*Gibt ein Initaliale Projektsturktur vor die Programme, deren Dokumentation
für Forschungs und Entwicklungs zwecken beinhaltet. Eignet sich
besonders für Studierende und Entwickler.*
### Ordner Struktur
#### /docs
Hauptsächlich für Dokumentations- und Schreibarbeiten, sollte benutzt werden
um Thesen, Seminar oder Projektarbeiten zu schreiben.

| folder | content |
| ------ | ------- |
| `/diagrams` | Kategorarisiert nach, *e.g.: uml, flow-chart etc.* |
| `/references` | Kategorisiert nach, *e.g.: literature, transcripts etc.* |
| `/sections` | Sections für notes, geschrieben in Markdown, Html oder Latex |
#### /sections
- Die Ordnung der Unterkapitel innerhalb einer section beginnt mit der Nummerierung 1. und wird je nach tiefe fortgesetzt mit 1.1 usw. 
- Jede section sollte ein Inhaltsverzeichnis mit links haben. Diese können über das Plugin "Table of Contents" eingefügt werden.
---
## `plugin` Verwendung
Aktuell werden folgende Plugins mit eingebunden:

| Plugin            | Verwendung                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------- |
| Charts            | Kann verwendet werden um Diagram in jeglicher Form, wie zum Beispiel Linien, Balken usw.                      |
| Heatmap           | Die Heat-map kann z.b. verwendet werden um Aktivitäten zu tracken, sie muss über dataviewjs eigebunden werden |
| Dataviewjs        | Dataviewjs ermöglich inline rendering von JavaScript und die Referenz auf den Obsidian Code.                  |
| Table Of Contents | Kann verwendet werden um Inhaltsverzeichnisse aus den Überschriften einer Markdown Datei zu generieren.       |
