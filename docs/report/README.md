# Schreibvorlage
## 1.1 Einführung
Diese Repository soll ein template bereitstellen welches allgemein gängige verwendete 
Latex Grundlagen für ein deutsche Hausarbeit implementiert. Weiterhin sollen Ordern-
strukturen als best-practice dargelegt werden die es ermöglichen sollen die Arbeit
Sauber und Übersichtlich zu halten.

## 2.1 Ordnerstruktur
### 2.1.1 Ordnerstrukturen von LaTex-Hausarbeiten

Der Einstiegspunkt für alle Resourcen sollte von der `main.tex` ausgehen.
Es werden externe Softwarepackete, Quellenangaben und Kapitel in Subordner ausgelagert
und in der main.tex eingebunden.

### 2.1.2 Der Ordner `sections`

In diesem Ordner wird für jedesn Kapitel eine `.tex` Datei angelegt die den vollständigen
Inhalt dieses Kapitels enthält. Der Name der Datei sollte gleichbedeutend mit dem Titel
des Kapitels sein.

### 2.1.3 Der Ordner `references`

In diesem Ordner finden sich die verwendete Literatur, sowie das eingebundene Literatur-
verzeichnis als `.tex` Datei.
Zusätzlich enthält dieser Ordner die Subordner summaries und transscirpts, welche
Zusammenfassungen der Literatur und Mitrschriften von Vorlesungen oder sonstiges 
beinhalten.

### 2.1.4 Der Ordner `packages`

Dieser Ordner enthält eine `.tex` Datei welche alle externen Softwarepakete beinhaltet die
in der `main.tex` eingebunden werden.

### 2.1.5 Der Ordner `bins`

Der binary Ordner enthält alle binär-dateien des Projektes, also etwa `.png`, `.jpg` oder
`.pdf` Dateien.
