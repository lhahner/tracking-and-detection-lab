## 1. Dashboard
| No. | Subject | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Progress |
| --- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| 1   | Foo     | Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. | 100%     |
### 1.2 Activity Chart
Apply by "Open Command Platte" > "Insert Graph"
```chart
type: bar
labels: [1,2,3,4,5]
series:
  - title: Test
    data: [1,2,3,4,5]
tension: 0.2
width: 50%
labelColors: false
fill: false
beginAtZero: false
bestFit: false
bestFitTitle: undefined
bestFitNumber: 0
```

### 1.1 Activity Heat-map
Apply by adding a Dataview in that format defined here: https://github.com/Richardsl/heatmap-calendar-obsidian/blob/master/EXAMPLE_VAULT/Overview.md 
```dataviewjs

```
```dataviewjs
dv.span("**Example Map**")

const calendarData = {
    year: 2025, // optional, remove this line to autoswitch year
    colors: {
        red: ["#ff9e82","#ff7b55","#ff4d1a","#e73400","#bd2a00",]
    },
    entries: []
}


    calendarData.entries.push({
        date: "2025-08-09",
        intensity: 1,
        content: await dv.span(`[](${"2025-08-09"})`), //for hover preview
    })

renderHeatmapCalendar(this.container, calendarData)

```
