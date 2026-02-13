# Inventory Attribution Analysis

## Overview

This project builds a structured inventory attribution model to decompose weekly US crude oil inventory changes into their primary drivers: imports, exports, and refinery runs.

Rather than relying on headline inventory numbers alone, this framework identifies whether weekly balance shifts are driven by supply inflow, export pull, or refinery demand dynamics.

---

## Methodology

### 1. Flow-Based Attribution

Inventory changes are analysed using week-on-week changes in:

* Crude imports
* Crude exports
* Refinery crude inputs

### 2. Dominant Driver Logic

* Largest absolute WoW flow change identified
* Confidence scored based on magnitude separation
* Offset detection for near two-way trade flows

### 3. Conviction Model

Conviction integrates:

* Inventory magnitude
* Driver clarity
* Offset adjustment

### 4. Statistical Context

* Rolling 52-week z-scores identify statistically unusual flow movements

---

## Key Features

* Multi-layer commentary engine
* Structural vs timing-driven interpretation
* Bias and conviction scoring
* Interactive dashboard with synced crosshair
* Automated weekly update via EIA API

---

## Tools Used

* Python
* Pandas
* Plotly
* EIA API
* Time-series analysis
* Statistical normalization (z-score)

---

## Sample Dashboard

👉 View Interactive Dashboard:
[https://ireneteo98.github.io/inventory-attribution-analysis/](https://ireneteo98.github.io/inventory-attribution-analysis/)

---

## Why This Matters

Headline inventory numbers alone can mislead.
This framework demonstrates a systematic, balance-driven approach to identifying the true mechanical drivers behind crude oil inventory movements and improving directional interpretation.

---
