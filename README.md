<div align="center">

# XFlagCV
### Flag Football Player Detection, Tracking &amp; Team Identification

*A computer vision pipeline for tracking players and assigning teams from drone footage*

***Andrew Nguyen***

<br />

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-YOLO11-brightgreen.svg)](#)
[![Tracker](https://img.shields.io/badge/Tracker-BoT--SORT-orange.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<br />

[Overview](#overview) | [Architecture](#architecture) | [Methodology](#methodology) | [Evaluation](#evaluation) | [Results](#results) | [Limitations](#limitations) | [Installation](#installation) | [Usage](#usage) | [Future Work](#future-work)

</div>

---

## Overview

<div align="center">
  <img src="./assets/clip.gif" alt="Demo" width="600" />
</div>

**XFlagCV** is a computer vision framework for detecting, tracking, and identifying players from drone footage of flag football games. Built on a custom trained YOLO11 detector, BoT-SORT tracking, and SigLIP based team classification, the pipeline turns raw aerial video into player tracked and team labeled footage — without relying on wearable sensors,  jersey numbers, or manual annotation.

Flag football presents a uniquely difficult case for player tracking. 
Unlike broadcast american football with stabilized camera work, and unified jerseys/numbers, flag football involves drone footage that can be wide, distant, and unstabilized.

Flag football also naturally packs 10+ players into a tight cluster with teammates wearing identical jerseys leaving no individual visual cues most tracking systems rely on.

### What It Does

- 🎯 **Detects players** 
- 🔗 **Tracks players frame-to-frame** 
- 🎽 **Assigns teams automatically** 
- 🎥 **Outputs labeled video** 

### What Makes It Hard

- 🤼 **Dense payer contact** 
- 🚁 **Random drone orientation**

---

## Architecture

<!-- pipeline diagram + one-line-per-stage breakdown goes here -->

---

## Methodology

<!-- detection / tracking / team assignment / stitching — the "how and why" -->

---

## Evaluation

<!-- training curves, precision/recall/mAP numbers, presnap accuracy -->

---

## Results

<!-- demo gif #2 or stills showing the full pipeline in action -->

---

## Limitations

<!-- known failure modes, honestly framed, with citations -->

---

## Installation

<!-- venv setup, requirements.txt, weight download -->

---

## Project Structure

<!-- repo file tree -->

---

## Usage

<!-- runnable code snippet -->

---

## Future Work

<!-- planned features, cited prior art -->

---

## Changelog

<!-- dated log of what's been done / what's next -->

---

## References

<!-- your two cited papers -->

</div>
