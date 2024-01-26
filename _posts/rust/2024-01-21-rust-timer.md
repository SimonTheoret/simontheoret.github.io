---
layout: post
title:  "Building a GUI pomodoro in Rust"
date:   2024-01-21 12:49:07 -0500
project: timer
background: gui.jpeg
---
Project repo [here](https://github.com/SimonTheoret/timer)

## Motivation
Durign the summer of 2023, I planned to learn my first somewhat low-level language. I'd
already heard of Rust as a blazingly fast, safe and up and coming programming language.
I read [the Book](https://doc.rust-lang.org/book/title-page.html) and wanted to
implement my first program in Rust to get to know the language. I had also some hopes to
analyze my own behaviors with the help of the collected data.


## Overview of the Pomodoro
If you don't know what the pomodoro technique is, here is the link to the wiki article:
[Pomodoro Technique](https://en.wikipedia.org/wiki/Pomodoro_Technique). I build a GUI
with the help of the [egui](https://github.com/emilk/egui) library and used threads to
count down and update the interface. It collects data _only_ in the pomodoro directory
(see below).

## Features
- _Simple_ and easy to use gui.
- Notification at end of the session
- Sound played at end of the session
- Register the pomodoro sessions in the data directory (`$XDG_DATA_HOME` or
  `$HOME/.local/share/pomodoro` on Linux, `FOLDERID_RoamingAppData\pomodoro` on Windows and
  `$HOME/Library/Application/pomodoro` Support on macOS). The data is saved in the `csv`
  format.

## Installation
To install the timer from the source, you need Rust installed on your system. To install
it, follow [this](https://www.rust-lang.org/tools/install) link.
1. Clone the repo:

        git clone https://github.com/SimonTheoret/timer

2. Change the current directory adnd build the binary:

        cd timer
        cargo build --release

3. Move the binary to a `bin` folder:

        mv ./target/release/timer_rust /bin
4.(**Optional**) Give an alias to the timer in your `.zshrc` or `.bashrc` file:

        alias timer='timer_rust'

